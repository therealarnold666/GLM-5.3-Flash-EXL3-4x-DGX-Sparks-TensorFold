# The two Sparks: the worker over ssh, and each node's RoCE link found from the route between them.
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.

# Run a command on the worker (rank 1). Key-based ssh only, no prompts.
worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$WORKER" "$@"; }

# The worker's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank 1.
worker_hf_cache() {    # WORKER_HF_CACHE wins: a worker whose cache is not its HF_HOME (e.g. a shared models disk)
  if [[ -n "${WORKER_HF_CACHE:-}" ]]; then echo "$WORKER_HF_CACHE"; return; fi
  worker 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing".
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}

# WORKER_WEIGHTS=nfs: rank 1 mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on the worker.
# ensure_nfs_volume creates it (or checks the one there names the same export); worker_nfs <cmd...> runs a command in
# a throwaway container with the volume at /hf.
ensure_nfs_volume() {
  local server="${NFS_SERVER:-$HEAD_ADDR}" have
  have=$(worker "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on the worker"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "the worker's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH"
  fi
}
worker_nfs() { worker "docker run --rm --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }

need_worker() {
  [[ -n "${WORKER:-}" ]] || die "WORKER is not set: put WORKER=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker true 2>/dev/null || die "cannot ssh to $WORKER without a password: set up key-based ssh (ssh-copy-id $WORKER)"
}

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: every RoCE device of this node on the link's subnet (the CX7's second port too, when it
# is up and addressed there) whose RoCE v2 IPv4 GID sits at the same index, the link's own device first.
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  echo "$hcas"
}

# The same function on the worker (its definition is sent over ssh).
worker_link_info() { worker "$(declare -f link_info); link_info $1"; }

# HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the two
# ranks talk over (NCCL and the rendezvous). FABRIC_PEER overrides the worker's link address when WORKER is reached
# over another network.
# cx7_peer: the worker's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a WORKER given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}
detect_link() {
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # both CX7 ports when both are cabled and addressed: a prompt chunk's all-gather is ~1.8x faster on two rails
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
}

# NCCL over the RoCE link, per rank: that rank's netdev, RoCE devices (both rails) and GID index, 4 channels.
# Measured between the Sparks: a decode-sized all-gather 32 us (80 with NCCL's defaults) and a prompt chunk's 32 MB
# 1.9 ms on two rails (3.5 on one). NCCL's other knobs (protocols, buffer sizes, QPs, NCCL_NET=IB and the like) were no
# better, and some of them kept NCCL on one rail.
nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  echo "-e NCCL_SOCKET_IFNAME=$dev -e NCCL_IB_HCA=$hcas -e NCCL_IB_GID_INDEX=$gid" \
       "-e NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-4} -e NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-4}" \
       ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG}
}
