#!/usr/bin/env bash
# Hermetic topology check: four CX7 edges pass, a missing edge or direct-RoCE mode fails.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
TP=4 TOPOLOGY=switchless-ring COMM=nccl
WORKER=user@worker1 WORKER2=user@worker2 WORKER3=user@worker3
WORKER_WEIGHTS=copy
MASTER_ADDR=192.0.2.10 MASTER_PORT=29551 SOCKET_IFNAME=""
HEAD_CX7_IB=hca0,hca1 WORKER_CX7_IB=hca0,hca1
WORKER2_CX7_IB=hca0,hca1 WORKER3_CX7_IB=hca0,hca1
HEAD_GID=3 WORKER_GID=3 WORKER2_GID=3 WORKER3_GID=3
NCCL_CHANNELS=4 NCCL_DEBUG=WARN
log() { :; }
warn() { :; }
die() { printf '%s\n' "$*" >&2; exit 1; }
source scripts/nodes.sh

node_inventory() {
  printf '%s\n' 'default eth0' \
    'p0 hca0 10.10.1.1/24 3' 'p1 hca1 10.10.4.1/24 3'
}
worker() {
  local rank=$1; shift
  printf '%s\n' 'default wlan0' 'master ok'
  case "$rank" in
    1) printf '%s\n' 'p0 hca0 10.10.1.2/24 3' 'p1 hca1 10.10.2.1/24 3' ;;
    2) printf '%s\n' 'p0 hca0 10.10.2.2/24 3' "p1 hca1 10.10.${BROKEN:-3}.1/24 3" ;;
    3) printf '%s\n' 'p0 hca0 10.10.3.2/24 3' 'p1 hca1 10.10.4.2/24 3' ;;
  esac
}
link_info() { printf '%s\n' '192.0.2.10 eth0 - -'; }

check_workers
[[ "$(configured_workers | tr '\n' ' ')" == '1 2 3 ' ]]
detect_links
[[ "${NODE_HCAS[0]}" == hca0,hca1 && "${NODE_HCAS[3]}" == hca0,hca1 ]]
[[ "$(rank_nccl_env 0)" == *'NCCL_SWITCHLESS_RING_ONLY=1'* ]]
[[ "$(rank_nccl_env 0)" == *'NCCL_ALGO=Ring'* ]]
if ( BROKEN=9; detect_links ) >/dev/null 2>&1; then
  echo 'missing 2-3 edge was accepted' >&2; exit 1
fi
if ( COMM=roce; check_workers ) >/dev/null 2>&1; then
  echo 'direct RoCE mode was accepted on a switchless ring' >&2; exit 1
fi
tmp_cfg=$(mktemp -d)
trap 'rm -rf "$tmp_cfg"' EXIT
mkdir "$tmp_cfg/scripts"
cp scripts/config.sh "$tmp_cfg/scripts/config.sh"
(
  source "$tmp_cfg/scripts/config.sh"
  [[ "$CONTAINER_NAME" == glm53-flash-tf-tp4 && "$PORT" == 8890 ]]
  [[ "$MEMORY_RESERVE_GIB" == 20 && "$KV_POOL_GIB" == 24 && "$SPLIT" == 0 ]]
  [[ "$TF_GLM_HC_EXCHANGE" == gather && "$TF_GLM_PREFILL_OVERLAP" == 0 ]]
  [[ "$STATE_DIR" == "$HOME/.local/state/glm53-tensorfold-tp4" ]]
)
(
  SPLIT=1
  source "$tmp_cfg/scripts/config.sh"
  [[ "$SPLIT" == 1 && "$TF_GLM_HC_EXCHANGE" == gather && "$TF_GLM_PREFILL_OVERLAP" == 2 ]]
  [[ "$(env | sed -n 's/^TF_GLM_HC_EXCHANGE=//p')" == gather ]]
)
(
  SPLIT=1 TF_GLM_HC_EXCHANGE=ring
  source "$tmp_cfg/scripts/config.sh"
  [[ "$SPLIT" == 1 && "$TF_GLM_HC_EXCHANGE" == ring && "$TF_GLM_PREFILL_OVERLAP" == 2 ]]
)
(
  TOPOLOGY=full-mesh
  source "$tmp_cfg/scripts/config.sh"
  [[ "$SPLIT" == 1 && "$TF_GLM_HC_EXCHANGE" == p2p ]]
)
(
  PORT=9001 MEMORY_RESERVE_GIB=22
  source "$tmp_cfg/scripts/config.sh"
  [[ "$PORT" == 9001 && "$MEMORY_RESERVE_GIB" == 22 ]]
)
printf '%s\n' 'TP4 ring topology tests passed'
