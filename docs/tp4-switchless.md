# TensorFold GLM-5.3 Flash on four switchless DGX Sparks

This fork adds a **four-rank, one-model TensorFold** launch path to MiaAI's GLM-5.3 Flash recipe. It uses the existing TP-N model patches 0066–0068, four direct CX7 links in a cycle, and a pinned switchless NCCL build. It does not use vLLM in the serving path.

## Status

The TP4 model code is supplied by MiaAI's 0066–0068 patches. This fork adds the four-rank launcher, topology validation and patched NCCL ring settings. Patch 0071 adds an opt-in owner-row FP32 exchange on the physical ring. Its isolated 8192-row communication benchmark is recorded in [the microbenchmark](ring-row-exchange-20261005/README.md); that measurement is not an end-to-end serving result. The published two- and three-Spark numbers below use different deployments.

## Physical layout

Rank order follows the physical cycle:

```text
rank 0 (head) ── rank 1
      │             │
rank 3        ── rank 2
```

Each edge may have two rails. Rank 0–2 and rank 1–3 have no direct cable. SSH and TensorFold's master rendezvous use the management LAN; the model collectives use CX7. The launcher confirms all four cycle edges have a shared CX7 subnet, permits the two missing diagonals and refuses a missing cycle edge.

TensorFold must run with `COMM=nccl` on this layout. MiaAI's one-shot `RoceComm` builds direct rank-to-rank queue pairs; the two diagonal pairs cannot form such a connection. The patched NCCL library runs the ring collectives over the four direct edges. The launcher refuses a missing or different NCCL library on any rank, verifies `LD_PRELOAD` inside the image, and sets `TF_NCCL_LIB` to the mounted library for TensorFold's own loader.

The conservative default is `SPLIT=0`; an enabled split defaults to `TF_GLM_HC_EXCHANGE=gather`. On the measured four-Spark site, `SPLIT=1` and 8192-row chunks already passed long-prompt, tool-call and decode checks. Patch 0071 lets that site set `TF_GLM_HC_EXCHANGE=ring`: each rank sends only the FP32 partials for the rows owned by the destination, forwarding the opposite-rank block through one physical neighbor. It keeps rank-order FP32 summation and uses the existing NCCL all-gather for finished BF16 rows. `ring` requires TP4, NCCL, the switchless ring and `SPLIT=1`; direct `p2p` remains refused on the ring. To roll back the exchange alone, restore `TF_GLM_HC_EXCHANGE=gather` and restart all four ranks.

## Prepare a site

1. Put the same patched NCCL `libnccl.so.2` on all four machines. The library from a proven four-Spark switchless vLLM deployment can be reused; the launcher checks its SHA256 on every rank.
2. Copy `scripts/local.tp4.example` to `scripts/local.sh` on rank 0 and fill in the real SSH targets, management address, two HCA names and RoCE v2 GID for each rank. `scripts/local.sh` is local state and must not be committed.
3. Make the TensorFold checkpoint and DFlash2 model accessible on rank 0. `scripts/prepare.sh` can copy missing files to each worker; this may move hundreds of GiB. Use `WORKER_WEIGHTS=copy` unless an NFS export is reachable from **every** worker. In particular, a switchless diagonal worker may not reach the head over CX7.
4. Stop the existing GPU inference containers on all four Sparks. The TP4 launcher refuses to start while another GPU container is running.

If the pinned published image is unreachable but the CUDA 13.0 vLLM base image is already available locally, build the fallback image from this fork:

```bash
git clone --depth 1 --branch v0.6.0 https://github.com/ashhart/TensorFold.git tf-src
docker build -f deploy/Dockerfile.local-base -t tensorfold-glm53:v0.6.0 \
  --build-arg PATCHES_HASH="$(bash -c 'source scripts/config.sh; image_hash')" .
```

Load the resulting image on each worker (`docker save` piped through SSH to `docker load`). This fallback image uses `/usr/bin/env` as its launch entrypoint; set `CUDA_HOST_INCLUDE=/usr/local/cuda/include` in `scripts/local.sh` when that path contains the matching CUDA 13.0 development headers on every Spark. The launcher mounts those headers inside each container for first-run CUDA extension builds.

From rank 0:

```bash
bash tests/test_tp4_ring.sh
DRY_RUN=1 ./start-tp4.sh
./start-tp4.sh
curl -fsS http://127.0.0.1:8890/v1/models
./start-tp4.sh stop
```

`./start-tp4.sh restart` stops and restarts the four TensorFold ranks after preflight. Workers start in rank 3, 2, 1 order; rank 0 starts last. The TP4 container name, state directory and API port are separate from MiaAI's TP2 recipe. Port 8890 is the default; choose another free port with `PORT=...`.

### Incremental owner-ring image on an already installed TP4 site

When every Spark already has the qualified `tensorfold-glm53:v0.6.0` image, patch 0071 can be applied locally on each machine without transferring the complete 31 GB image. Copy `patches/0071-glm-tp4-owner-ring.patch` and `deploy/Dockerfile.owner-ring-incremental` to each Spark, then run from this repository on **each** node:

```bash
hash=$(bash -c 'source scripts/config.sh; image_hash')
docker build -t tensorfold-glm53:owner-ring-20261005 \
  --build-arg PATCHES_HASH="$hash" \
  -f deploy/Dockerfile.owner-ring-incremental patches
```

Check the `tf.patches` label, the SHA-256 of the installed `tensorfold/cuda/comm.py` and `tensorfold/families/glm5_next/cuda/hcsplit.py`, and the existing checkpoint revision on every node before starting. The image labels and both source hashes must match across ranks. The old image tag remains available. On the measured site, the site file sets `IMAGE=tensorfold-glm53:owner-ring-20261005`, `SPLIT=1` and `TF_GLM_HC_EXCHANGE=ring`; `gather` is the one-variable fallback. See [the measured rollout](owner-ring-rollout-20261005.md) for the exact A/B and checks.

For a permanent deployment in `%h/glm53-tf-tp4`, copy `deploy/glm53-tf-tp4*.service` and `deploy/glm53-tf-tp4-watch.timer` to `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, then enable the service and timer. The watchdog probes `/health` twice before restarting the four ranks. Keep any previous inference unit and watchdog disabled while TensorFold owns the GPUs.

Defaults for the first validation run: `CONTEXT=1048576`, `PARALLEL=4`, `KV=fp8`, `KV_POOL_GIB=24`, `MEMORY_RESERVE_GIB=20`, `DRAFTER=dflash2`, `SPLIT=0`. These are conservative **starting settings**, not measured optima. TensorFold's capacity gate may reduce the context if the actual free-memory budget is smaller.

## Validation before production use

Use the same checkpoint and prompts on MiaAI TP2 and this TP4 path. First confirm one greedy completion, DFlash2 drafted-versus-serial equality, API tools and image requests. Then measure:

- 8K, 32K and 128K cold prefill and time to first token;
- single-stream prose and code decode at 32K context;
- aggregate throughput and p95 first-token latency at 4 and 8 concurrent requests;
- `/health` pool tokens and minimum free memory on each rank under a long prompt;
- a 195K needle and a 1M-token needle, followed by a sustained mixed-workload run.

Record the model revision, image hash, NCCL SHA, GPU clocks, context, sampling policy and both per-request and aggregate throughput. Run TP2 and TP4 in separate windows because both use the same four GPUs.

## What four Sparks may improve

MiaAI reports 60.4 prose / 114.7 code tokens/s for one request and 108.8 prose / 227.9 code tokens/s for four requests on TP2. Its TP3 measurements are 77.6 / 104.3 for one request and 146.2 / 169.6 for four. The third Spark improved prose and total prose throughput, while code slowed in those tests. Cold prefill at 32K stayed near 2,000 tokens/s on both layouts.

TP4 divides the same weights among four ranks, leaving more memory per Spark for a shared KV pool and multiple long conversations. That is its clearest expected advantage. A larger pool can reduce queuing or eviction when several long agent sessions run together. Single-request decode and cold prefill may improve, stay flat or regress: every layer communicates across more ranks, and the ring has less cross-section bandwidth than a full mesh. No TP4 speed multiplier is claimed until matched measurements exist.

The checkpoint, quantization, model behavior and per-request maximum context are unchanged. TP2 already offers the model's 1M-token window; TP4's likely context benefit is **more simultaneous long sessions**, subject to measured memory headroom.

## Rollback

`./start-tp4.sh stop` stops this fork's four TensorFold containers. Restore the previous vLLM service with its existing systemd unit or launcher, then check its `/health` and one completion. This deployment uses port 8890 for the model API; the existing port 8888 console gateway forwards to it. The TensorFold fork does not alter the vLLM image, weights or systemd unit.
