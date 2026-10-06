# TensorFold GLM-5.3 Flash on four switchless DGX Sparks

This repository targets a **four-rank, one-model TensorFold** deployment on a switchless DGX Spark ring. It builds on Mia AI's GLM-5.3 Flash recipe and TP-N model patches 0066–0068, with four direct CX7 links in a cycle and a matching switchless NCCL build. It does not use vLLM in the serving path.

## Status

The TP4 model code is supplied by Mia AI's 0066–0068 patches. This fork adds the four-rank launcher, topology validation and patched NCCL ring settings. Patch 0071 adds an opt-in owner-row FP32 exchange on the physical ring. Its isolated 8192-row communication benchmark is recorded in [the microbenchmark](ring-row-exchange-20261005/README.md); that measurement is not an end-to-end serving result.

An end-to-end four-Spark [sparkDash run](sparkdash-ablit-20261006/README.md) now reports C1–C4 decode and 8K–256K cold prefill. It used a locally built image and Ablit checkpoint, so the measurements do not describe the conservative checkout defaults or isolate the value of a fourth Spark.

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

The measured site's subsequent [four-HCA RigMark A/B](4hca-rigmark-20261005/README.md) uses the two direct RoCE functions toward each neighbour and a separately built NCCL 2.30.7 extended-GID library. It raises 32K/64K cold prefill by about 21%/20% over a two-HCA arm using the same library. The original hardened two-GID library remains the conservative option; the extended switches in `scripts/nodes.sh` are opt-in.

For this same measured site, [mixed-load tuning](fill-budget-20261006/README.md) kept the single-direction owner-row route and raised `TF_GLM_FILL_ROWS` from 1024 to 2048 in the head's untracked `scripts/local.sh`. With three decodes active, 32K/64K cold prefill TTFT improved by about 10% in three matched trials per depth; decode SSE-gap p95 rose by 2.5–3.1 ms. The generic default remains 1024 rows for other sites.

An opt-in [short-prompt admission patch](short-prompt-priority-20261006/README.md) prioritizes a later short prompt at the next chunk boundary when a long prompt is filling. On the measured site it reduced the short request's mixed-load first-token median from 10.24 to 3.77 seconds, while C4 short-only aggregate throughput remained within 2%. It cannot interrupt the current 8192-row forward; `SHORT_PROMPT_ROWS=0` preserves the old scheduler.

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

First confirm one greedy completion, DFlash2 drafted-versus-serial equality, API tools and image requests. For a comparison against another deployment, use the same checkpoint and prompts in both runs. Then measure:

- 8K, 32K and 128K cold prefill and time to first token;
- single-stream prose and code decode at 32K context;
- aggregate throughput and p95 first-token latency at 4 and 8 concurrent requests;
- `/health` pool tokens and minimum free memory on each rank under a long prompt;
- a 195K needle and a 1M-token needle, followed by a sustained mixed-workload run.

Record the model revision, image hash, NCCL SHA, GPU clocks, context, sampling policy and both per-request and aggregate throughput. Run other deployments and TP4 in separate windows when they share GPUs.

## Measured performance and limits

On the measured four-Spark site, [sparkDash](sparkdash-ablit-20261006/README.md) reported 149.8 prose and 200.5 code aggregate tok/s at four concurrent requests, and 2,339 / 2,281 tok/s cold prefill at 32K / 64K. These figures used an Ablit checkpoint, local image, FP8 KV, DFlash2, split owner-row exchange, and a four-HCA ring. Each point was measured once. Mia AI's published three-Spark results used a different checkpoint and deployment; the comparison cannot assign a gain to the extra Spark alone.

TP4 partitions the weights across four ranks and can leave more memory per Spark for a shared KV pool and multiple long conversations. Communication across four ranks can also add latency. The per-request context limit is still the model's 1,048,576 tokens; any multi-session capacity benefit depends on actual free memory and workload. Mixed prefill/decode, eight concurrent requests, long-term stability, and response quality were not measured in the sparkDash run.

## Rollback

`./start-tp4.sh stop` stops this fork's four TensorFold containers. If replacing another inference service, restore that service with its existing unit or launcher, then check its `/health` and one completion. This deployment uses port 8890 by default for the model API; configure any separate gateway to forward to that port. The TensorFold fork does not alter other images, weights, or systemd units.
