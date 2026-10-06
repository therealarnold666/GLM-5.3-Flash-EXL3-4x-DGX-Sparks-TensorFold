# GLM-5.3-Flash EXL3 on four DGX Sparks — TensorFold TP4

This fork serves **one GLM-5.3-Flash model across four DGX Sparks** in tensor parallelism. The Sparks are connected by a direct, switchless ConnectX-7 ring; rank order follows the physical cable cycle. The head runs the OpenAI-compatible API, and `./start-tp4.sh` manages all four ranks. This repository's deployment path is **TP4**. For two- or three-Spark installations, use [Mia AI's original TensorFold project](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

```text
                  management LAN: SSH + rendezvous
             ┌───────────────────────────────────┐
             │                                   │
       head / rank 0 ─── rank 1 ─── rank 2 ─── rank 3
             └──────────── CX7 ring ─────────────┘
```

The four physical CX7 links form the cycle `rank 0 → 1 → 2 → 3 → 0`; there is no direct 0–2 or 1–3 link. The launcher checks the ring and the matching patched NCCL library before starting. See the [TP4 deployment guide](docs/tp4-switchless.md) for cabling, preflight, image preparation, rollout, and rollback.

## Measured four-Spark result

The table below is the [2026-10-06 sparkDash v1.8.8 run](docs/sparkdash-ablit-20261006/README.md), with [request-level data](docs/sparkdash-ablit-20261006/results.json) and a [reproduction script](docs/sparkdash-ablit-20261006/run.mjs). The measured site used four Sparks, a locally built TensorFold image with the short-prompt scheduler, an **Ablit EXL3 checkpoint**, `SPLIT=1`, owner-row ring exchange, FP8 KV, DFlash2, a 1,048,576-token window, and `PARALLEL=4`. The API was reached through an SSH tunnel. These are **measured-site settings**, not the clean checkout's defaults.

| sparkDash workload | Four-Spark TP4 | Mia AI's published three-Spark TP3 |
| --- | ---: | ---: |
| Prose decode, C1 / C4 aggregate | 82.2 / 149.8 tok/s | 77.6 / 146.2 tok/s |
| Code decode, C1 / C4 aggregate | 123.8 / 200.5 tok/s | 104.3 / 169.6 tok/s |
| 32K cold prefill | 2,339 tok/s; 14.02 s TTFT | 2,064 tok/s; 15.89 s TTFT |
| 64K cold prefill | 2,281 tok/s; 28.75 s TTFT | 2,004 tok/s; 32.72 s TTFT |
| 256K cold prefill | 1,840 tok/s; 142.47 s TTFT | 1,655 tok/s; 158.45 s TTFT |

Both columns use sparkDash, but **they are different deployments and checkpoints**. The TP3 figures are from [Mia AI's published table](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold#3-sparks-experimental); they are not a same-day, same-weight A/B. Each point in this run was measured once. The result supports this four-Spark deployment's observed performance; it does not isolate the benefit of a fourth Spark, establish statistical superiority, or measure mixed prefill/decode and eight-way concurrency. The C4 prose difference is only 2.5%.

Other four-Spark measurements: [ring exchange](docs/ring-row-exchange-20261005/README.md), [four-HCA prefill](docs/4hca-rigmark-20261005/README.md), [mixed prefill budget](docs/fill-budget-20261006/README.md), and [short-prompt admission](docs/short-prompt-priority-20261006/README.md).

## Requirements

- Four DGX Sparks or compatible GB10 systems, each with 128 GB unified memory and sufficient free GPU memory for one rank.
- A directly cabled CX7 ring in rank order 0–1–2–3–0. Each neighbor link needs working RoCE v2 addressing. A management LAN must reach all workers for SSH and TensorFold rendezvous.
- Passwordless SSH from the head to ranks 1, 2, and 3; Docker with the NVIDIA runtime and `rsync` on all four nodes.
- The **same patched switchless NCCL `libnccl.so.2`** on all four nodes. The launcher checks the library SHA-256 and each node's HCA/GID mapping. The library is site supplied; this repository does not install it.
- The TensorFold image and model weights on every rank, or enough disk and network access for `scripts/prepare.sh` to build/copy them. The default checkpoint is [Mia AI's EXL3 4-bpw release](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold). **The locally measured Ablit checkpoint is not bundled here.** DFlash2 is the default drafter; check its [license](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) before use.

## Start the four-rank service

Run these commands on the head. Configure addresses, HCA names, GID indices, and NCCL paths for **your** ring in the ignored `scripts/local.sh`; do not commit that file.

```bash
git clone https://github.com/therealarnold666/GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold.git
cd GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold
cp scripts/local.tp4.example scripts/local.sh
# Edit scripts/local.sh for your four machines and patched NCCL paths.
bash tests/test_tp4_ring.sh
DRY_RUN=1 ./start-tp4.sh
./start-tp4.sh
```

`./start-tp4.sh` selects `TP=4`, the switchless ring topology, and NCCL communication. It prepares the image and weights when needed, starts worker ranks before the head, and smoke-tests the API. The default API port is **8890**; the served model name is `GLM-5.3-Flash-EXL3`.

```bash
curl -fsS http://127.0.0.1:8890/health
curl -fsS http://127.0.0.1:8890/v1/models
./start-tp4.sh restart    # preflight, then restart all four ranks
./start-tp4.sh stop       # stop all four ranks
```

On a clean checkout, TP4 starts conservatively with `SPLIT=0`, `PARALLEL=4`, FP8 KV, DFlash2, and a requested 1,048,576-token context; actual admission depends on available memory. For the validated ring path, set `SPLIT=1` and `TF_GLM_HC_EXCHANGE=ring` only after the baseline works. The [deployment guide](docs/tp4-switchless.md) explains the matching image and NCCL requirements and the `gather` rollback. Local tuning used in the benchmark is documented in the linked experiment notes; copying the public example alone does not reproduce its numbers.

## Scope and provenance

This fork builds on Mia AI's TensorFold GLM recipe and its model patches, then adds the four-rank switchless launcher, topology checks, owner-row ring exchange, and later scheduling experiments. The original two- and three-Spark recipes remain in the upstream project; the extra launchers retained in this fork are compatibility code, not this repository's supported deployment target.

Source code is under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) and [CREDITS](CREDITS.md) for TensorFold, model, runtime, patch, and contributor attribution. Model weights are downloaded separately and retain their own licenses.
