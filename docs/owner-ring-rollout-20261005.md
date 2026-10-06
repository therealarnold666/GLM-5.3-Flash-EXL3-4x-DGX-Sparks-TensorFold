# TP4 owner-row ring rollout (2026-10-05)

## Change

Patch `0071-glm-tp4-owner-ring.patch` adds `TF_GLM_HC_EXCHANGE=ring` to the existing `SPLIT=1` prompt path. On four ranks, each rank sends FP32 partials only for the destination's rows. The opposite rank's block passes through one physical neighbor. The owner sums FP32 partials in rank order with the same HC kernel; completed BF16 rows still use the existing NCCL all-gather. Decode, DFlash2, KV cache, HTTP API, checkpoint and quantization code are untouched. `gather` stays available as a one-variable rollback.

The measured site keeps four NCCL channels, `TF_GLM_PREFILL_ROWS=8192`, `TF_GLM_FILL_ROWS=1024` when decode is active, `TF_GLM_FILL_BUDGET_MS=200`, `PARALLEL=4`, FP8 KV, the TR3 4-bpw checkpoint and the 1M context window. The image is `tensorfold-glm53:owner-ring-20261005`, built incrementally from the former `tensorfold-glm53:v0.6.0` on each node. The image patch label is `5c1662c5151a`. All four images report identical SHA-256 values for the modified `comm.py` (`5ff82d8bd7719a1833dd979ac6eef099d282d5b9b189d8a1669625da99195f10`) and `hcsplit.py` (`f46ebd708f2258a997093afa92d8283f16c15eae6cdf7eeb6437b99f72388c65`). The patched NCCL library and weights remain the same.

The old image and the prior head deployment directory (`~/glm53-tf-tp4.pre-owner-ring-20261005`) remain available. The managed service had been stopped before this rollout; both A/B arms were started manually with the four-rank launcher.

## Matched cold prefill A/B

Same image, weights, four-Spark ring, 8192-row chunks, four channels and requests. Each length had one discarded warmup and three formal unique prompts; formal `cached_tokens` was zero throughout. Direct port 8890 reached through an SSH tunnel, without the console gateway.

| Prompt | Full FP32 gather | Owner-row ring | Change |
| --- | ---: | ---: | ---: |
| 32K | 1435.0 tok/s; TTFT 22.831 s | 1919.2 tok/s; TTFT 17.071 s | +33.7% throughput |
| 64K | 1407.9 tok/s; TTFT 46.545 s | 1861.2 tok/s; TTFT 35.209 s | +32.2% throughput |

Raw runs: [gather prefill](ring-row-exchange-20261005/prefill-gather.json) and [ring prefill](ring-row-exchange-20261005/prefill-ring.json). The standalone communication result is in [ring-row-exchange-20261005](ring-row-exchange-20261005/README.md).

## Other capabilities

- `ring` startup loaded four ranks with the same 1,048,576-token window and 4,585,472-token shared pool; startup smoke returned `OK`.
- 8K/32K, 256-token decode (three samples each): prose 44.5 / 42.5 tok/s; code 94.9 / 97.9 tok/s. [Raw run](ring-row-exchange-20261005/decode-ring.json). These sit near the prior 1005 values of prose 43.7 / 44.3 and code 95.5 / 103.7; a same-prompt decode A/B would be needed to attribute small differences.
- The tool-call array fixture returned the expected `add_tags` call with three string tags and integer document ID.
- A 125,729-token needle prompt returned the exact passphrase (`violet-harbor-7291`); prefill was 70.664 s.
- Sanitized Hermes session replays retained exact cache counts of 9,344 / 27,264 / 109,568 tokens on immediate repeats. Hot TTFT was 0.209 / 0.255 / 0.329 s for 8K / 32K / 128K targets. The private local replay receipt is intentionally not published.
- Three simultaneous 2,048-token streams continued during a new 31,448-token cold fill (`cached_tokens=0`): each produced 241–248 visible chunks during the 25.0 s fill; maximum client-visible gaps were 0.201 / 0.201 / 0.201 s. See [mixed-load.json](ring-row-exchange-20261005/mixed-load.json).
- With the same image, prompt text, seed, temperature and output limit, `gather` and `ring` produced identical SHA-256 hashes for a prose reply and a 256-token code reply. Code decode time was 2.948 s (`gather`) and 2.961 s (`ring`). See [gather fixture](ring-row-exchange-20261005/fixed-replies-gather.json) and [ring fixture](ring-row-exchange-20261005/fixed-replies-ring.json). The two paths call the same HC post Triton kernel with different source strides, retaining rank-order FP32 arithmetic.

## Site operation and rollback

Keep the site-specific workers, HCA pins and checkpoint revisions in `scripts/local.sh`. For the new serving path set `IMAGE=tensorfold-glm53:owner-ring-20261005`, `SPLIT=1` and `TF_GLM_HC_EXCHANGE=ring`. `start.sh` rejects `ring` unless TP4, `COMM=nccl` and the switchless topology are selected. Restart through `./start-tp4.sh stop` followed by `./start-tp4.sh`; the launcher starts workers 3, 2, 1 before rank 0. A one-variable rollback sets `TF_GLM_HC_EXCHANGE=gather` and restarts all four ranks. To restore the old image too, set `IMAGE=tensorfold-glm53:v0.6.0` and `TF_GLM_HC_EXCHANGE=gather`.

This site's image was built locally on each Spark from its already-installed old image, then the two modified source files and patch label were checked byte-for-byte across the four images. This avoids streaming the 31 GB complete image over the management network. The compiled-kernel cache was copied from the old image's cache because this patch changes only Python communication and HC dispatch, not CUDA extensions. A normal full `scripts/prepare.sh` on a fresh site remains the general installation path.

Final state: `scripts/local.sh` on the head selects `IMAGE=tensorfold-glm53:owner-ring-20261005`, `SPLIT=1`, `TF_GLM_HC_EXCHANGE=ring` and 8192 prefill rows. The four ranks were restarted through the TP4 launcher without `PREPARE=0`; the prepared-state marker was refreshed after checking the unchanged checkpoint and four image labels/source hashes. `glm53-tf-tp4.service` and `glm53-tf-tp4-watch.timer` are active and enabled. `/health` reports a 1,048,576-token context, 4,585,472-token pool and four streams; the authenticated port-8888 gateway lists `glm-5.3-flash` and forwards a completion to TensorFold successfully.
