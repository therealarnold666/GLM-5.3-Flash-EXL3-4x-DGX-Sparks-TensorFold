# TP4 short-prompt admission experiment (2026-10-06)

Patch `0072-glm-short-prompt-priority.patch` adds an opt-in scheduler setting,
`TF_GLM_SHORT_PROMPT_ROWS` (`SHORT_PROMPT_ROWS` in `scripts/local.sh`). The default
is `0`, preserving existing FIFO and grouped prefill. With `512`, a waiting
foreground prompt with at most 512 rows remaining runs at the next prefill
chunk boundary before an older long prompt. It runs separately only when a
longer filling prompt is present; simultaneously arriving short prompts still
share a grouped forward. An already running GPU forward cannot be interrupted.
The patch does not change model weights, arithmetic, decode or cache rules.

The measured site uses four DGX Sparks on a switchless TP4 ring, four HCAs,
`SPLIT=1`, owner-row ring exchange, 8192 prefill rows, 2048 rows while another
request decodes, a 200 ms fill-slice budget, FP8 KV and DFlash2. **Both arms
served the same local EXL3 Ablit checkpoint.** An earlier baseline RigMark JSON
mistakenly named the original TR3 checkpoint in its metadata; the live rank
commands and the serving process start time were checked after that run. The
unmodified original JSON SHA-256 is in [the compact summary](rigmark-summary.json).

## Mixed long-then-short arrival

Three repetitions each started a new 32,768-token cold prefill, then inserted
an approximately 120-token code request 0.5 seconds later. Each short request
generated 256 tokens. Cold requests reported zero cached tokens. The endpoint
was shared, so these are observed samples, not an isolated capacity limit.

| Metric, median | Before | Patch 0072 v2 |
| --- | ---: | ---: |
| Short request first token | **10.241 s** | **3.767 s** |
| Short request alone, first token | 0.388 s | 0.392 s |
| 32K cold request first token during mix | 15.305 s | 17.459 s |

The short request improved by **63%**, while the long request took about
**14% longer** because the short request ran first. Individual short TTFTs
with the patch were 4.344, 3.586 and 3.767 seconds. This does not deliver
subsecond admission: the current 8192-row forward still finishes before the
new request can be selected. See the [baseline](mixed-baseline.json) and
[candidate](mixed-candidate-v2.json) metric receipts. These receipts contain
timings, token counts and hashes; no prompt text, generated text, address or
credential is published.

## Standard RigMark 1.3.0 check

The same prompt corpus, request settings and comparison ID were used before
and after the patch. All 15 output gates passed in each arm; all 15 paired
single-request output hashes matched. Nine cold samples per arm reported zero
cache hits, and nine immediate replays reported full hits.

| Median | Before | Patch 0072 v2 |
| --- | ---: | ---: |
| Code decode | 109.2 tok/s | 108.9 tok/s |
| Prose decode | 57.8 tok/s | 57.4 tok/s |
| 8K cold prefill | 2251 tok/s | 2247 tok/s |
| 32K cold prefill | 2321 tok/s | 2321 tok/s |
| 64K cold prefill | 2263 tok/s | 2242 tok/s |
| C4 capped aggregate | 143.9 tok/s | 141.4 tok/s |
| C4 first token | 0.665 s | 0.872 s |

Short-only C4 aggregate stayed within 2%; its measured first-token median
was 0.207 seconds higher. The C4 test uses approximately 130-token prompts
and a 256-token output cap, so it does not measure completed agent tasks.
Three C4 rounds are insufficient to assign the small differences solely to
this patch. [Compact numeric summary](rigmark-summary.json) ·
[candidate card](rigmark-candidate-v2.card.txt).

## Build and rollback

`deploy/Dockerfile.short-prompt-incremental` applies patch 0072 on an already
qualified `tensorfold-glm53:owner-ring-20261005` image. Build on every rank,
using `patches/` as context and `scripts/config.sh`'s `image_hash` as
`PATCHES_HASH`; ensure the resulting patch label and the source hash match
across ranks. The measured site selected the incremental image and
`SHORT_PROMPT_ROWS=512` through its ignored `scripts/local.sh`. The image and
setting are opt-in; generic defaults remain unchanged. The production service
and its watchdog were active and `/health` was idle after the full benchmark.

To roll back, stop the watchdog and service, restore the previous local
configuration and owner-ring image, remove patch 0072 from the site's local
patch directory, refresh the prepared-state marker after checking the images
and unchanged checkpoint on all ranks, then start the service before the
watchdog. The original owner-ring image remains available on every node.

Under a sustained stream of new short prompts, the priority rule can delay
long prefill progress. Only three arrivals were sampled here. A later change
to bound that delay or interrupt large idle fill chunks needs its own A/B.
