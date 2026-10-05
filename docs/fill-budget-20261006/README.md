# TP4 mixed-load fill-row tuning (2026-10-06)

## Decision and live state

Keep the existing single-direction owner-row ring and the existing
`tensorfold-glm53:owner-ring-20261005` image. The measured four-Spark site now
sets `TF_GLM_FILL_ROWS=2048` in head's untracked
`~/glm53-tf-tp4/scripts/local.sh`; `TF_GLM_PREFILL_ROWS=8192`,
`TF_GLM_FILL_BUDGET_MS=200`, `TF_GLM_FILL_DRAFTS=1`, four HCAs, `SPLIT=1`,
`PARALLEL=4`, FP8 KV, DFlash2 and all other settings stay at their previous
values. The user systemd service and watchdog timer are active. Restart after
removing the temporary systemd-manager overrides confirmed that the saved
site file alone supplies 2048 rows; the four ranks use the original image and
the API returned a completion. The prior local file is backed up on the head
as `scripts/local.sh.pre-fillrows2048-20261006`.

The bidirectional relay experiment remains opt-in and unused. No TensorFold
source or image was changed for this tuning.

## Why this knob

The existing scheduler already interleaves layer-sliced prompt work and
drafted decode rounds. Its fill chunks were 1024 rows whenever another
request decoded, while solo prompts used 8192 rows. On this TP4 ring, a
2048-row fill chunk removes some chunk-boundary work without increasing the
200 ms slice target. This knob only applies to mixed prompt filling; solo
prefill and solo decode follow their existing paths. The earlier two-Spark
choice of 1024 rows was not a measured TP4 optimum.

We also tested the simpler 200→150 ms slice-budget change, holding 1024
rows. It reduced decode event gaps but increased cold prefill TTFT by about
10% at both depths, so it was rejected.

## Matched mixed-load A/B

The [mixed-load runner](../../experiments/owner-ring-bidir/bench_mixed_load.py)
starts three deterministic 4096-token decode streams, waits for their first
events, then adds one 32,768- or 65,536-token cold prefill. The request
prompts, generation settings, original serving image, checkpoint and HCA
configuration matched. Each arm restarted the service. Each depth had three
trials; every retained cold request reported `cached_tokens=0` and the exact
requested token count. Every arm had exactly 24 benchmark requests and no
external request in its window. All six paired prefill output hashes and all
18 paired decode output hashes matched; all decode streams produced 4096
tokens. The inter-event gap is measured at the client SSE stream and can
include more than one model token. This custom mixed-load test is not the
RigMark 1.3 protocol.

| Setting | 32K cold TTFT | 64K cold TTFT | 32K decode SSE-gap p95 | 64K decode SSE-gap p95 |
| --- | ---: | ---: | ---: | ---: |
| 1024 rows, 200 ms (prior) | 23.835 s | 48.491 s | 29.96 ms | 31.57 ms |
| 1024 rows, 150 ms (rejected) | 26.346 s | 53.747 s | 24.65 ms | 25.04 ms |
| **2048 rows, 200 ms (kept)** | **21.342 s** | **43.745 s** | **33.06 ms** | **34.04 ms** |

Changing 1024→2048 rows improved mixed cold prefill TTFT by 10.5% at 32K
and 9.8% at 64K, or effective prefill rate by 11.7% and 10.8%. The SSE-gap
p95 increased by 3.1 and 2.5 ms (about 10% and 8%); the median wall time
to finish each 4096-token decode stream decreased from 45.51→44.27 s at
32K and 63.09→60.69 s at 64K. At 32K, the largest observed gap was 143 ms
before and 73 ms after; at 64K, 73 ms before and 152 ms after. No seconds-long
stall occurred in this warmed A/B. The prior [owner-route A/B](../bidir-owner-ring-20261005/mixed-load.md)
had 0.5–3.4 s outliers on a different candidate image; this test does not
establish their cause or prove that they can never recur.

Raw timestamp receipts are gzip-compressed JSON:
[1024 rows / 200 ms](mixed-budget200.json.gz),
[1024 rows / 150 ms](mixed-budget150.json.gz),
[2048 rows / 200 ms](mixed-rows2048.json.gz).

## Other-path checks

- RigMark 1.3 full-answer decode gate with `reasoning_effort=low`: code,
  prose and structured outputs passed **3/3**. This was one run per workload,
  a functional regression check rather than a matched speed comparison.
  [Receipt](rigmark-rows2048-decode-gate.json) · [card](rigmark-rows2048-decode-gate.card.txt).
- One RigMark 32K and 64K solo cold/replay pair returned about 2327 and
  2271 cold tokens/s, with zero cold cache hits and full immediate replay
  hits. That short smoke used a 256-token decode cap, which truncated its
  three answer tasks and failed their output gates; it was excluded from
  decode validation and not retained as a reportable RigMark result.
- Two decode streams plus two concurrent 8K cold prefills completed; all
  cold caches were zero, both decode streams produced 1024 tokens, and the
  `multi_prefill` counter gained one grouped chunk with two pieces.
  [Functional receipt](multi-prefill-smoke.json).

The config is specific to this measured TP4 site. The generic script's
1024-row default remains unchanged for other topologies. To roll back on the
head, restore the backed-up `scripts/local.sh`, restart
`glm53-tf-tp4.service`, and confirm its log again reports 1024 fill rows.
