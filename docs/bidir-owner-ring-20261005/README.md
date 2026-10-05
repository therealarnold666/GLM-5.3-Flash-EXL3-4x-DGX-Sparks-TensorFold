# TP4 owner-row bidirectional relay A/B (2026-10-06)

## Decision

The bidirectional relay passed routing and output checks, but did **not** improve
end-to-end GLM-5.3-Flash serving in a matched RigMark A/B. The original
`tensorfold-glm53:owner-ring-20261005` image and its `SPLIT=1`,
`TF_GLM_HC_EXCHANGE=ring` site settings were restored on the four Sparks.
`glm53-tf-tp4.service` and `glm53-tf-tp4-watch.timer` are active. The candidate
image remains built but is not serving; its source is in
[`experiments/owner-ring-bidir`](../../experiments/owner-ring-bidir/README.md),
outside `patches/` so a normal `prepare.sh` does not include it.

## Candidate and controls

On the physical `0–1–2–3–0` ring, the existing owner-row exchange sends each
rank's opposite-owner FP32 block clockwise through a neighbour. The candidate
sends even ranks' opposite blocks clockwise and odd ranks' counterclockwise.
The intermediate rank forwards over its other direct edge. Each rank sends
two 32 MiB blocks in each direction per 8192-row exchange; there is no
diagonal peer connection, row splitting, additional buffer or change to the
rank-order FP32 reduction. The test flag was `TF_GLM_OWNER_BIDIR=0/1`.

The 4HCA configuration, NCCL 2.30.7 candidate library, four NCCL channels,
TensorFold code/image, checkpoint and DFlash2 revisions, 1M context, FP8 KV,
`PARALLEL=4`, 8192 prefill rows, and every RigMark request setting were the
same in both serving arms. Both arms used
`tensorfold-glm53:owner-ring-bidir-20261005`; the SHA-256 of its modified
`tensorfold/cuda/comm.py` was
`017e822f494a6a380c5b8b51772a117afc7118dfd6f44763d2f93bec94396717`
on all four nodes. Only the flag changed. The original image was retained for
rollback.

## Four-rank communication microbenchmark

[`tools/bench_ring_row_exchange.py`](../../tools/bench_ring_row_exchange.py)
ran in a second process on each Spark, with the qualified serving image and
NCCL environment, while user clients were paused. Its 8192×4096 FP32 partial
has 32 MiB per owner. Each run used one warmup and three timed repetitions of
90 exchanges per method. We ran forward and reverse method order, checked the
first and last markers, and required every received owner block to equal the
full all-gather block. The service request counter stayed at 227 during both
clean runs. A preliminary 30-exchange run had competing service requests and
was excluded.

| Order, rank 0 | Clockwise owner ring | Bidirectional relay | Change in exchange latency |
| --- | ---: | ---: | ---: |
| Gather → clockwise → bidirectional | 6.950 ms | 5.743 ms | −17.4% |
| Bidirectional → clockwise → gather | 7.262 ms | 5.716 ms | −21.3% |

All four ranks agreed within about 0.02 ms. Full FP32 all-gather measured
18.42 ms in both clean orders. Raw logs are
[`micro-clean-forward-rank0.log`](micro-clean-forward-rank0.log) through rank 3
and [`micro-clean-reverse-rank0.log`](micro-clean-reverse-rank0.log) through
rank 3.

## Matched RigMark A/B

RigMark protocol 1.3.0, source revision `68800bf29cf1`, comparison ID
`spark-glm53-owner-bidir-20261006`, `reasoning_effort=low`, temperature 0,
seed 20260905. Each arm sent exactly 54 benchmark requests: `/health`
`requests_total` went from 1 to 55, with no competing requests. Each cold
prefill reported `cached_prompt_tokens=0`, and each immediate replay reported
the full prompt length cached. Both arms passed all 15 basic output gates.
All 15 paired decode visible-output SHA-256 hashes and all 18 paired prefill
cold/replay output hashes matched.

| Metric (median) | Clockwise (`0`) | Bidirectional (`1`) | Change |
| --- | ---: | ---: | ---: |
| 8K cold prefill | 2,248.2 tok/s; 3.644 s TTFT | 2,248.0; 3.644 s | −0.01% |
| 32K cold prefill | 2,314.6 tok/s; 14.157 s | 2,308.9; 14.192 s | −0.25% |
| 64K cold prefill | 2,252.6 tok/s; 29.094 s | 2,241.4; 29.239 s | −0.50% |
| Code decode | 109.3 tok/s | 109.1 | −0.24% |
| Prose decode | 60.9 tok/s | 60.8 | −0.15% |
| C4 capped code aggregate | 158.2 tok/s | 159.3 | +0.73% |

The changes are below the practical resolution of this RigMark run; the
candidate has no demonstrated serving gain. The isolated exchange is faster,
but its elapsed time is partly hidden by the existing overlapped prefill, and
other work remains in the serving critical path. A layer-level profile would
be needed to apportion that critical path.

Receipts: [clockwise JSON](rigmark-mode0.json), [bidirectional JSON](rigmark-mode1.json),
[matched comparison card](compare.card.txt), and the two [`metadata-mode0.json`](metadata-mode0.json)
and [`metadata-mode1.json`](metadata-mode1.json) files. The two `rigmark-mode*.log`
files keep per-sample timings.

## Port counters and scope

The baseline port snapshot began after its RigMark had started, so total
baseline bytes are incomplete. Its directional split is still indicative:
about 83.3% on `f0` and 16.7% on `f1` within each PCI root. The complete
candidate window carried about 72.1% / 27.9%, and the two roots still split
about 50/50. This is less than the ideal 50/50 direction split because the
NCCL all-gathers still use their own ring direction. Raw snapshots are the
`ports-mode{0,1}-{before,after}.txt` files.

The A/B was one controlled serving run per arm, so sub-percent differences
are observations rather than reliable gains or regressions. Production was
restored to the prior image and local settings, with a successful `OK` smoke
reply after restart.

A subsequent [three-decode-plus-cold-prefill mixed-load A/B](mixed-load.md)
also found no reliable serving gain. The original production image and
watchdog were restored after that test.
