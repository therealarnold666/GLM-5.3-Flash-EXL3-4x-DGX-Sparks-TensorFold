# Bidirectional owner-row relay under mixed prefill and decode

Date: 2026-10-06. This extends the [solo A/B](README.md) with the workload that
previously caused visible decode stalls: three active decode streams plus one
new 32K or 64K cold prefill. It is a custom mixed-load test using the same
OpenAI-compatible API and RigMark's prefill text; it is **not** a RigMark 1.3
protocol result.

## Controls and validity

- Both arms used `tensorfold-glm53:owner-ring-bidir-20261005` on all four
  Sparks, with the same 4HCA, `SPLIT=1`, FP8 KV, 8192-row prefill setting,
  DFlash2, `PARALLEL=4`, model and checkpoint. Only
  `TF_GLM_OWNER_BIDIR=0` versus `1` changed. Each arm restarted the service.
- Each trial started three 256-token-prompt `/v1/completions` streams at
  `max_tokens=4096`, `ignore_eos=true`; after all three emitted output and
  a two-second decode-only interval, it sent a 32,768- or 65,536-token cold
  prefill with an eight-token completion. Two trials per depth and arm.
  Prompt token IDs, temperature, and generation settings matched across arms.
- The first and last SSE event times were recorded locally. Decode streams
  continued beyond the prefill's first token in every retained trial. The
  inter-event p95 is a proxy for visible streaming pauses; SSE events may
  contain multiple tokens, so it is not a true per-token latency.
- All eight cold prefills reported exactly `cached_tokens=0` and the requested
  prompt-token count. Both arms had exactly 16 benchmark requests
  (`requests_total` increased by 16), without external traffic. All four
  paired prefill output hashes and all twelve paired decode output hashes
  matched. All decode streams returned 4096 tokens.
- An initial pilot used 2048 decode tokens and did not fully cover 64K
  prefill, so it was discarded. Another attempted repeat hit the prompt cache
  even with a new `cache_salt`; the final sweep uses a fresh prompt nonce and
  explicitly rejects cache hits. Only the final two valid trials per depth
  appear below.

## Results

Medians are taken across the two prefill trials or, for decode, the six
stream-level p95 values per depth. Lower is better.

| Concurrent workload | Metric | Clockwise `0` | Bidirectional `1` | Change |
| --- | --- | ---: | ---: | ---: |
| 3 decode + 32K cold prefill | Prefill TTFT | 26.45 s | 25.21 s | −4.7% |
| 3 decode + 32K cold prefill | Decode SSE-gap p95 | 79.4 ms | 72.3 ms | −9.0% |
| 3 decode + 32K cold prefill | Median of stream maximum gaps | 1.27 s | 1.75 s | +38% |
| 3 decode + 64K cold prefill | Prefill TTFT | 48.80 s | 50.87 s | +4.2% |
| 3 decode + 64K cold prefill | Decode SSE-gap p95 | 80.7 ms | 80.4 ms | −0.4% |
| 3 decode + 64K cold prefill | Median of stream maximum gaps | 2.12 s | 2.25 s | +6% |

The four individual cold TTFTs were 27.80 and 25.11 s at 32K, and 49.02
and 48.58 s at 64K for mode `0`; 25.72 and 24.71 s at 32K, and 51.63 and
50.10 s at 64K for mode `1`. Decode-only segments had median SSE-gap p95
around 26–32 ms. During prefill, it was around 72–81 ms, with occasional
system-wide stalls of roughly 0.5–3.4 s in individual trials. The maximum
gap is noisy, but it does not support calling the new route harmless.

## Decision

The bidirectional route improves the isolated owner-row exchange and the
link-direction balance (see [solo A/B](README.md)), but this mixed-load A/B
does not establish a serving benefit. The 32K measurements mildly favor it;
the 64K prefill gets slower, decode p95 is unchanged at 64K, and maximum
stalls remain. Two trials per depth are sufficient to reject a clear large
gain, not to resolve small differences. Keep the candidate opt-in and outside
the normal patch set. The next performance investigation should instrument
the prefill/decode scheduler's fill slices and verify windows; the current
startup reports 1024-row fill chunks, a 0.5 fill share and a 200 ms target,
yet the client still observes multi-second pauses.

Compressed raw receipts: [clockwise](mixed-mode0-v2.json.gz),
[bidirectional](mixed-mode1-v2.json.gz). Reproducer:
[`bench_mixed_load.py`](../../experiments/owner-ring-bidir/bench_mixed_load.py).
After the A/B, `scripts/local.sh` was restored byte-for-byte on the head,
`PREPARE` was removed from the user manager environment, and the original
`tensorfold-glm53:owner-ring-20261005` image, service and watchdog timer were
verified active.
