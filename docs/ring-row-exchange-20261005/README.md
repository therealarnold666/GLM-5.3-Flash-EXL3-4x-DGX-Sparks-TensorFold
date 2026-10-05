# Four-Spark ring row-exchange microbenchmark (2026-10-05)

## Question and scope

The current GLM-5.3-Flash TensorFold TP4 `SPLIT=1` path all-gathers a full FP32 hyper-connection partial before each owner-row glue. Can each rank instead receive only the partials for its own rows, using only the existing physical ring links? This is an isolated communication test. It did not load model weights, start the API, alter host fabric routing, or deploy a new TensorFold path.

## SparkRing comparison

SparkRing source inspected at commit `6d27dd10a9f97da21278ae504ea6235f8634f303`. Its [transport overview](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/spark_transport/README.md) and [architecture](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/docs/architecture/overview.md) distinguish three paths: RoCEnante for selected small collectives, patched NCCL ring for larger prefill collectives, and a research-only hardware-forwarded opposite-rank mesh. The [hardware-forwarding contract](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/spark_transport/fabric/cx7_hairpin_diagonal/README.md) builds logical diagonal paths by forwarding packets in an intermediate ConnectX-7 ASIC; it needs host fabric setup and still shares cable bandwidth. The [NCCL fallback](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/spark_transport/nccl/README.md) explicitly does not provide IP forwarding between opposite ranks. SparkRing's [bidirectional ring prefill research](https://github.com/FujitsuPolycom/sparkring/blob/6d27dd10a9f97da21278ae504ea6235f8634f303/spark_transport/experiments/tiled_prefill/README.md) is a related scheduling idea, but its BF16 all-reduce contract does not directly replace TensorFold's FP32 owner-row partial exchange.

Our current site has patched NCCL ring routing only. `HcSplit`'s direct `exchange_all` wants 0–2 and 1–3 peer connections that are not present. The tested owner-row algorithm forwards the opposite-rank block through the clockwise neighbor using NCCL send/receive on physical ring edges. It does **not** use SparkRing hardware forwarding or install any SparkRing component.

## Geometry and method

- Four DGX Sparks in the existing `0–1–2–3–0` ring, same `tensorfold-glm53:v0.6.0` serving image and patched NCCL 2.30.7 library, four NCCL channels and the serving ring environment.
- Checkpoint `text_config.hidden_size=4096`; prefill chunk `8192` rows, so each rank's FP32 partial is `[8192,4096]` = **128 MiB**. Each owner block is `[2048,4096]` = **32 MiB**. Full all-gather writes a 512 MiB receive tensor per rank.
- Owner exchange round 1 sends one block to each direct neighbor plus one opposite-destination block clockwise. Round 2 forwards the opposite block clockwise. Each rank transmits 128 MiB across its ring links per exchange; no diagonal peer connection is opened.
- Each mode had a warmup and three timed repeats of 90 exchanges, matching the serving path's 90 partial-gather sites per full 8192-row chunk. Timing uses CUDA events and an enclosing wall clock. Every rank checked first and last elements of the gathered and owner blocks against distinct origin/destination markers. This checks rank routing, not the model's numerical equivalence.
- Script: [`tools/bench_ring_row_exchange.py`](../../tools/bench_ring_row_exchange.py). Raw container logs: [`rank0.log`](rank0.log), [`rank1.log`](rank1.log), [`rank2.log`](rank2.log), [`rank3.log`](rank3.log).

## Results

Median CUDA time per exchange (milliseconds):

| Rank | Full FP32 all-gather | Owner rows, neighbor forwarding |
| --- | ---: | ---: |
| 0 | 29.0534 | 12.3947 |
| 1 | 29.0535 | 12.3974 |
| 2 | 29.0536 | 12.3992 |
| 3 | 29.0534 | 12.3997 |

Across ranks: **29.05 ms → 12.40 ms**, a **57.3% communication-time reduction** for this isolated shape (2.34×). Ninety exchanges take **2.615 s → 1.116 s**. Wall and CUDA timings agree to within ~0.001 ms per exchange. The current serving-path CUDA event probe measured roughly 2.67–2.70 s per chunk for the FP32 gather, close to this standalone 2.615 s.

These numbers do not include TensorFold's FP32 rank-order sum, HC kernels, BF16 completed-row all-gather, interleaved model computation or stream overlap. They are **not** an end-to-end prefill prediction. A model-integrated implementation also needs buffer ownership, synchronization and numerical validation, especially because the existing path uses a second CUDA stream.

## Decision and cleanup

The owner-row neighbor-forwarding prototype is worth implementing behind an opt-in flag. First test exact rank-order FP32 sum and BF16 output against the existing path; then compare repeated 32K/64K/128K cold prefill and decode/concurrency behavior. The benchmark itself changes no production code or configuration.

After saving logs, all four benchmark containers and temporary copied scripts were removed. `glm53-tf-tp4` remained inactive on all four nodes, as it was before the test.

The later [model-integrated rollout](../owner-ring-rollout-20261005.md) added an opt-in owner-row ring path and measured cold 32K/64K prefill, decode, long-prompt correctness and mixed load on the same four nodes.
