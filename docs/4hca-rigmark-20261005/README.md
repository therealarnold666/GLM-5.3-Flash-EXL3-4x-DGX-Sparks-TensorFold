# Four-HCA TP4 RigMark qualification (2026-10-05)

## Change and controls

All four Sparks have two direct RoCE functions for each physical neighbour: `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, and `roceP2p1s0f1`. Every function was LINK_UP at 200 Gb/s, MTU 9000, with a valid IPv4 RoCE v2 GID at index 3. The former TensorFold site pinned the first two functions, one per neighbour. The candidate pins all four. No cable, model image, checkpoint, context, KV, DFlash2, scheduler, `SPLIT=1`, owner-row ring exchange, or four-channel count changed.

The prior hardened NCCL (`sha256:78cb83871792`) deliberately rejects more than two eligible listener GIDs. A first four-HCA startup failed at NCCL initialization with that exact diagnostic; the two-HCA service was restored before proceeding. For a matched comparison, the **same** NCCL 2.30.7 candidate library was then used in both formal arms. It is the locally built SparkRing dual-PCI-domain / extended-IPv4-GID candidate, SHA-256 `e64ac9b8531ea5102dfe03f1e496db31b0457a3172fc98c5315f73ed3e35427e` on all four machines. Its build receipt pins NVIDIA NCCL source `73cf112295c33aee2b895f329f592f2a9b4b0f97`, patch SHA-256 `8e2b8715d62d2b07a74caca3778da0eff2e7b54caf8a184bb728f179a5d1eba4`, CUDA 13.0.88 and `sm_121`. The codec, PCI-root and legacy-handle CPU test passed. The old library remains on all nodes at `~/nccl-switchless-v0.0.1`.

Both formal arms used `tensorfold-glm53:owner-ring-20261005`, the `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` checkpoint at `25a44fdbf16862a46b7cc9921142c6c81350af2f`, DFlash2 revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, FP8 KV, 1,048,576-token context, `PARALLEL=4`, 8192 prefill rows, `TF_GLM_HC_EXCHANGE=ring`, `NCCL_IB_MERGE_NICS=0`, `NCCL_ALGO=Ring` and four channels. The four-HCA arm additionally set `NCCL_IB_EXTENDED_IPV4_GIDS=1` and `NCCL_IB_PRESERVE_PCI_DOMAIN=1`, required to advertise all four IPv4 GIDs while keeping a route within the selected PCI root. `scripts/nodes.sh` passes these two variables only when the site sets them.

## Matched RigMark 1.3.0

Both runs used RigMark revision `68800bf29cf1f85f834affdf81fe3e8703327342`, comparison ID `spark-glm53-20261004`, the same request settings (`reasoning_effort=low`, temperature 0, seed 20260905), the same 8888 gateway path, and exactly 54 requests per run as confirmed by TensorFold's `/health` counter. Each run passed all 15 basic output gates. All nine cold prefill requests in **each** arm reported `cached_prompt_tokens=0`; all nine immediate replays reported full hits. The 15 paired decode samples produced identical visible-output SHA-256 hashes.

| RigMark median | 2 HCA | 4 HCA | Change |
| --- | ---: | ---: | ---: |
| 8K cold prefill | 1,897 tok/s; TTFT 4.32 s | **2,254 tok/s; 3.63 s** | **+18.8%** |
| 32K cold prefill | 1,924 tok/s; TTFT 17.03 s | **2,324 tok/s; 14.10 s** | **+20.8%** |
| 64K cold prefill | 1,882 tok/s; TTFT 34.83 s | **2,262 tok/s; 28.97 s** | **+20.2%** |
| Code decode | 109.3 tok/s | 110.2 tok/s | +0.8% |
| Prose decode | 59.1 tok/s | 59.9 tok/s | +1.4% |
| Structured ceiling | 149.3 tok/s | 152.3 tok/s | +2.0% |
| C4 capped code aggregate | 148.8 tok/s | 150.6 tok/s | +1.2% |

Receipts: [2 HCA](2hca.json), [4 HCA](4hca.json), and the [RigMark matched-request card](compare.card.txt). The initially repeated original-library two-HCA run is excluded from this comparison: other clients sent 128 requests during its window. Both candidate-library arms were isolated by the 54-request counter deltas.

`port_xmit_data` deltas across the full 4-HCA RigMark show each Spark sending 443.35–443.36 GB on the primary PCI root and 441.64–441.67 GB on the second root; the second root carried **49.9%** of transmitted bytes. Raw [before](counters-before.txt) and [after](counters-after.txt) snapshots are retained. All 16 functions reported zero receive errors and transmit discards at the end. The head's four mlx5 sensors each read about 50°C after the run. These are run-window checks, not a sustained thermal qualification.

## Active site and rollback

The head's ignored `scripts/local.sh` now pins four HCA names per rank and sets `NCCL_HOST_DIR` / `WORKER*_NCCL_HOST_DIR` to each rank's `~/nccl-4hca-test-20261005`, with the two extended-routing switches set to 1. The source image and old NCCL directory remain available. `glm53-tf-tp4.service` and `glm53-tf-tp4-watch.timer` are active; `/health` shows the original 1M context and four streams.

To return to the former production two-HCA NCCL build on the head, stop the watchdog before the service, restore the backed-up site file and launcher helper, then start the service before the watchdog:

```bash
systemctl --user stop glm53-tf-tp4-watch.timer
systemctl --user stop glm53-tf-tp4.service
cp -a ~/glm53-tf-tp4/scripts/local.sh.pre-4hca-20261005 ~/glm53-tf-tp4/scripts/local.sh
cp -a ~/glm53-tf-tp4/scripts/nodes.sh.pre-4hca-candidate-20261005 ~/glm53-tf-tp4/scripts/nodes.sh
systemctl --user start glm53-tf-tp4.service
systemctl --user start glm53-tf-tp4-watch.timer
curl -fsS http://127.0.0.1:8890/health
```

The current four-HCA setup is an opt-in site test on a separately built NCCL library. Its 8K–64K RigMark result establishes the observed benefit on this cluster; it does not establish long-duration thermal behaviour or performance for other checkpoints/topologies.
