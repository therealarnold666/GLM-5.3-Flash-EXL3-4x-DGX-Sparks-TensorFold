# sparkDash：本机 TP4 Ablit 与 Mia AI TP3 公布成绩

测试时间：2026-10-06 07:24–07:29 UTC。原始逐请求结果见 [results.json](results.json)，复现脚本见 [run.mjs](run.mjs)。

这个目录记录的是**四机 TP4 的一次实测**。仓库默认下载 Mia AI 的原始 EXL3 权重；本次使用的 Ablit 权重和本地镜像不包含在仓库中，因此直接运行默认配置不会得到同一组数字。

## 口径

- 使用 Mia AI `sparkDash` v1.8.8（本机检出的提交 `754f40a`）的原生 `DecodeBench` 和 `PrefillBench`，从工作站经 SSH 本地转发请求 head 的 `127.0.0.1:8890`。没有切换或重启生产服务。
- 后端实测为 4 台 Spark 的 TensorFold，镜像 `tensorfold-glm53:short-prompt-v2-20261006`，Ablit 权重 `ablit-o-proj-l15-46-skip44-20261006`，TP4，1,048,576 上下文，`parallel=4`。服务默认带 `--thinking`，但 sparkDash 请求显式关闭 thinking；解码温度 0、top-p 1、每条流强制 400 输出 token。预填充各长度使用新的随机前缀、各 8 token 上限。
- Mia 数字取 [官方仓库 README 的“三台 Sparks，4 并发，sparkDash”表](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold#3-sparks-experimental)。其中是原始 4-bit 权重的已公布测量，`FP8 KV`、DFlash2+copy、1M 上下文。官网另有 8 并发成绩，不与本次 `parallel=4` 横向比较。
- 每个点各测一次。sparkDash 预填充提示采用唯一前缀以避免前缀复用，但本次记录不含服务端 `cached_tokens` 字段，故“冷”由测试设计保证，无法用逐请求缓存计数复核。

## 解码：并发聚合 tok/s

| 并发 | prose 本机 TP4 | Mia TP3 | 差值 | code 本机 TP4 | Mia TP3 | 差值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82.2 | 77.6 | +5.9% | 123.8 | 104.3 | +18.7% |
| 2 | 103.9 | 95.7 | +8.5% | 161.8 | 139.0 | +16.4% |
| 3 | 136.3 | 121.2 | +12.5% | 185.0 | 158.7 | +16.6% |
| 4 | 149.8 | 146.2 | +2.5% | 200.5 | 169.6 | +18.2% |

每条流都完整输出 400 token，全部成功：C1/C2/C3/C4 的有效解码 token 总数分别为 399/798/1197/1596。没有失败流。prose C4 的 +2.5% 接近单次测量波动范围，不能据此宣称稳定胜出。

## 冷预填充：tok/s 与 TTFT

| 目标长度 | 本机实际 tokens | 本机 tok/s | Mia tok/s | 差值 | 本机 TTFT | Mia TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 8,216 | 2,168 | 2,001 | +8.4% | 3.79 s | 4.59 s |
| 16K | 16,411 | 2,296 | 2,000 | +14.8% | 7.15 s | 8.20 s |
| 32K | 32,795 | 2,339 | 2,064 | +13.3% | 14.02 s | 15.89 s |
| 64K | 65,560 | 2,281 | 2,004 | +13.8% | 28.75 s | 32.72 s |
| 128K | 131,097 | 2,123 | 1,882 | +12.8% | 61.75 s | 69.67 s |
| 256K | 262,171 | 1,840 | 1,655 | +11.2% | 142.47 s | 158.45 s |

## 解读与局限

本机 4 台的绝对吞吐与 TTFT 在这组 4 并发以内的测试中均领先 Mia 公布的 3 台；64K 冷预填充优势约 14%，代码 C4 聚合优势约 18%。不过本机多用一台设备、采用 Ablit 权重和后续自定义 TensorFold 镜像，远程路径也通过 SSH 转发；差异不能归因于“增加第四台”或某一项补丁。Mia 公布表是另一时期、另一套权重/镜像的单轮数据。本测试没有测 8 并发、混合 prefill+decode、质量或长期稳定性。

结束后 head 的 `glm53-tf-tp4.service` 仍为 `active`，`/health` 的 `requests_running=0`，所有流均空闲。

## 复现请求口径

先在测试机取得 sparkDash v1.8.8 的源码（本次使用提交 `754f40a`），再把本机 `18890` 转发到四机 head 的 API 端口 `8890`。在仓库根目录运行：

```bash
SPARKDASH_ROOT=/path/to/sparkDash \
BENCH_HOST=127.0.0.1 BENCH_PORT=18890 BENCH_MODEL=glm-5.3-flash \
node docs/sparkdash-ablit-20261006/run.mjs
```

脚本把新结果写到忽略的 `local-results/`，不会覆盖此处留存的测试记录。它会向指定 API 发起完整的解码与最长 256K 的预填充测试；请在专用测试窗口运行。若要比较相同部署，需另外固定权重、镜像、拓扑、推理参数、时钟和缓存状态。
