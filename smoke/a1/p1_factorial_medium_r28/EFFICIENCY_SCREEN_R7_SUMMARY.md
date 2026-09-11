# r28 efficiency screen r7

设备：GPU1（RTX 4090），batch=1，640×640，20 次 warmup、60 次测量，model-forward only。

## 同设备四组基线

| 组别 | mean ms | p50 ms | p99 ms | img/s | peak MiB |
|---|---:|---:|---:|---:|---:|
| A Dense + NMS | 5.809 | 5.795 | 6.326 | 172.1 | 686.9 |
| B Dense + E2E | 6.236 | 6.226 | 6.355 | 160.4 | 687.5 |
| C MoE + NMS | 13.004 | 12.947 | 13.823 | 76.9 | 695.5 |
| D MoE + E2E | 13.879 | 13.779 | 14.534 | 72.0 | 696.1 |

相对 Dense，MoE 推理耗时约增加 124%（C 对 A）和 123%（D 对 B）；显存只增加约 9 MiB。

## 推理代理筛选

| 变体 | C mean ms | C img/s | D mean ms | D img/s |
|---|---:|---:|---:|---:|
| Top-2（基线） | 13.004 | 76.9 | 13.879 | 72.0 |
| Top-1 | 12.736 | 78.5 | 13.454 | 74.3 |
| cap4 + Top-2 | 13.557 | 73.8 | 14.523 | 68.9 |
| cap4 + Top-1 | 12.335 | 81.1 | 13.291 | 75.2 |
| cap2 + Top-1 | 12.502 | 80.0 | 12.722 | 78.6 |
| 仅保留后层 MoE | 5.668 | 176.4 | 6.299 | 158.8 |

`cap4/cap2` 是推理代理：保留原 16 专家权重，但将路由重映射到前 N 个专家，不能当作独立训练的 N-expert 精度结论。`仅保留后层 MoE` 用于估算位置对效率的影响，同样没有精度结论。

## 当前判断

- Top-1 只带来约 2–3% 延迟改善，说明主要瓶颈不是专家卷积数量，而是路由与 dispatch 同步。
- 限制专家数并未稳定改善耗时，验证了需要设备端分组/融合 dispatch，而不是简单减少专家数。
- 只在后层使用 MoE 可将延迟拉回 Dense 水平，但需要独立训练验证精度。

证据：`efficiency_screen_r7_{a,b,c,d}_evidence.json`。远端原始结果位于 `/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28/efficiency_screen_r7_{a,b,c,d}`。

## Dispatch 原型 r9

新增 `experimental_dispatch_mode="dense_gather"`，在设备端计算专家输出并进行 Top-K gather，完全绕过 Python `mask.any()` 分发。C/D 输出相对原 sparse 路径最大误差均为 0；GPU1 clean repeat（20 warmup、60 samples）如下：

| 组别 | sparse mean | dense-gather mean | sparse p99 | dense-gather p99 |
|---|---:|---:|---:|---:|
| C | 13.011 ms | 14.439 ms | 13.532 ms | 15.243 ms |
| D | 13.916 ms | 14.376 ms | 14.433 ms | 14.721 ms |

该原型消除了路由同步但仍略慢（C +11.0%，D +3.3%），说明仅去掉同步还不够；下一步需要真正的 grouped/fused sparse kernel，避免计算全部专家。

## vmap grouped 原型

又测试了基于 `torch.func.vmap` 的 Top-K 参数选择原型。C/D 的单个 MoE block 输出误差约为 `4.4e-3` 以内，但全模型输出等价性检查出现明显误差（C `5.5e-1`、D `6.76e+2`），因此该原型未通过验收，不纳入正式实现。其 clean 延迟约 C 14.1 ms、D 14.3 ms，也没有优于现有 sparse 路径。
