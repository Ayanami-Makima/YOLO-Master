# A1 P2 小规模效率筛选 r1

## 结论

六组晚层效率筛选已完成。实验只在 backbone 层 8 引入 MoE，层 4/6 保留冻结官方
C3k2ResidualFactor；所有变体均从同一个零增益 initializer 出发，在 GPU1、固定 512
张验证图和相同 1 epoch pilot 下比较。

精度方面，六组固定 val512 的 mAP50-95 为 `0.426750–0.427500`，相对匹配 Dense
对照的最大差异为 `+0.000424` / `-0.000134`，没有可辨识的精度收益或损失。效率方面，
四组 MoE 的 E2E batch=1 延迟为 `9.094–9.405 ms`，匹配 Dense 对照为 `6.242–6.369 ms`，
慢 `45.7%–47.7%`；吞吐由 `157.0–160.2 img/s` 降到 `106.3–110.0 img/s`，下降约
`31.7%–33.7%`。峰值显存只增加约 `0.3–1.3 MiB`，不是主要瓶颈。

因此，本轮筛选的决策为：

- Dense 等活动 MLP 计算控制保留为可行效率基线；
- late-layer MoE 的 2/4 experts、Top-1/Top-2 均因明显延迟损失而不进入长训；
- 不把 1 epoch 的近似等精度写成正式 MoE 收益结论，也不继续盲目增加 epoch 或专家数。

## 设计与预算

| 项目 | 固定设置 |
| --- | --- |
| 基线 | SHA `acce839c7e895d6b179de7f7093fa879e237cc7b`，`yolo26n.pt` 预训练 C3k2 基座 |
| MoE 位置 | 仅 backbone 层 8；层 4/6 为 Dense residual factor |
| 变体 | Dense ratio 4/6 对照；MoE 2/4 experts × Top-1/Top-2 |
| 数据 | COCO pilot：训练 5000、固定验证 512 |
| pilot | 1 epoch，batch 4，imgsz 640，SGD，lr0 `1e-4`，seed `260829`，AMP off |
| 性能基准 | GPU1，batch 1，30 warmup + 100 samples，E2E 与 NMS 各测 |

Dense ratio 4/6 是按活动 Top-1/Top-2 MLP 计算量设计的控制，不宣称参数量严格相等；实际
参数量、静态 THOP 和原始样本均保存在 benchmark JSON/CSV 中。THOP 对动态稀疏路由只作辅助
参考，不能替代实测延迟。

## 固定 val512 精度

| 变体 | MoE | experts | Top-k | mAP50-95 | precision | recall | 相对匹配 Dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late_dense_eq_top1 | 否 | – | – | 0.427076 | 0.647779 | 0.527554 | – |
| late_dense_eq_top2 | 否 | – | – | 0.426884 | 0.646809 | 0.528088 | – |
| late_moe_2e_top1 | 是 | 2 | 1 | 0.427138 | 0.648146 | 0.527588 | +0.000062 |
| late_moe_2e_top2 | 是 | 2 | 2 | 0.426774 | 0.645313 | 0.527404 | -0.000110 |
| late_moe_4e_top1 | 是 | 4 | 1 | 0.427500 | 0.645982 | 0.527523 | +0.000424 |
| late_moe_4e_top2 | 是 | 4 | 2 | 0.426750 | 0.645849 | 0.526232 | -0.000134 |

这是 1 epoch 筛选 pilot 的固定 val512 指标，不替代 P1 正式 COCO 结果，也不足以支持长训
收敛结论。它只用于判断是否值得把某个结构带入下一阶段。

## GPU1 batch=1 性能

下表为 E2E（完整模型输出和后处理）均值；原始逐次样本见 `benchmark/samples.csv`。

| 变体 | 参数量 | 静态 THOP (GFLOPs) | E2E mean (ms) | p50 | p99 | 吞吐 (img/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late_dense_eq_top1 | 3,280,376 | 8.1494 | 6.2423 | 6.1774 | 6.5607 | 160.20 | 688.0 |
| late_dense_eq_top2 | 3,412,472 | 8.2559 | 6.3690 | 6.2095 | 9.8010 | 157.01 | 688.5 |
| late_moe_2e_top1 | 3,351,168 | 8.0718 | 9.0944 | 9.0143 | 9.3343 | 109.96 | 688.3 |
| late_moe_2e_top2 | 3,351,168 | 8.1791 | 9.3136 | 9.2612 | 9.6546 | 107.37 | 688.3 |
| late_moe_4e_top1 | 3,616,456 | 8.0718 | 9.1488 | 9.1080 | 9.4670 | 109.30 | 689.3 |
| late_moe_4e_top2 | 3,616,456 | 8.1791 | 9.4054 | 9.4085 | 9.7273 | 106.32 | 689.3 |

NMS 模式得到相同方向：Dense 为 `5.649–5.657 ms`，MoE 为 `8.347–8.620 ms`。Top-2
相对同 experts 的 Top-1 只再增加约 `0.22–0.26 ms`；2→4 experts 在固定 Top-k 下几乎
不改善延迟，说明主要成本不是专家矩阵乘法，而是 Router、索引/分发和聚合。

## 门禁与下一步

本轮使用“固定 val512 无明显 AP 损失 + 实测延迟可接受”作为筛选门禁。MoE 的 AP 通过，
但相对匹配 Dense 慢约 46%–48%，即使放宽到 20% 额外延迟也不能通过。因此停止 MoE 长训，
保留原始 checkpoint、训练日志、固定验证 CSV 和性能原始样本；后续若继续 P2，应转向
one-to-one assigner/loss 的精度主线，或另做设备端融合/编译优化后再重新测量。

## 证据索引

- `protocol.json` / `PROTOCOL.md`：六组定义、数据哈希和预算。
- `eval_fixed_val512/evaluation_evidence.json`：固定 512 图像精度与 checkpoint 哈希。
- `eval_fixed_val512/*/validator/*.csv`：逐图指标与 assignment 原始数据。
- `benchmark/benchmark_evidence.json`：GPU1 延迟、吞吐、显存和辅助 THOP。
- `benchmark/samples.csv`：100 次 batch=1 原始计时样本。
- `pilot/*/results.csv`：六组 1 epoch 训练日志末行。
