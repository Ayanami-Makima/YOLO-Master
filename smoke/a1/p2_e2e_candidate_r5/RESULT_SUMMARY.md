# P2-E r5：one-to-one 候选预算单因素结果

## 结论

本轮只改变 one-to-one TaskAlignedAssigner 的候选预算：官方默认 `tal_topk=7`，处理组为
`tal_topk=10`；`tal_topk2=1`、冲突规则 `overlap`、模型、数据、优化器、冻结策略和预算均不变。
B（Dense + End-to-End）和 D（MoE + End-to-End）各有一组对照和处理，四组均从 r28 原始
initializer 独立启动。

`tal_topk=10` 将候选正样本从约 `45.606` 增加到 `63.066`/图（约 `+38.3%`），但候选 GT 覆盖
本来已为 `0.998303`，最终 one-to-one 正样本、recall、mAP、FP/FN 和 loss 均没有改善；冲突
anchor 增加约 `67%–72%`，最终 GT 覆盖略降。B/D 处理组与各自对照在固定 val512 的 mAP、
precision、recall 完全一致。因此 r5 拒绝 `tal_topk=10`，保留官方 `tal_topk=7, topk2=1`，
不扩大到三 seed 或长训。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 基线 | SHA `acce839c7e895d6b179de7f7093fa879e237cc7b`，r28 原始 B/D initializer |
| 单因素 | one-to-one `tal_topk=7` vs `10`；`tal_topk2=1` |
| 数据 | COCO pilot：5000 train / 固定 512 val（3536 GT） |
| 训练 | 5 epochs、batch 4、imgsz 640、SGD、lr0 `1e-4`、seed `260829`、GPU1、AMP off |
| 评估 | GPU1、固定 val512、batch 1、imgsz 640、conf `0.001`、max_det `300` |
| 不变项 | 官方 C3k2 base/BN 冻结、ResidualFactor 路径、E2E one-to-one 输出和后处理 |

## 固定 val512 结果

| 运行 | tal_topk | mAP50-95 | precision | recall | 候选正样本/图 | 最终正样本/图 | 候选 GT 覆盖 | 最终 GT 覆盖 | 冲突 anchor/图 | matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B control | 7 | 0.426536 | 0.632169 | 0.533208 | 45.6063 | 6.9350 | 0.998303 | 0.996324 | 1.2598 | 0.837826 |
| B candidate10 | 10 | 0.426536 | 0.632169 | 0.533208 | 63.0669 | 6.9331 | 0.998303 | 0.996041 | 2.1614 | 0.837910 |
| D control | 7 | 0.425947 | 0.641006 | 0.531202 | 45.6063 | 6.9311 | 0.998303 | 0.995758 | 1.3012 | 0.837997 |
| D candidate10 | 10 | 0.425947 | 0.641006 | 0.531202 | 63.0650 | 6.9291 | 0.998303 | 0.995475 | 2.1909 | 0.837951 |

候选预算增加没有增加最终监督：`topk2=1` 的二次筛选仍将每个 anchor/GT 的最终分配压缩到
约 6.93 个正样本/图。候选覆盖已接近饱和，继续扩大 `tal_topk` 只制造更多候选冲突，不能
弥补 one-to-one 的精度差距。

## 决策

1. 拒绝 `tal_topk=10` 处理组，保留官方 `tal_topk=7, topk2=1`。
2. 不启动该处理的三 seed 或长训练，不把候选正样本数量增加写成精度收益。
3. 后续优先做候选质量/置信度校准或 one-to-one loss 权重的单因素 pilot，继续固定数据、seed、
   冻结和预算，并保留候选、冲突、最终匹配、分类/定位 loss 与 FP/FN 中间量。

## 证据

- `protocol.json`：实验协议、请求哈希和固定条件。
- `evaluation/evidence.json`：四个 checkpoint 的固定 val512 指标与 assignment 汇总。
- `evaluation/*/validator/per_image_metrics.csv`：512 张图逐图候选/匹配、TP/FP/FN 与损失。
- `evaluation/*/validator/assignment_metrics.csv`：逐图 assignment 中间量和匹配统计。
- `training/*_results.csv`：四组 5 epoch 训练结果。
- 服务器运行目录：`/data/data2/TuJiajun/A1-smoke-r4/p2_e2e_candidate_r5/`。
