# P2-E r4：one-to-one 冲突消解单因素结果

## 结论

本轮只改变 one-to-one assigner 在一个 anchor 同时被多个 GT 选中时的冲突消解指标：
`overlap` 为当前原生 IoU 规则，`align` 为 task-aligned metric 规则。B（Dense + End-to-End）
和 D（MoE + End-to-End）各运行一组对照和处理，保持数据、MoE、预算、冻结策略和 NMS-free
推理路径不变。

`align` 没有形成可复现的精度改善：B 的固定 val512 mAP50-95 从 `0.425601` 降至
`0.424558`（`-0.001043`），D 从 `0.425896` 变为 `0.425849`（`-0.000047`）。最终正样本数
几乎不变（B `6.8789`，D `6.8789 → 6.8750`），assignment matched IoU 反而小幅下降；
recall、FN 和 mAP 没有一致改善。因此 r4 作为无效缓解封存，不扩大到三 seed，也不进入长训。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 基线 | SHA `acce839c7e895d6b179de7f7093fa879e237cc7b`，r28 原始 B/D initializer |
| 单因素 | 冲突消解优先级：`overlap` vs `align` |
| 数据 | COCO pilot：5000 train / 固定 512 val |
| 训练 | 5 epochs、batch 4、imgsz 640、SGD、lr0 `1e-4`、seed `260829`、AMP off |
| 评估 | GPU1、固定 val512、batch 1、imgsz 640、conf `0.001`、max_det `300` |
| 不变项 | 官方 C3k2 base/BN 冻结、MoE 路径、E2E one-to-one 输出和后处理 |

## 精度结果

| 运行 | 冲突指标 | mAP50-95 | precision | recall | 每图正样本 | matched IoU | cls loss | box loss | FN@IoU50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B control | overlap | 0.425601 | 0.622710 | 0.536556 | 6.8789 | 0.837884 | 1.309890 | 1.064839 | 0.9922 |
| B treatment | align | 0.424558 | 0.622397 | 0.536221 | 6.8789 | 0.837191 | 1.304110 | 1.069328 | 0.9980 |
| D control | overlap | 0.425896 | 0.629765 | 0.531780 | 6.8789 | 0.837987 | 1.313399 | 1.062455 | 1.0000 |
| D treatment | align | 0.425849 | 0.642872 | 0.529913 | 6.8750 | 0.837386 | 1.303569 | 1.065557 | 0.9961 |

处理组相对同模型对照：

- B：mAP `-0.001043`、recall `-0.000335`、matched IoU `-0.000693`、FN `+0.0059`；
- D：mAP `-0.000047`、recall `-0.001867`、matched IoU `-0.000601`、FN `-0.0039`。

D 的 precision 上升伴随 recall 下降，且没有转化为 mAP 改善，不能据此宣称处理有效。

## Assign 过程审计

| 运行 | 候选筛选后正样本 | 冲突 anchor/图像 | 冲突消解后正样本 | 二次 top-k 后正样本 |
| --- | ---: | ---: | ---: | ---: |
| B overlap | 45.628 | 1.280 | 44.274 | 6.933 |
| B align | 45.616 | 1.262 | 44.280 | 6.933 |
| D overlap | 45.614 | 1.287 | 44.252 | 6.933 |
| D align | 45.616 | 1.274 | 44.270 | 6.929 |

候选数量和冲突数量几乎不变，说明该单因素只替换少量冲突 anchor 的 GT 归属，并没有解决
one-to-one 正样本稀疏问题。`topk2=1` 后的正样本仍约为每图 GT 数量，未形成额外监督预算。

## 决策

1. 拒绝 `align` 冲突消解处理组，保留 `overlap` 为当前 one-to-one 基线。
2. 不启动该处理的三 seed 或长训练；不把 D 的 precision 变化写成收益。
3. 后续若继续 P2，应把重点放在候选生成/冲突消解前的有效候选覆盖或分类置信度校准，仍保持一次只改一个因素，并继续保存上述中间量。

## 证据

- `protocol.json` / `PROTOCOL.md`：实验协议和固定条件。
- `eval_fixed_val512_corrected/evidence.json`：四个 checkpoint 的固定 val512 指标与哈希。
- `eval_fixed_val512_corrected/*/validator/per_image_metrics.csv`：逐图 TP/FP/FN、recall 和尺寸分档。
- `eval_fixed_val512_corrected/*/validator/assignment_metrics.csv`：逐图 assignment 中间量、损失和匹配 IoU。
- 远端训练日志：`/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28/p2_e2e_conflict_r4_*log`。
