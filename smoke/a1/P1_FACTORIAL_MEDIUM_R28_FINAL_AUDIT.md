# A1 P1 r28 中等规模正式实验最终审计

## 结论

A1 P1 r28 的 12 个正式单元已全部完成，恢复感知补充审计通过 12/12。实验权重、协议和锁定仓库未在审计过程中修改。

运行时路径、ONNX、资源和延迟补充证据见 `P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md`。受控低竞争 CPU 基准已完成：B−A 为 −18.900 ms（−0.37%），D−C 为 −16.183 ms（−0.22%）；故在锁定协议下满足任务书“CPU latency measurable improvement”的测量要求，但差异不足以支持显著加速的主张。12 个 ONNX 图均不含 `NonMaxSuppression`，B/D 的实际运行时抑制调用数为 0；仍须保留严格 raw all-row ONNX 比较为 `partial` 的限制：A/C 为 24/24，B/D 为 20/24，4 个 B/D 极低分 TopK 尾行不同，而 `conf=0.001` 的实际预测语义比较为 24/24。

原版 `audit_p1_checkpoints_r28.py` 的总状态为 `failed`，原因是它只读取固定位置的单段 driver log，并把断点恢复后最后运行段的局部 `microbatch_index` 当作全程计数。该失败不来自模型权重：12 个最终 checkpoint 的冻结层、C3k2 base、BatchNorm、gain、路由/专家更新和 15 轮指标均通过直接检查。

为处理断点恢复证据，新增了只读、只追加的恢复感知审计。它绑定原审计文件、请求、initializer、最终 checkpoint 及恢复记录的 SHA-256，不改写原审计结论，最终状态为 `passed`，失败单元为 0。

## 正式实验设置

- 设计：A/B/C/D 2×2 因子实验，3 个种子，共 12 个单元
- 种子：260829、260830、260831
- 训练/验证：20,000 / 5,000 张 COCO 图像
- 训练轮数：15 epochs
- batch：4
- optimizer：SGD
- 初始学习率：0.0001
- AMP：关闭
- 训练层：4、6、8、23
- 其他模型层、所有 BatchNorm、ResidualFactorAdapter 内官方 C3k2 base：冻结
- 层 4/6/8：C3k2ResidualFactor；检测头：层 23
- A/C：非 end-to-end + NMS
- B/D：end-to-end one-to-one
- A/B：Dense A2C2f factor
- C/D：A2C2fMoE factor，硬 Top-2

## Checkpoint 与冻结审计

12 个单元全部满足：

- `results.csv` 均为 15 行正式 epoch，指标均为有限值；
- A/C 的模型及检测头 `end2end=False`，B/D 为 `True`；
- 三处官方 C3k2 base 全部逐张量不变，最大误差 0；
- `frozen_factor_base_parameters=459232`；
- 冻结模型层全部不变；
- BatchNorm 参数和统计量全部不变；
- 三处 residual gain 均非零；
- A/B 不含路由器；
- C/D 三处 factor 的路由参数和专家参数均发生更新；
- 请求、initializer 和最终 checkpoint 的当前 SHA-256 与首次正式审计记录一致。

Dense A/B 的恢复记录中，最终 `p1_runtime_policy.json` 的 microbatch 数是最后恢复段的局部计数，或文件缺失；但每次 `resume_started.json` 都记录了三个 factor adapter、冻结 base、冻结 factor BN、残差 gain 初始学习率 0.01、无 warmup，以及空路由参数组。A/B 没有 routed module，私有路由探索关闭，因此该局部计数不影响 Dense 模型训练语义。完整 1–15 epoch 序列对应的总 microbatch 数为 75,000。

## 路由与准入证据

正式训练前的独立 routing-probe 证据保持不可变：

- `formal_admission.json`：`status=passed`，`formal_may_start=true`
- `routing/hard_top2_5000.json`：`status=passed`
- `residual_activity/routing_probe_512.json`：`status=passed`

原准入策略明确把少量死专家作为非阻断 A1 诊断项，而不是正式启动硬门禁。最终 C/D checkpoint 进一步确认所有三层路由器及专家参数均获得更新。本次快速审计未重新运行耗时的正式权重全 val5000 路由分布诊断。

## 正式结果

以下均为各单元第 15 轮 `metrics/mAP50-95(B)`：

| 种子 | A | B | C | D |
|---|---:|---:|---:|---:|
| 260829 | 0.40192 | 0.39512 | 0.40247 | 0.39452 |
| 260830 | 0.40177 | 0.39481 | 0.40182 | 0.39492 |
| 260831 | 0.40232 | 0.39472 | 0.40236 | 0.39451 |
| 均值 | 0.40200 | 0.39488 | 0.40222 | 0.39465 |
| 样本标准差 | 0.00028 | 0.00021 | 0.00035 | 0.00023 |

因子效应：

- NMS 条件下 MoE 效应 C−A：+0.00021
- end-to-end 条件下 MoE 效应 D−B：−0.00023
- Dense 的 end-to-end 效应 B−A：−0.00712
- MoE 的 end-to-end 效应 D−C：−0.00757
- MoE 主效应：−0.00001
- end-to-end 主效应：−0.00734
- 交互效应：−0.00045

## P1 解释

在当前 20k/5k、15 epoch、冻结预训练 C3k2 base 的等预算设置下，Dense 与 MoE 的总体 mAP50-95 基本相同，没有观察到稳定的 MoE 增益。主要差异来自检测路径：end-to-end 的 B/D 均比对应 NMS 的 A/C 低约 0.0073 mAP，且三个种子方向一致。

因此，本轮 P1 支持以下结论：A2C2fMoE factor 在当前预算下没有显著优于 Dense A2C2f factor；end-to-end 路径在该设置下明显弱于非 end-to-end + NMS。该结论限定于本次训练预算和冻结策略，不外推到从头训练或更长训练日程。

## 关键审计证据

- 原正式 checkpoint 审计：`audits/formal_checkpoints.json`
- 恢复感知补充审计：`audits/formal_checkpoints_recovery_aware_v2.json`
- 补充审计器：`audits/tools/audit_p1_r28_recovery_aware.py`
- 正式准入：`audits/formal_admission.json`
- 硬 Top-2 路由：`audits/routing/hard_top2_5000.json`
- 残差活动：`audits/residual_activity/routing_probe_512.json`
