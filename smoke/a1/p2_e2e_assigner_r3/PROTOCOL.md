# A1 P2-E r3：one-to-one 正样本预算单因素 pilot

## 目的

P2-E r1 显示 one-to-one 分支每图约 6.88 个正样本，而 Dense 分支约 60.57 个；r2-c 的分类
损失增益没有稳定改善。因此本轮只检验 assigner 的一个因素：将 one-to-one `tal_topk2` 从
`1` 放宽到 `2`。这不会改模型结构、MoE、数据、优化器或冻结策略。

`tal_topk2=1` 是当前对照；`tal_topk2=2` 是训练监督正样本预算处理组。推理仍使用原生
one-to-one、NMS-free 输出；若处理组出现明显重复匹配或破坏 one-to-one 输出闭环，则判为失败。

## 固定设置

- 基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- 来源：`p1_factorial_medium_r28` 原始 initializer
- 单元：B（Dense + one-to-one）和 D（MoE + one-to-one）
- 数据：pilot train5000 / val512，640×640，batch=4
- 预算：5 epochs、SGD、`lr0=1e-4`、AMP=False、seed=260829
- P1 约束：训练层 4/6/8/23，BatchNorm 和官方 base 冻结；D 使用确定性 hard Top-2
- 设备：GPU1；B-1 → B-2 → D-1 → D-2 串行

## 运行时门禁

每个 run 必须记录并核对 `A1_E2E_O2O_TAL_TOPK2` 和 native
`E2ELoss.one2one.assigner.topk2`；仓库根目录固定插入 `sys.path[0]`，防止调用其它 editable
checkout。gain 固定为 `1.0`，因此 r2-c 的分类损失因素不混入本轮。

## 判定

统一使用固定 val512 比较 precision、recall、mAP50-95、TP/FP/FN、正样本数、assignment IoU、
重复/重叠率和分类/框/DFL loss。单 seed、5 epoch 只用于方向筛选；只有处理组在 B、D 均无
明显回归且 recall/mAP 或匹配质量有明确改善，才扩大到三 seed、15 epoch。否则封存为 assigner
预算无效缓解。

## 断连

使用 `nohup` 串行脚本，四个 run 和固定 val512 评估均有独立日志、manifest、status、checkpoint；
任一 run 失败立即停止后续，保留已有证据，不覆盖旧 r2/r2-c 目录。
