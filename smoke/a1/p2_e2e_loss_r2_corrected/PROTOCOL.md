# A1 P2-E r2-c：one-to-one 分类损失单因素 pilot（导入路径修正版）

## 目的

P2-E r1 指出 End-to-End 的 one-to-one 分支分类项可能是精度差距来源。本轮只检验一个
因素：将 one-to-one 分类损失乘以 `1.2`，以 `1.0` 作为同预算中性对照。assigner、MoE、
数据、优化器、冻结策略和 NMS-free 推理均保持不变。

## 重要审计修正

先前 `p2_e2e_loss_r2` 的四个 run 虽然完成，但执行 `python scripts/a1/*.py` 时 Python 优先
解析了远端另一个 editable ultralytics checkout（`/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master`），
导致 `A1_E2E_O2O_CLS_GAIN=1.2` 未进入实际训练，B/D 的控制与处理权重完全相同。该目录保留为
审计证据，**不作为有效实验结果**。

本修正版在所有 runner/evaluator 中显式把当前仓库根目录插入 `sys.path` 首位，并在单 batch
审计中验证：`gain=1.0` 与 `gain=1.2` 的 one-to-one loss 和梯度确实不同。

## 固定设置

- 基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- 来源：`p1_factorial_medium_r28` 原始 initializer（禁止使用 routing probe 或已训练 checkpoint）
- 单元：B（Dense + one-to-one）和 D（MoE + one-to-one）
- 数据：pilot train5000 / val512，640×640，batch=4
- 预算：5 epochs、SGD、`lr0=1e-4`、AMP=False、seed=260829
- P1 约束：训练层 4/6/8/23，BatchNorm 和官方 base 冻结；D 使用确定性 hard Top-2
- 设备：GPU1；B-1.0 → B-1.2 → D-1.0 → D-1.2 串行

## 判定

统一使用固定 val512 评估 precision、recall、mAP、TP/FP/FN、匹配 IoU、正例数及分类/框/DFL
loss。单 seed、5 epoch 仅用于方向筛选；只有处理组相对同单元控制组方向明确且无回归，才进入
三 seed 或正式预算验证。旧的 `p2_e2e_loss_r2` 结果不参与判定。

## 断连

使用 `nohup` 启动串行脚本。每个 run 有独立 manifest、status、日志和 checkpoint；脚本遇到任一
失败立即停止后续 run，避免把失败结果误报为完成。
