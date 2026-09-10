# P2-E r4：one-to-one 冲突消解指标单因素实验

本轮只改变 one-to-one assigner 在一个 anchor 被多个 GT 同时选中时的冲突消解优先级：

- `overlap`：按预测框与 GT 的 IoU 选择，保持当前原生行为；
- `align`：按 task-aligned metric（分类分数与定位重叠的组合）选择。

B（Dense + End-to-End）和 D（MoE + End-to-End）各运行一组 overlap 对照和一组 align 处理，
四组均从 r28 原始 initializer 独立启动。MoE、数据、模型层、冻结策略、训练预算和 NMS-free
推理路径不变。

固定条件：seed `260829`、5000/512 图像、5 epochs、batch 4、imgsz 640、SGD、lr0 `1e-4`、
AMP off、GPU1；评估使用固定 val512、batch 1、imgsz 640。评估同时保存候选筛选后正样本数、
冲突 anchor 数、冲突消解后正样本数、最终正样本数、matched IoU、recall、TP/FP/FN、分类和
定位损失。

判定：只有在不破坏 End-to-End 闭环的前提下，`align` 相对各自 `overlap` 对照能够以可重复
的 recall、匹配 IoU、FN 或 mAP 改善为依据，才考虑下一轮；否则封存为无效缓解。
