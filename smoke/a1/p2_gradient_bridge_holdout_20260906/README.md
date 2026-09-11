# P2 梯度通路：不重叠样本复核

2026-09-06。固定 train 列表索引 128–159，共 32 图、207 GT；与上一轮索引 0–31 不重叠。
分别使用 r28 B/D 的 seed260829、260830、260831 最终 checkpoint；CPU、640、batch1、无增强、
冻结 BN/base，无优化器更新。`holdout` 仅指诊断中留出的训练图，不是新的验证/测试集。

| seed | B factor cosine | D factor cosine | D router cosine | D router 负 cosine 图数 |
| --- | ---: | ---: | ---: | ---: |
| 260829 | 0.603870 | 0.635841 | 0.532462 | 5/32 |
| 260830 | 0.633075 | 0.619127 | 0.650216 | 4/32 |
| 260831 | 0.585105 | 0.517074 | 0.530530 | 5/32 |

三个 seed 均确认 native one-to-one 检测损失到 factor/router 的梯度为 0；0.1 bridge 可产生非零梯度，
前向 boxes/scores 最大误差为 0，模型 state_dict 不变。总体同向、少数反向这一描述在第二组图仍成立。
负方向样本比例会随图像变化，不能宣称完全无冲突；也不能据此宣称 detach 导致掉点或 bridge 能涨点。
两轮合计 64 张不同训练图，每轮的三个 seed 都重复相同图像，不能当成 192 张不同图。

证据：`evidence_seed*.json`、`summary.json`。可用 `audit_p2_gradient_bridge.py --offset 128 --images 32` 复现。
