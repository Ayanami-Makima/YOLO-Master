# A1 P2-E r2：one-to-one 分类损失单因素 pilot

## 目标

P2-E r1 显示 B/D 的 one-to-one 分支分类 loss 高于 A/C，且 one-to-one 每图只有约 6.88 个正样本。
本轮只检验一个因素：将 one-to-one 分支的分类损失乘以 `1.2`。`1.0` 为同预算中性对照，`1.2`
为治疗组；不修改 assigner 的唯一匹配约束，不改变 MoE、数据、优化器或 NMS-free 推理。

## 固定边界

- 公共基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- 来源：r28 原始 initializer（禁止使用 routing probe 或已训练 checkpoint）
- 单元：B（Dense + one-to-one）和 D（MoE + one-to-one）；A/C 的 r28 结果作为固定参照
- 数据：固定 pilot train5000 / val512 manifest；输入 640×640；batch=4
- 预算：5 epochs、SGD、`lr0=1e-4`、AMP=False、seed=260829、P1 冻结和 hard Top-2 路由策略
- 设备：GPU1；四个 run 串行执行；每个 run 均写入独立目录，可通过 status 文件检查

## 四个受控 run

| 单元 | `A1_E2E_O2O_CLS_GAIN` | 角色 |
| --- | ---: | --- |
| B | 1.0 | 中性对照 |
| B | 1.2 | one-to-one 分类损失治疗组 |
| D | 1.0 | 中性对照 |
| D | 1.2 | one-to-one 分类损失治疗组 |

## 判定

首轮只比较同一单元的 `1.2 - 1.0`：mAP50-95、precision、recall、固定 val512 的 TP/FP/FN、
正样本数、匹配 IoU、分类/定位 loss，并确认 D 仍为 one-to-one、NMS-free。单 seed、5 epoch、
pilot 数据只用于筛选方向，不能写成正式 P2 收益；只有方向明确且闭环无回归，才扩大到三 seed 和
正式预算验证，否则记录为无效缓解。

## 断连与恢复

训练命令使用 `nohup`，每个 run 的请求、环境、status 和 checkpoint 写入独立路径。SSH 断连不应
终止进程；若进程异常退出，依据对应 `last.pt` 和 status 做显式 resume，不从已完成 run 重头覆盖。
