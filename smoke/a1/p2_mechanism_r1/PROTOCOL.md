# A1 P2-B 机制诊断 r1

## 目标

在不重训、不修改 P1 checkpoint 的前提下，验证 r28 的 MoE + End-to-End 负结果是否能由可复现的路由机制解释。首轮只做固定 pilot val512 的推理路由诊断；结果不得写成新的精度结论。

## 固定边界

- 公共基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- 来源 checkpoint：r28 三个 seed 的正式 C/D `last.pt`
- 数据：`/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master/configs/a1/p1_pretrained/pilot_data/val2017.txt`，固定 512 图像
- 设备：单卡 GPU1（RTX 4090）
- 输入：640×640，eval，router noise=0，hard Top-2
- 输出：每个 routed module 的 expert load、selection fraction、normalized entropy、Gini、dead-expert count；C/D 预测候选数和同类高 IoU 重复率

## P2-B 证据门禁

必须保存逐模块原始 CSV/JSON、checkpoint SHA256、数据 manifest SHA256 和运行命令。只有在三个 seed 中重复观察到同一机制，并完成至少一个受控缓解或无效验证，才能写成 P2 负结果。单次死专家不单独构成结论。

## 后续阶段

1. r1：固定 val512 路由/重复框诊断（本次）。
2. r2：单 batch 真实标签反向，记录 router、factor、head 梯度范数和零比例。
3. r3：固定 router 与 one-to-one/one-to-many 的匹配冲突对照。
4. r4：训练/评估路由漂移与 dense fallback 受控验证。
