# A1 P1 预训练 2×2 pilot 报告

更新时间：2026-08-29。

## 结论先行

本轮计划中的 P1 pilot **已完整执行**：统一预训练初始化审计、固定 COCO 子集、四格 1 epoch
预检、四格 5 epoch 训练、完整 COCO val、真实 NMS 调用跟踪、重复框、CPU/GPU 延迟、MoE 路由和
ONNX 导出审计均有结构化证据。

但本轮 **未通过 P1 精度/部署验收**，不能填写“P1 已完成/已通过”：四格完整 COCO val 的
mAP50-95 都接近 0；A/B/C 的 ONNX 在至少一个固定输入上未通过预声明的 PyTorch 数值一致性容差。
这是可复现的否定性 pilot 结果，不是 P1 正向成果。

## 锁定协议

| 项目 | 值 |
| --- | --- |
| 指导书锁定源码 | `acce839c7e895d6b179de7f7093fa879e237cc7b` |
| P0/P1 共同 checkpoint | `yolo26n.pt` |
| checkpoint SHA-256 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| 协议 SHA-256 | `83afec2749a8fa5c8c57d0113ff48f90de9f27fc885935a6c8580efafa0dcf72` |
| 训练子集 | COCO train2017 固定 5,000 图、36,404 框、80 类 |
| epoch 内验证 | COCO val2017 固定 512 图、3,536 框、80 类 |
| 最终评测 | 完整 COCO val2017，5,000 图、36,335 实例 |
| 训练 | seed `260829`、5 epoch、640、batch 4、SGD、`lr0=0.001`、AMP off、workers 0 |
| 硬件 | scut-server，2 × RTX 4090；C→A / D→B 双 lane |

数据选择先贪心覆盖 80 类，再以固定 seed 随机填充并排序。训练/验证 ID 无交集；pilot、preflight
和完整 COCO 的标签缓存命名空间互相隔离。四组均从原始 checkpoint 重新初始化，preflight 权重没有
进入正式训练。

## 2×2 定义与初始化

| Cell | FFN | 检测路径 | 实际转移张量 | 转移参数/缓冲占目标比例 |
| --- | --- | --- | ---: | ---: |
| A | A2C2f dense | one-to-many + NMS | 479 | 77.49% |
| B | A2C2f dense | one-to-one，NMS-free | 599 | 78.79% |
| C | A2C2fMoE | one-to-many + NMS | 479 | 39.71% |
| D | A2C2fMoE | one-to-one，NMS-free | 599 | 41.55% |

四组所有同名同形状张量均逐项验证为已转移。B/D 比 A/C 多 120 个可转移张量，因为官方 checkpoint
本身带有 one-to-one 分支。未转移的 A2C2f/A2C2fMoE、专家和路由参数按同一 seed 初始化。

## 训练与完整 COCO val

四格 1 epoch preflight 和 5 epoch 正式运行均通过：epoch 连续、loss/指标有限、batch 未自动缩减，
没有 OOM、SIGSEGV、Traceback 或数据错误，`best.pt`/`last.pt` 和完整原生日志均已哈希绑定。

最终使用各组 `best.pt` 在完整 COCO val2017 上重新评测：

| Cell | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.001550 | 0.0414 | 0.000471 | 0.000149 |
| B | 0.001180 | 0.0162 | 0.000378 | 0.000134 |
| C | 0.000808 | 0.0194 | 0.000148 | 0.000036 |
| D | 0.001260 | 0.0312 | 0.000565 | 0.000244 |

数值差分 B−A 为 −0.0015 AP point，D−C 为 +0.0208 AP point，但四个绝对精度均失效，
这些差分没有任务意义，不能用于声称“精度下降不超过 2 AP”。

## NMS、重复框与延迟

在完整 val 的前 16 张固定图上：

| Cell | 检测框 | NMS wrapper 路径 | 真实 suppression kernel | 重叠框诊断 |
| --- | ---: | --- | ---: | --- |
| A | 12 | 16 次 NMS 路由 | 5 | 1 个重复框 |
| B | 4 | 16 次 end-to-end score filter | 0 | 0 |
| C | 6 | 16 次 NMS 路由 | 5 | 0 |
| D | 2 | 16 次 end-to-end score filter | 0 | 0 |

B/D 的真实 suppression kernel 为 0，证明 NMS-free 路径生效。由于模型精度极低、输出框过少，
重复框统计不能被解释为 B/D 质量更好。

固定 batch 1、50 次 warmup、200 次计时样本：

| 设备 | A mean | B mean | C mean | D mean |
| --- | ---: | ---: | ---: | ---: |
| CPU，Torch/OpenCV 4 线程 | 136.34 ms | 136.21 ms | 145.23 ms | 145.37 ms |
| RTX 4090 GPU0 | 6.123 ms | 6.165 ms | 12.203 ms | 12.331 ms |

CPU 后处理均值 A/B 为 0.274/0.109 ms，C/D 为 0.242/0.103 ms，NMS-free 确实减少后处理；但
端到端收益被网络推理耗时淹没。当前 MoE 让 GPU 总时延约翻倍，尚无部署收益。

## 路由与导出

C/D 各审计 6 个路由器；64 张固定图、每个路由器 128 次 Top-2 选择。4/8 专家层均无样本内死专家，
归一化选择熵约 0.94–0.99。16 专家层只有 C 的一个路由器有 1 个专家未被该小样本选择，其余无死专家。
这表明没有明显整体路由塌缩，但低精度模型上的路由均衡不能证明 MoE 有效。

四组均成功生成可由 `onnx.checker` 读取的 ONNX，图内 `NonMaxSuppression` 节点数均为 0；B/D 输出为
`[1,300,6]` one-to-one 语义。C/D 的 ONNX 明确采用 dense expert fallback，**不保留稀疏跳过或其时延**。
固定 4 张输入的 PyTorch↔ONNX 严格数值检查结果为：D 4/4 通过，A/B/C 至少 1 张失败，因此总导出阶段
为 `partial`，没有放宽容差掩盖失败。

## 根因判断与下一门槛

官方 `yolo26n.pt` 能完整支撑 P0，因为 P0 只切换 checkpoint 已训练好的 two-head 路径；P1 的 matched
shell 把三个官方 C3k2 块换成 A2C2f/A2C2fMoE。即使 dense A/B 也有约 21–23% 目标参数没有兼容权重，
MoE C/D 则约 58–60% 需要随机训练。`lr0=0.001 + 5 epoch` 是微调预算，不足以训练这些新块，所以
A/B 与 C/D 一起失效；这不是 COCO、GPU、NMS 或单独 MoE 的故障。

下一版不应直接扩大到昂贵的 full-COCO 30 epoch。建议先做固定子集的恢复门：

1. 设计保持官方 C3k2 功能的 warm-start/identity-residual MoE 注入，或对新块采用分阶段高学习率；
2. 先用 A/C 做 `lr0=0.003/0.01` 的短程门控试验，完整 val mAP50-95 未明显起升则停止；
3. 只有 dense A 能恢复到可用精度且 C 不出现路由/稳定性问题，才重跑四格 10 epoch；
4. P1 正式通过仍需有效精度的 2×2、导出一致性和单 seed 限制声明（或补 3 seeds）。

## 证据与复现

- 协议/请求/固定列表：[`../../configs/a1/p1_pretrained/`](../../configs/a1/p1_pretrained/)
- 紧凑训练证据：[`p1_pretrained/training/`](p1_pretrained/training/)
- val/NMS/延迟/路由/导出：[`p1_pretrained/evidence/`](p1_pretrained/evidence/)
- 协议生成：[`../../scripts/a1/prepare_p1_pretrained.py`](../../scripts/a1/prepare_p1_pretrained.py)
- 初始化与数据审计：[`../../scripts/a1/audit_p1_pretrained.py`](../../scripts/a1/audit_p1_pretrained.py)
- lane runner：[`../../scripts/a1/run_p1_lane.py`](../../scripts/a1/run_p1_lane.py)
- 评测器：[`../../scripts/a1/evaluate_p1_matrix.py`](../../scripts/a1/evaluate_p1_matrix.py)
- 路由审计：[`../../scripts/a1/audit_p1_routing.py`](../../scripts/a1/audit_p1_routing.py)

远程完整产物保留在 `/data/data2/TuJiajun/A1-smoke-r4/p1_pretrained_r1/`；checkpoint 和 ONNX 二进制
未提交到 Git，提交的是其路径、大小、SHA-256、结构和数值检查结果。
