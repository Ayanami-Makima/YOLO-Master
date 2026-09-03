# A1 P1 r28 运行时、导出与资源闭环补充报告

## 当前结论

r28 的 2×2、三种子正式训练与 checkpoint 审计已完成；本补充闭环进一步验证了 NMS 与 end-to-end 的真实运行时路径、ONNX 图、导出一致性和资源量。正式训练 checkpoint 始终未被改写。

截至 2026-09-02，P1 的训练、精度、运行时路径、ONNX 导出、资源与受控低竞争 CPU 延迟数据均已补齐。CPU 基准按锁定协议完成 3 个种子 × 4 个单元 ×（50 次预热 + 200 次采样），且 12 个单元均完整写出样本。任务书要求的“CPU latency measurable improvement”在该测量口径下满足；不过两个差异均不足 0.4%，只能称为可测但很小的改善，不能表述为显著部署加速。严格的全行 ONNX 原始输出一致性仍为 `partial`，见下文限制。

## 锁定范围

- 正式权重：r28 的 12 个 epoch-15 `last.pt`，不挑选 best epoch
- 设计：A/B/C/D × seeds 260829、260830、260831
- 训练/验证：COCO 20,000 / 5,000，15 epochs，batch 4，SGD，`lr0=0.0001`，AMP 关闭
- 模型：官方预训练 C3k2 base 冻结；层 4/6/8 为残差 factor adapter，层 23 为检测头
- 推理审计：固定 16 张 COCO val 图像，640×640，batch 1，`conf=0.001`，`iou=0.7`，`max_det=300`
- 延迟设置：50 次预热、200 次采样、batch 1；CPU 协议为 4 threads 与 CPU affinity 28–31
- 闭环 protocol SHA-256：`007efefc86a1b3e76773f0c28f610de03715393a259e0846c3954652c8e15dfe`

## 三种子精度结果

第 15 轮 `mAP50-95`：

| 种子 | A: Dense+NMS | B: Dense+E2E | C: MoE+NMS | D: MoE+E2E |
|---|---:|---:|---:|---:|
| 260829 | 0.40192 | 0.39512 | 0.40247 | 0.39452 |
| 260830 | 0.40177 | 0.39481 | 0.40182 | 0.39492 |
| 260831 | 0.40232 | 0.39472 | 0.40236 | 0.39451 |
| 均值 | 0.40200 | 0.39488 | 0.40222 | 0.39465 |
| 样本标准差 | 0.00028 | 0.00021 | 0.00035 | 0.00023 |

主要效应：C−A 为 +0.00021，D−B 为 −0.00023，MoE 主效应约 −0.00001；Dense 的 B−A 为 −0.00712，MoE 的 D−C 为 −0.00757，end-to-end 主效应为 −0.00734。故本预算下没有观察到稳定的 MoE 精度增益，end-to-end 比对应 NMS 路径低约 0.73 AP。

## 真实推理路径审计

12/12 checkpoint 在固定图像上完成运行时插桩：

- A/C：每个模型 16/16 次调用走 `nms`，且 16/16 次调用了真实抑制核；
- B/D：每个模型 16/16 次调用走 `end2end_score_filter`，抑制核调用数均为 0；
- B/D 的结果是 one-to-one score filtering，不是“导出图没有 NMS 但 Python 预测仍调用 NMS”的表面无 NMS。

因此，NMS 与 end-to-end 两条路径在正式权重上均已实际执行并被区分。

## ONNX 导出与一致性

全部 12 个模型均成功导出为固定 640×640、FP32、opset 17 的 ONNX；图遍历（主图、嵌套图、局部函数）中每个模型的 `NonMaxSuppression_count=0`。A/C 为 raw BCN 输出，B/D 为 `[1,300,6]` one-to-one 输出。

严格的原始输出比较不做置信度过滤：

- A/C：24/24 个输入比较通过；
- B/D：20/24 个输入比较通过；
- 4 个失败均发生在 end-to-end 双 TopK 的排名约 296–300 的极低分尾部，分数约 `5e-6`；这是 ONNX 与 PyTorch 浮点舍入造成的 TopK 成员替换，不是 NMS，也不只是可忽略的普通行序变化。

这项严格 all-row 门禁被如实标记为 `partial`。追加的、与实际预测相同的 `conf=0.001` 语义审计显示 B/D 的 24/24 个输入均通过（类别保持的一对一集合比较）；低分尾部不会进入锁定的预测输出。该追加审计说明运行时语义一致，但不替代严格原始输出差异的记录。

## 资源数据

固定有限输入 `[1,3,640,640]`、两次独立重复的 THOP executed FLOPs 结果在 12/12 checkpoint 上完全重复一致：

| 组 | 参数量 | 固定输入执行 GFLOPs |
|---|---:|---:|
| A | 2,993,452 | 7.417971 |
| B | 3,148,280 | 8.042931 |
| C | 5,424,348 | 8.251725 |
| D | 5,579,176 | 8.876685 |

这里的 GFLOPs 是 `2 × THOP MACs` 的固定输入执行计数，不宣称为数据分布平均值或稀疏理论 FLOPs。原资源工具使用未初始化假输入，在 MoE 路由器正确拒绝 NaN/Inf 时会静默返回 0；该旧计数不再用于结论。

GPU batch-1 峰值已分配显存（三种子均值）为 A 705.82 MiB、B 696.17 MiB、C 698.98 MiB、D 701.66 MiB。

## 无背景 GPU 延迟数据

GPU0 在启动前为 32 MiB、0% utilization、无 compute app 的状态下完成了 3 seeds × 4 cells × 200 样本的测量。运行进程以 `CUDA_VISIBLE_DEVICES=0` 启动，固定为 GPU0；总 wall-time 均值为：

| 组 | 均值 ms | 种子间样本标准差 ms |
|---|---:|---:|
| A | 12.496 | 0.188 |
| B | 12.049 | 0.327 |
| C | 25.880 | 0.103 |
| D | 26.111 | 0.979 |

B−A 为 −0.447 ms（约 −3.6%），D−C 为 +0.231 ms（约 +0.9%）。MoE 的 C−A 为 +13.384 ms，D−B 为 +14.062 ms，说明当前未优化的 MoE factor 有显著 GPU 推理开销。GPU 延迟是补充证据；A1 的 CPU 性能测量结果见下一节。

此前在有背景任务的 GPU1 上取得的 `gpu_latency_parallel_r3` 证据保持原样，仅作共享环境下的对照，不用于上述结论。

## 受控低竞争 CPU 延迟

此前一次全量 CPU 测量发现固定 affinity 28–31 受到外部任务竞争，因此在写出样本前中止并保留日志。随后在低竞争窗口以相同锁定协议完成 CPU-only 复测：4 个 Torch/OpenCV threads、affinity 28–31、batch 1；每个单元先预热 50 次，再采样 200 次。总计 3 个种子 × 4 个单元 × 200 = 2,400 个计时样本（不含预热），范围为 RAM 图像 → preprocess → inference → postprocess → Results，不含模型加载和磁盘 I/O。测量时记录的启动 load average 为 7.09 / 13.59 / 25.94；未停止其他用户作业，因此这里表述为“受控低竞争”，不宣称机器绝对空闲。

| 组 | CPU total 均值 ms | 种子间 SD ms |
|---|---:|---:|
| A | 5118.699 | 7.305 |
| B | 5099.800 | 2.874 |
| C | 7235.376 | 6.724 |
| D | 7219.193 | 2.416 |

B−A 为 −18.900 ms（−0.369%），D−C 为 −16.183 ms（−0.224%）。两条 end-to-end 路径均有可测但很小的 CPU total-latency 改善，满足任务书的字面性能指标；结果不足以支持“明显加速”或抵消 MoE 的总体 CPU 开销等更强结论。

## 证据位置

- 锁定 protocol：`smoke/a1/p1_factorial_medium_r28/closure_r1/protocol.json`
- 运行时 NMS 路径：`smoke/a1/p1_factorial_medium_r28/closure_r1/predict/evidence.json`
- 严格 ONNX：`smoke/a1/p1_factorial_medium_r28/closure_r1/export/evidence.json`
- 阈值语义 ONNX：`smoke/a1/p1_factorial_medium_r28/closure_r1/export_threshold_semantics_r1/evidence.json`
- 原资源与显存：`smoke/a1/p1_factorial_medium_r28/closure_r1/resources/evidence.json`
- 确定性 GFLOPs：`smoke/a1/p1_factorial_medium_r28/closure_r1/resources_deterministic_r1/evidence.json`
- 无背景 GPU0 延迟及全部样本：`smoke/a1/p1_factorial_medium_r28/closure_r1/gpu_latency_clean_r1/latency/`
- GPU 补充延迟及全部样本：`smoke/a1/p1_factorial_medium_r28/closure_r1/gpu_latency_parallel_r3/latency/`
- 低竞争 CPU 延迟及全部样本：`smoke/a1/p1_factorial_medium_r28/closure_r1/cpu_latency_low_contention_r1/latency/`
- CPU 高竞争中止记录：`smoke/a1/p1_factorial_medium_r28/closure_r1/latency_cpu_contended_aborted_20260902T104100Z/run.log`
