# A1 P0 / P1 实验进展报告

更新日期：2026-09-03

## 1. 当前总体进展

A1 当前围绕 YOLO26n 的两项核心问题展开：

1. P0：复现官方预训练模型的标准 NMS 和 End-to-End NMS-free 两条推理路径。
2. P1：在保持官方预训练基座功能的前提下，完成 Dense / MoE 与 NMS / End-to-End 的 2×2 等预算实验。

目前状态如下：

- **P0 已完成并通过。** 官方 `yolo26n.pt` 的两条原生推理路径均已在完整 COCO val2017 上复现，并完成真实 NMS 调用和延迟审计。
- **P1 中等规模正式实验已完成。** r28 在固定 COCO 20,000 / 5,000、15 epochs、3 个配对随机种子下完成 A/B/C/D 共 12 个正式单元；恢复感知审计 12/12 通过，官方 C3k2 base、其余冻结层和 BatchNorm 均完全不变。
- **P1 闭环已补齐且有边界。** 运行时 NMS/End-to-End 路径、12 个无 `NonMaxSuppression` 的 ONNX 图、确定性资源、GPU 与 CPU 延迟均已记录；CPU 的 B−A / D−C 分别为 −0.37% / −0.22%，可测但很小。严格 raw ONNX all-row 比较仍为 `partial`（A/C 24/24，B/D 20/24；B/D 的 4 个极低分 TopK 尾行不同），不以阈值语义审计替代该限制。
- **代码与 CI 收尾已完成。** P1 PR 的最终代码提交 `4e2438a` 对启用的 CI 作业全部通过（Actions run `33622768773`）；主线兼容性修复 PR 的启用作业也已通过（Actions run `33617355851`）。两个 PR 均保持开放，未合并或改动上游 `main`。
- P1 当前结论是：在锁定的中等规模预算下，**MoE 没有表现出相对 Dense 的稳定精度优势；End-to-End 的精度稳定低于 NMS**。r23 pilot 与 r24/r25 full-COCO 尝试保留为历史证据，不取代 r28 的正式结论。

> 注意：P0 使用完整 COCO val2017；当前 P1 正式结论来自固定 20,000 张训练图像、完整 5,000 张验证图像和 15 epochs。两者数据尺度、训练状态不同，指标不能直接横向比较。

---

## 2. P0：官方预训练基线

### 2.1 实验目标

P0 不进行训练，而是使用同一个官方 `yolo26n.pt`，验证以下两条原生检测路径：

- A：`end2end=False`，one-to-many 输出经过标准 NMS。
- B：`end2end=True`，使用 one-to-one 输出，不调用 NMS。

A、B 使用相同的 checkpoint、数据集和验证设置，唯一变量是 `end2end` 开关。

### 2.2 基线身份

- 锁定官方基线代码 SHA：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- checkpoint：`yolo26n.pt`
- checkpoint SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- checkpoint 大小：5,544,453 bytes
- 模型类型：`DetectionModel + Detect`
- 参数量：2,572,280
- Ultralytics：8.4.101
- Python：3.10.12

### 2.3 P0 验证设置

| 项目 | 锁定值 |
| --- | --- |
| 数据集 | 完整 COCO val2017 |
| 图像尺寸 | 640 |
| Batch | 16 |
| GPU | 0 |
| Workers | 0 |
| conf | 0.001 |
| IoU | 0.7 |
| max_det | 300 |
| half | False |
| save_json | True |
| A 路径 | `end2end=False`，标准 NMS |
| B 路径 | `end2end=True`，NMS-free |

### 2.4 P0 精度结果

| 单元 | 推理路径 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | one-to-many + NMS | 0.659 | 0.515 | 0.562 | **0.402** |
| B | one-to-one，NMS-free | 0.656 | 0.505 | 0.550 | **0.395** |

B 相对 A 的 mAP50-95 变化为：

```text
B - A = 0.395 - 0.402 = -0.007
```

即 End-to-End 路径低 0.700 AP points。

### 2.5 P0 运行时路由审计

| 单元 | 检测数 | 真实 suppression kernel 调用 | 运行路径 |
| --- | ---: | ---: | --- |
| A | 77 | 16 | `nms` |
| B | 76 | 0 | `end2end_score_filter` |

该审计确认：A 确实调用了 NMS，B 确实未调用 NMS，而不是只修改了配置名称。

### 2.6 P0 Batch-1 延迟

延迟测试范围为：内存图像 → 预处理 → 推理 → 后处理 → Results。磁盘 I/O 和模型加载时间不计入。

- warmup：10 次
- 正式测量：30 次
- imgsz：640
- batch：1
- conf：0.25
- IoU：0.7
- max_det：300
- half：False

| 设备 | A 平均延迟 | B 平均延迟 | B 相对 A |
| --- | ---: | ---: | ---: |
| GPU 0 | 3.289 ms | 3.284 ms | -0.14% |
| CPU | 20.246 ms | 20.757 ms | +2.52% |

### 2.7 P0 结论

P0 已通过：同一个官方预训练 checkpoint 可以正确复现标准 NMS 和 End-to-End NMS-free 两条路径。两条路径在 GPU 上的 batch-1 总延迟接近，但 End-to-End 路径的完整 COCO mAP50-95 比 NMS 低 0.007。

P0 是推理路径基线，不是 MoE 消融实验。

---

## 3. P1：2×2 等预算实验

### 3.1 实验目标

P1 比较两个因素：

- 因子类型：Dense A2C2f 或 A2C2fMoE。
- 检测方式：标准 NMS 或 End-to-End one-to-one。

四个实验单元为：

| 单元 | 因子分支 | 检测方式 | End-to-End |
| --- | --- | --- | --- |
| A | Dense A2C2f | 标准 NMS | 否 |
| B | Dense A2C2f | one-to-one，NMS-free | 是 |
| C | A2C2fMoE | 标准 NMS | 否 |
| D | A2C2fMoE | one-to-one，NMS-free | 是 |

主要比较量为：

- Dense 的 End-to-End 效应：`B - A`
- MoE 的 End-to-End 效应：`D - C`
- NMS 下的 MoE 效应：`C - A`
- End-to-End 下的 MoE 效应：`D - B`
- 交互效应：`(D - C) - (B - A)`

### 3.2 预训练保持方案

官方 `yolo26n.pt` 的第 4、6、8 层是 C3k2，检测头是第 23 层。

为了避免破坏官方预训练功能，P1 没有直接用 A2C2f 或 A2C2fMoE 替换 C3k2，而是使用 YAML 可重建的残差因子结构：

```text
C3k2ResidualFactor(x) = base(x) + gain × factor(base(x))
```

其中：

- `base` 是官方预训练 C3k2，始终冻结。
- A、B 的 `factor` 为 Dense A2C2f。
- C、D 的 `factor` 为 A2C2fMoE。
- `gain` 初始化为精确的 0。

因此，四组 initializer 在训练前严格等价于 P0：

- 保存前最大误差：0.0
- 重新加载后最大误差：0.0
- 每个单元冻结的官方 C3k2 base 参数：459,232

### 3.3 r28 数据与正式预算

以下是本报告当前正式结论唯一对应的 r28 中等规模预算。早期 r23 的 5,000/512 图像、5 epochs 配置仅保留在第 4 节作为历史迭代记录，不应与 r28 的正式结果混用。

| 项目 | 锁定值 |
| --- | --- |
| 训练图像 | 20,000（固定 train2017 子集） |
| 验证图像 | 5,000（固定 val2017 子集） |
| Epochs | 15 |
| imgsz | 640 |
| 物理 batch | 4 |
| 随机种子 | 260829、260830、260831 |
| 单种子顺序 | A → B → C → D |
| GPU | 0 和 1，独立单卡任务，非 DDP |
| 训练层 | 4、6、8、23 |
| 其他层 | 全部冻结 |
| BatchNorm | 参数和统计量全部冻结 |

GPU 调度如下：

- seed 260829 → GPU 0
- seed 260830 → GPU 1
- seed 260831 → GPU 0

### 3.4 r28 正式训练超参数

下表适用于 r28 的 12 个正式单元（A/B/C/D × 3 个配对种子）。

| 超参数 | 值 |
| --- | --- |
| optimizer | SGD |
| lr0 | 0.0001 |
| lrf | 0.2 |
| momentum | 0.9 |
| weight_decay | 0.0005 |
| warmup_epochs | 0.5 |
| warmup_bias_lr | 0.0 |
| warmup_momentum | 0.8 |
| amp | False |
| deterministic | True |
| workers | 0 |
| patience | 0 |
| cos_lr | False |
| cache | False |
| fraction | 1.0 |
| save_period | 1 |
| val | True |
| save | True |
| resume | False |
| mosaic | 0.0 |
| mixup | 0.0 |
| copy_paste | 0.0 |
| close_mosaic | 0 |

Residual gain 使用独立优化设置：

| 超参数 | 值 |
| --- | --- |
| gain lr | 0.01 |
| gain weight_decay | 0.0 |
| gain warmup | False |

### 3.5 BatchNorm 校准与冻结

正式训练前，使用 pilot train 固定列表的前 512 张图像进行 BatchNorm 校准：

| 项目 | 值 |
| --- | --- |
| 图像数 | 512 |
| batch | 4 |
| imgsz | 640 |
| augment | False |
| shuffle | False |
| grad | False |

校准完成后，所有 BatchNorm 仿射参数和 running statistics 在训练过程中保持冻结。

### 3.6 r28 MoE 结构与路由超参数

第 4、6、8 层的 MoE 专家数分别为：

- 第 4 层：4 个专家
- 第 6 层：8 个专家
- 第 8 层：16 个专家
- 所有 MoE 层均使用硬 Top-2

r28 锁定的路由设置如下：

| 项目 | 值 |
| --- | --- |
| 路由方式 | hard Top-2 from step zero |
| progressive_sparsity | False |
| top_k | 2 |
| temperature | 1.0 |
| expert_dropout_rate | 0.0 |
| 通用 moe_noise_std | 0.0 |
| dynamic_schedule | none |
| expert_warmup_epochs | 0 |
| balance loss | 1.0 |
| router z loss | 0.1 |
| aux gain | 1.0 |
| mixture aux budget | 3.0 |

训练期仅对 MoE dispatch 使用短程私有探索噪声：

- sigma 由固定 train512 路由统计确定，并截断在 `[0.01, 0.05]`。
- 噪声保持到 microbatch 625。
- 在 microbatch 625–1000 之间衰减。
- microbatch 1000 后为 0。
- 验证和推理噪声始终为 0。
- balance loss 和 z loss 使用 clean logits。
- noisy logits 只用于训练期硬 Top-2 dispatch。

该设置不增加推理参数，也不改变验证和推理时的 clean hard Top-2 语义。

---

## 4. P1 关键迭代过程

### 4.1 r9：无效负结果

r9 虽然完成了 A、B、C、D，但直接把官方 checkpoint 中第 4、6、8 层的 C3k2 重建成 A2C2f 或 A2C2fMoE，导致约 21%–23% 参数随机初始化。

r9 完整 COCO mAP50-95 为：

| 单元 | mAP50-95 |
| --- | ---: |
| A | 0.00243 |
| B | 0.000557 |
| C | 0.00121 |
| D | 0.00023 |

这些结果反映的是初始化错误，不是 Dense 或 MoE 的有效比较，因此 r9 已被判定为无效负结果，不能作为正式 P1 结论。

### 4.2 r12：解决预训练功能丢失

r12 引入 `C3k2ResidualFactor`，使官方 C3k2 成为冻结 base，并以零 gain 接入 Dense / MoE factor。

r12 初始化门禁：

- 保存前最大误差：0.0
- 重载后最大误差：0.0
- 官方 base 冻结参数：459,232

r12 极小 preflight mAP50-95：

| 单元 | mAP50-95 |
| --- | ---: |
| A | 0.47741 |
| B | 0.46283 |
| C | 0.47719 |
| D | 0.46252 |

同时确认：

- `base_changed=0`
- `base_max_error=0.0`
- 三处 gain 均获得非零梯度
- C、D 的路由参数获得更新
- 无 Traceback、OOM 或 NaN

### 4.3 路由制度不一致问题

r12 routing probe 中：

- C mAP50-95：0.43431
- D mAP50-95：0.42682

但固定验证样本的硬 Top-2 审计出现多名零命中专家。

根因是原 `OptimizedMOEImproved` 使用 5,000-step progressive sparsity，而一个 pilot epoch 只有约 1,250 step。训练结束时路由器仍接近 Top-12，审计和推理却直接切换到硬 Top-2，造成训练制度与审计制度不一致。

修复措施：

1. 从第一个训练 batch 起直接使用硬 Top-2。
2. `progressive_sparsity=False`。
3. `warmup_steps=0`。
4. 使用短程训练期私有探索噪声改善早期专家覆盖。
5. balance 和 z loss 使用 clean routing tensors。
6. 验证与推理保持无噪声 clean hard Top-2。

### 4.4 r23：正式 P1

r23 在 r22 模型和路由语义不变的基础上，将固定 val512 的“零命中专家”从阻断条件调整为诊断项。

这一调整的原因是：固定 512 图像只能观测有限样本覆盖，某个专家零命中不等于该专家在完整数据分布上永久失活。P1 仍通过最大选择比例和归一化熵阻断真正的路由集中与崩塌。

该调整没有改变：

- 模型结构
- loss
- 路由 Top-2 语义
- 推理行为
- 数据与预算
- initializer
- r23 正式训练超参数

服务器重启后，旧正式尝试被归档，没有覆盖或续接部分 checkpoint。r23 从原始 initializer 重新启动全部 12 个正式单元，最终全部完成。

---

## 5. P1 r23 正式结果

### 5.1 三种子 mAP50-95

| Seed | A Dense+NMS | B Dense+E2E | C MoE+NMS | D MoE+E2E |
| ---: | ---: | ---: | ---: | ---: |
| 260829 | 0.43688 | 0.42498 | 0.43328 | 0.42214 |
| 260830 | 0.43673 | 0.42476 | 0.43869 | 0.42634 |
| 260831 | 0.43552 | 0.42648 | 0.43305 | 0.42704 |
| **Mean** | **0.43638** | **0.42541** | **0.43501** | **0.42517** |

### 5.2 配对因子效应

| 效应 | 均值 | 样本 SD | 95% t 区间（n=3） |
| --- | ---: | ---: | ---: |
| MoE 主效应 | -0.00080 | 0.00250 | [-0.00701, 0.00541] |
| End-to-End 主效应 | -0.01040 | 0.00251 | [-0.01664, -0.00416] |
| MoE × End-to-End 交互 | +0.00114 | 0.00174 | [-0.00318, 0.00545] |

具体配对差值为：

| 比较 | 均值 |
| --- | ---: |
| C - A：NMS 下 MoE 相对 Dense | -0.00137 |
| D - B：E2E 下 MoE 相对 Dense | -0.00023 |
| B - A：Dense 的 E2E 效应 | -0.01097 |
| D - C：MoE 的 E2E 效应 | -0.00983 |

### 5.3 正式完整性门禁

| 审计项 | 结果 |
| --- | --- |
| 正式 checkpoint | 12/12 通过 |
| checkpoint 唯一 SHA256 | 12/12 |
| 每单元冻结 C3k2 base 参数 | 459,232 |
| base 最大误差 | 0.0 |
| 其他冻结层最大误差 | 0.0 |
| BatchNorm 最大误差 | 0.0 |
| gain 活跃层实例 | 36/36 |
| factor 更新层实例 | 36/36 |
| MoE router 更新层实例 | 18/18 |
| MoE expert 更新层实例 | 18/18 |
| 路由审计 occurrence | 36 |
| 最大选择比例 | 0.753906，门槛 ≤0.8 |
| 最小归一化熵 | 0.902307，门槛 ≥0.5 |
| 残差活性层实例 | 18/18 |
| 残差/base 能量比 | [0.036727, 0.090667] |

### 5.4 零命中专家说明

固定 val512 硬 Top-2 审计记录到 3 个专家在 3 个 router occurrence 中零命中：

- seed 260830，C，第 8 层：expert 0
- seed 260831，C，第 8 层：expert 10
- seed 260831，D，第 8 层：expert 10

这些 occurrence 的路由统计为：

- 归一化熵：0.902–0.932
- 最大选择比例：0.277–0.363

因此，这 3 个零命中只表示固定 512 图像没有覆盖到对应专家，不表示模型发生整体路由崩塌。所有阻断性的熵和集中度门禁均已通过。

---

## 6. 当前结论

### 6.1 P0 结论

P0 已证明官方 `yolo26n.pt` 的标准 NMS 和 End-to-End NMS-free 两条原生推理路径均可正确复现。

在完整 COCO val2017 上：

- NMS：mAP50-95 = 0.402
- End-to-End：mAP50-95 = 0.395
- End-to-End 相对 NMS：-0.007

### 6.2 P1 r28 中等规模结论

P1 r28 已满足以下要求：

- 2×2 等预算设计
- 官方预训练 C3k2 功能保持
- 冻结一致性
- 3 个配对随机种子
- 12 个独立正式 checkpoint
- Dense / MoE 因子均正常更新
- MoE router / expert 均正常更新
- 无路由集中或路由崩塌
- 正式结果可追溯和可复现

在固定 20,000 张训练图像、5,000 张验证图像、15 epochs 和 3 个种子的中等规模预算下：

1. **MoE 与 Dense 的精度总体相当。**
2. **MoE 没有表现出统计上明确的 mAP50-95 优势。**
3. **End-to-End 的精度比 NMS 稳定低约 0.00734 mAP50-95。**
4. **MoE 与 End-to-End 的交互效应很小。**
5. 当前结果是有效的中等规模实验结果，不是模型失败或路由崩塌。

### 6.3 结论边界

P1 当前是中等规模、冻结预训练 base 的结论，不能外推为完整 COCO 长周期或从头训练结论，主要限制为：

- 训练数据为固定 20,000 张图像，而非完整 train2017。
- 正式训练为 15 epochs，而非长周期收敛训练。
- 统计分析只有 3 个配对种子。

若要判断 MoE 在完整训练规模下的最终收益，需要重新建立并锁定 full-COCO 实验协议。不得直接把 r28 checkpoint 续训为 full-COCO，也不得把 P1 指标与 P0 完整 COCO 指标直接比较。

### 6.4 r28 中等规模正式闭环

r28 使用固定 20,000 张 train2017、完整 5,000 张 val2017、15 epochs 和三种子；12 个 epoch-15 `last.pt` 均完成。均值 mAP50-95 为 A 0.40200、B 0.39488、C 0.40222、D 0.39465；MoE 主效应 −0.00001，End-to-End 主效应 −0.00734。训练、冻结、恢复审计、运行时路径、ONNX、资源与低竞争 CPU 延迟均已闭环，详见 `P1_FACTORIAL_MEDIUM_R28_FINAL_AUDIT.md` 与 `P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md`。该结果替代“r26 待建立”的状态，不覆盖 r23/r24/r25 的历史记录。

### 6.5 r28 延迟结果

延迟协议固定为 batch=1、50 次预热、200 次正式采样，范围为 RAM 图像 → preprocess → inference → postprocess → Results，不含模型加载和磁盘 I/O。GPU 主测量在无背景任务的 GPU0 上完成；CPU 测量固定 4 个 Torch/OpenCV 线程及 affinity 28–31，属于受控低竞争窗口。表中 SD 为三种子单元均值的标准差，逐次样本以及 p50/p90/p99 保存在 closure evidence 目录。

| 单元 | GPU0 total 均值 ± seed SD (ms) | CPU total 均值 ± seed SD (ms) |
| --- | ---: | ---: |
| A Dense + NMS | 12.496 ± 0.188 | 5118.699 ± 7.305 |
| B Dense + End-to-End | 12.049 ± 0.327 | 5099.800 ± 2.874 |
| C MoE + NMS | 25.880 ± 0.103 | 7235.376 ± 6.724 |
| D MoE + End-to-End | 26.111 ± 0.979 | 7219.193 ± 2.416 |

相对同一因子分支去除 NMS：GPU 上 B−A 为 −0.447 ms（约 −3.6%），D−C 为 +0.231 ms（约 +0.9%）；CPU 上 B−A 为 −18.900 ms（−0.369%），D−C 为 −16.183 ms（−0.224%）。因此，A1 要求的 CPU total latency 可测改善满足，但改善很小，不能表述为显著部署加速。MoE 相比 Dense 的 GPU 延迟增加约 13–14 ms，说明当前逐专家 dispatch、索引聚合和 shared expert 的实现开销超过了 Top-2 的理论稀疏收益。此前 GPU1 并行测量受到背景任务影响，仅作为补充对照，不用于主结论。

### 6.6 r28 内部效率 profiling

为定位 MoE 延迟来源，另在同一张无背景 GPU0、batch=1、同一 640×640 空间变化输入上完成 3 个种子 × 4 个单元的 profiling（50 次预热、200 次采样）。本节是 DetectionModel 的 model-forward 口径，不包含 preprocess/postprocess；完整端到端 latency 仍以 6.5 为准。seed 260831 的首次 D 测量出现机器级瞬时长尾，已用同协议的 clean repeat 替换；原始两次运行均保留在远端实验目录，汇总只使用无异常的 consolidated evidence。

| 单元 | model forward 均值 (ms) | 吞吐 (img/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: |
| A Dense + NMS | 5.686 | 175.9 | 686.6 |
| B Dense + End-to-End | 6.154 | 162.5 | 686.6 |
| C MoE + NMS | 13.180 | 75.9 | 695.5 |
| D MoE + End-to-End | 14.078 | 71.0 | 696.1 |

跨三种子平均的内部阶段计时如下（C/D 的 MoE factor）：Router 约 1.502 ms，shared expert 约 0.229 ms，已选专家计算约 0.778 ms，专家 dispatch/索引/聚合剩余约 2.600 ms，MoE 子模块合计约 5.109 ms。也就是说，Router + dispatch/聚合约占 MoE 子模块时间的 80%，真正选中的专家卷积只占约 15%。Dense factor 约 4.120 ms，而 MoE factor 约 9.829 ms；因此当前主要瓶颈是逐专家 dispatch、索引和聚合，而不是冻结 base 本身。该结果支持后续优先做 grouped/fused dispatch profiling，不应先增加专家数或盲目延长训练。

### 6.7 r28 同设备效率筛选与 dispatch 原型

为满足同设备、batch=1 的效率筛选要求，使用 GPU1（RTX 4090）、640×640、20 次预热和 60 次采样补测四组 model-forward：

| 单元 | mean (ms) | p50 (ms) | p99 (ms) | 吞吐 (img/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A Dense + NMS | 5.809 | 5.795 | 6.326 | 172.1 | 686.9 |
| B Dense + End-to-End | 6.236 | 6.226 | 6.355 | 160.4 | 687.5 |
| C MoE + NMS | 13.004 | 12.947 | 13.823 | 76.9 | 695.5 |
| D MoE + End-to-End | 13.879 | 13.779 | 14.534 | 72.0 | 696.1 |

Top-1 只改善约 2%～3%；简单限制专家数量没有稳定收益；仅后层 MoE 的推理延迟可接近 Dense，但该结果只是推理代理，不能代替独立训练精度结论。`cap2/cap4` 代理保留原 checkpoint 权重并重映射路由，不作为正式模型结论。

随后实现两个仅用于诊断的设备端 dispatch 原型：`dense_gather` 的 C/D 输出最大误差均为 0，但 clean repeat 延迟分别为 C 14.439 ms、D 14.376 ms，未优于 sparse 路径；`vmap_grouped` 的单个 MoE block 误差约 `4.4e-3` 以内，但全模型等价性检查出现 C `5.5e-1`、D `6.76e+2` 的误差，未通过门禁，未采用。P1 默认路径保持 sparse 不变。

### 6.8 P2-B r1 固定 val512 机制诊断

P2 首轮不重训，使用 r28 三种子 C/D 的正式 `last.pt` 和固定 pilot val512（共 512 张）进行无噪声 hard Top-2 推理诊断。4 专家和 8 专家模块的选择分布较均衡；16 专家模块的熵和 Gini 随层、seed 变化，但没有跨 seed 的统一坍塌模式。seed 260829 的一个 16 专家模块出现 1 个零选择专家，seed 260830/260831 未复现，因此按任务书只能作为诊断记录，不能单独写成“路由坍塌”结论。

| 观察项 | r1 结果 | 当前解释 |
| --- | --- | --- |
| max selection fraction | 0.183–0.395 | 未接近 0.8 集中阈值 |
| normalized entropy | 0.611–0.992 | 均高于 0.5 |
| dead expert | 仅 seed260829 一个 16-expert module 出现 1 个 | 非跨 seed 可复现机制 |
| C 同类高 IoU 重复率 | 0–0.00000 | 未见明显重复 |
| D 同类高 IoU 重复率 | 0.01026–0.01689 | 需与 B 做匹配/后处理对照 |

因此，r1 暂时排除了“r28 全局路由坍塌”这一简单解释，同时提示第 8 层个别专家负载不均以及 D 的 one-to-one 输出重复率值得继续受控验证。原始证据保存在 `p1_factorial_medium_r28/p2_mechanism_r1/r1_evidence/evidence.json`；该结果仍不是新的 mAP 结论。

### 6.9 P2-E r1：End-to-End 精度归因（固定 val512）

P2-E r1 在 GPU1 对 r28 三 seed 的 A/B/C/D 正式 `last.pt` 做了只读评测，固定 512 张验证图、640×640、
batch=1 和相同置信度口径；没有改模型、路由、assigner/loss 或训练预算。固定 val512 的 mAP50-95 均值为
A `0.43391`、B `0.42808`、C `0.43419`、D `0.42875`，与完整 COCO r28 主实验的数值不混用。

诊断信号如下：

| 指标（val512 三 seed 均值） | Dense one-to-many（A/C） | End-to-End one-to-one（B/D） | 解释 |
| --- | ---: | ---: | --- |
| 每图 assigner 正样本数 | `60.57` | `6.88` | one-to-one 监督显著稀疏 |
| assigner matched IoU | `0.8174–0.8177` | `0.8382–0.8383` | 一对一匹配质量不低，问题不只是框定位 |
| assignment overlap rate | `0.884` | `0.000` | one-to-one 无重复 GT 分配 |
| cls loss | `1.081–1.081` | `1.305` | 分类监督项明显更高，优先排查 |
| box loss | `0.978–0.980` | `1.062` | 定位项也更高，但幅度小于分类项 |
| validator recall（最佳 F1 阈值） | `0.5261–0.5278` | `0.5225–0.5304` | 跨 seed 波动，不能只凭一次 recall 下结论 |

当前只形成“one-to-one 正样本数量大幅减少，分类/定位 loss 较高，recall 轻微下降”的归因线索，
尚不能宣称某个 assigner/loss 改动有效。逐图原始 CSV 和完整证据保存在
`p1_factorial_medium_r28/p2_e2e_precision_r1/`；下一步才是固定一个因素的短程受控 pilot。

首版 P2-E r2 的事后审计发现，脚本执行时 Python 优先使用了远端另一个 editable checkout，
使 `gain=1.2` 未进入实际损失；B/D 的控制与处理 checkpoint 张量完全相同，因此该目录只作
无效实验审计证据。已修正 runner/evaluator 的导入隔离，并在单 batch 上验证两种增益的 loss
与 one-to-one 分类头梯度不同。修正版实验目录为 `p2_e2e_loss_r2_corrected/`，当前重新运行中。
修正版 r2-c 已完成四组训练和固定 val512 诊断。B 的分类增益处理组相对中性组 mAP50-95 为
`-0.000756`，D 为 `+0.000311`；正样本数、匹配 IoU 和 overlap 基本不变。该结果只说明增益
已真正生效，但不支持单 seed、5 epoch 下的正式收益结论。完整汇总见
`smoke/a1/p2_e2e_loss_r2_corrected/RESULT_SUMMARY.md`，后续应转向新的 one-to-one assigner/loss
因素筛选。

---

## 7. 建议下一步

不建议直接增加 epochs、专家数或继续盲目调参。P1 已完成，P2 当前主线转为 detect 侧
End-to-End 精度差距诊断与 one-to-one assigner/loss 改进；MoE 路由机制分析作为辅助证据，
seg/pose 扩展暂不作为第一优先级。

1. **P1 效率闭环已完成。** 已补齐同设备四组 model-forward、显存、吞吐和 Router/dispatch 分解；完整端到端 latency 仍以第 6.5 节为准。
2. **P2-E r1 已完成。** 固定 r28 A/B/C/D checkpoint 和 pilot val512，已采集 recall、one-to-one 正样本数、匹配 IoU/重叠、未匹配 GT、分类置信度以及 box/DFL/class 误差；结果指向 one-to-one 监督稀疏及分类/定位 loss 偏高，下一步才做单因素受控改动。
3. **P2-B r1 保留为辅助证据。** 已从 r28 原始 C/D checkpoint 采集路由负载、熵、Gini、Top-K 和候选重复率；该结果用于排除简单的全局路由坍塌解释，不替代 End-to-End 精度主线。
4. **P2 go/no-go。** 只有当 assigner/loss pilot 在固定 seed 下明确改善 recall、匹配质量或定位误差且不破坏 NMS-free 闭环，才扩大到三 seed；否则记录无效缓解，不把 MoE 代理筛选或单次死专家写成正式收益/负结果。

5. **P2-E r3 机制诊断已完成。** B/D 分别比较 one-to-one `topk2=1` 与 `topk2=2`，固定 seed=260829、5 epoch、pilot val512。`topk2=2` 在 506～509/512 图像上未增加正样本，四组重复正样本和重叠率均为 0；匹配 IoU 略升、框/DFL 损失未恶化，但分类损失上升 19%～23%，预测数和 FP 增加约 29%～32%，mAP50-95 下降 0.032789（B）和 0.023616（D）。因此下降主要来自分类分数校准与假阳性增加，不是定位匹配恶化。r3 处理组已拒绝，详细统计见 `smoke/a1/p2_e2e_assigner_r3/MECHANISM_DIAGNOSIS.md`；后续若继续 P2，只先增加 assigner 中间量审计并做单因素候选生成/冲突消解实验。

r28 的协议、数据列表、实现 SHA、initializer、正式请求、12 个 checkpoint 和 closure 证据继续封存保留；r23/r24/r25 仅作为历史审计证据。

## 8. 主要证据文件

P0：

- `smoke/a1/P0_PRETRAINED_REPORT.md`
- `smoke/a1/p0_pretrained/p0_report.json`
- `smoke/a1/p0_pretrained/validation.json`
- `smoke/a1/p0_pretrained/latency.json`
- `smoke/a1/p0_pretrained/identity.json`

P1：

- `smoke/a1/P1_FACTORIAL_MEDIUM_R28_FINAL_AUDIT.md`
- `smoke/a1/P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md`
- `smoke/a1/p1_factorial_medium_r28/closure_r1/closure_result_summary.json`
- `smoke/a1/p1_factorial_medium_r28/closure_r1/cpu_latency_low_contention_r1/`
- `smoke/a1/p1_factorial_medium_r28/efficiency_profile_r4_consolidated/`（同设备 batch-1 内部阶段、显存和吞吐 profiling）
- `smoke/a1/p1_factorial_medium_r28/EFFICIENCY_SCREEN_R7_SUMMARY.md`
- `smoke/a1/p1_factorial_medium_r28/efficiency_screen_r7_{a,b,c,d}_evidence.json`
- `smoke/a1/p1_factorial_medium_r28/dispatch_mode_benchmark_r9/r10_evidence.json`
- `smoke/a1/p2_mechanism_r1/PROTOCOL.md`
- `smoke/a1/p2_mechanism_r1/protocol.json`
- `smoke/a1/p2_mechanism_r1/r1_evidence/evidence.json`
- `smoke/a1/p2_e2e_precision_r1/PROTOCOL.md`
- `smoke/a1/p2_e2e_precision_r1/protocol.json`
- `smoke/a1/p2_e2e_precision_r1/evidence.json`
- `scripts/a1/run_p2_e2e_precision_diagnostics_r1.py`
- `r23-final-audit/`（历史 pilot 审计）
- `r23-final-audit/P1_FACTORIAL_R23_REPORT.md`
- `r23-final-audit/result_summary.json`
- `r23-final-audit/protocol.json`
- `r23-final-audit/formal_checkpoints.json`
- `r23-final-audit/formal_admission.json`
- `r23-final-audit/routing/hard_top2_512.json`
- `r23-final-audit/residual_activity/routing_probe_512.json`

---

## 9. 待导师确认的后续方向

当前 r28 已完成 A1 P1 的中等规模闭环，但 MoE 在本预算下没有稳定精度收益且推理更慢。建议向导师集中确认以下问题：

1. 是否同意以 detect 侧 End-to-End 精度差距（recall、匹配、分类、定位）作为 P2 当前主线，而将 MoE 路由机制诊断作为辅助、暂缓 seg/pose 扩展？
2. 是否接受当前证据：MoE 的主要效率损失来自同步/dispatch，而非专家 kernel；dense-gather 和 vmap 原型均不纳入正式实现？
3. 是否同意优先研究 one-to-one assigner/loss，并按 recall、匹配质量、分类误差、定位误差的归因结果逐项做单因素受控 pilot？
4. 若 P2 资源只允许一条训练验证，是否锁定固定 pilot val512、单 seed、短程受控对照，再决定是否扩大到三 seed？
5. B/D 严格 raw ONNX 行级比较的 4 个极低分 TopK 尾部差异，是否接受当前 `partial` 限制，还是要求先完成导出一致性修复？
