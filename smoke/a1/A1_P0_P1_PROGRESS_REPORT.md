# A1 P0 / P1 实验进展报告

更新日期：2026-08-30

## 1. 当前总体进展

A1 当前围绕 YOLO26n 的两项核心问题展开：

1. P0：复现官方预训练模型的标准 NMS 和 End-to-End NMS-free 两条推理路径。
2. P1：在保持官方预训练基座功能的前提下，完成 Dense / MoE 与 NMS / End-to-End 的 2×2 等预算实验。

目前状态如下：

- **P0 已完成并通过。** 官方 `yolo26n.pt` 的两条原生推理路径均已在完整 COCO val2017 上复现，并完成真实 NMS 调用和延迟审计。
- **P1 pilot 已完成并通过。** r23 已完成 3 个配对随机种子、4 个实验单元，共 12 次正式训练；所有正式 checkpoint、冻结状态、参数更新、路由状态和残差活性审计均通过。中等规模扩展实验尚未完成。
- P1 的当前结论是：在锁定的 pilot 预算下，**MoE 没有表现出相对 Dense 的精度优势；End-to-End 的精度稳定低于 NMS**。
- full-COCO r24/r25 已完成协议、初始化、preflight、routing probe 和正式准入验证；实测 118,287 张训练图、5,000 张验证图、30 epochs 的完整 12 单元实验预计需要约 16 天，因此在正式训练初期主动停止，不作为结果。

> 注意：P0 使用完整 COCO val2017，P1 使用固定的 5,000 张训练图像和 512 张验证图像。两者数据尺度不同，指标不能直接横向比较。

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

### 3.3 P1 数据与正式预算

| 项目 | 锁定值 |
| --- | --- |
| 训练图像 | 5,000 |
| 验证图像 | 512 |
| Epochs | 5 |
| imgsz | 640 |
| 物理 batch | 4 |
| nbs | 16 |
| accumulate | 4 |
| 等效 batch | 16 |
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

### 3.4 P1 正式训练超参数

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

### 3.6 MoE 结构与路由超参数

第 4、6、8 层的 MoE 专家数分别为：

- 第 4 层：4 个专家
- 第 6 层：8 个专家
- 第 8 层：16 个专家
- 所有 MoE 层均使用硬 Top-2

r23 锁定的路由设置如下：

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

### 6.2 P1 pilot 结论

P1 r23 已满足以下要求：

- 2×2 等预算设计
- 官方预训练 C3k2 功能保持
- 冻结一致性
- 3 个配对随机种子
- 12 个独立正式 checkpoint
- Dense / MoE 因子均正常更新
- MoE router / expert 均正常更新
- 无路由集中或路由崩塌
- 正式结果可追溯和可复现

在固定 5,000 张训练图像、512 张验证图像、5 epochs 和 3 个种子的 pilot 预算下：

1. **MoE 与 Dense 的精度总体相当。**
2. **MoE 没有表现出统计上明确的 mAP50-95 优势。**
3. **End-to-End 的精度比 NMS 稳定低约 0.0104 mAP50-95。**
4. **MoE 与 End-to-End 的交互效应很小。**
5. 当前结果是有效的 pilot 负结果，不是模型失败或路由崩塌。

### 6.3 结论边界

P1 目前只是 pilot-budget 结论，不能外推为完整 COCO 结论，主要限制为：

- 训练数据只有 5,000 张图像。
- 验证数据只有固定 512 张图像。
- 正式训练只有 5 epochs。
- 统计分析只有 3 个配对种子。

若要判断 MoE 在完整训练规模下的最终收益，需要重新建立并锁定 full-COCO 实验协议。不得直接把当前 pilot checkpoint 续训为 full-COCO，也不得把 P1 pilot 指标与 P0 完整 COCO 指标直接比较。

### 6.4 full-COCO 扩展尝试与当前状态

r24/r25 使用完整 COCO train2017 的 118,287 张训练图像、完整 val2017 的 5,000 张验证图像和 30 epochs。所有正式训练前门禁均已通过：

- 12/12 preflight checkpoint 通过。
- 6/6 routing probe checkpoint 通过。
- 36/36 硬 Top-2 路由器通过熵和集中度门禁。
- 18/18 残差层通过活性门禁。
- formal admission 通过。

正式训练实测约 5 batch/s，每个单元约需 50 小时；按三种子、A/B/C/D 和双 GPU 顺序执行，关键路径预计约 16 天。考虑当前资源周期，r25 已在正式 A 单元第 1 个 epoch 初期停止，未续跑，也未将部分 checkpoint 作为实验结果。

下一步计划建立全新 r26 中等规模实验：固定抽取 20,000 张 COCO train2017 图像，使用完整 5,000 张 COCO val2017，训练 15 epochs。模型、A/B/C/D 定义、3 个随机种子、优化器、冻结策略、BatchNorm 策略和路由门禁均保持不变；所有 initializer 重新从官方 `yolo26n.pt` 独立构建，不复用 r25 权重。

---

## 7. 建议下一步

1. 冻结并保留 r23 的协议、数据列表、实现 SHA、initializer、正式请求和 12 个 checkpoint。
2. 将 r23 作为 A1 P1 的可复现 pilot 基线，不覆盖现有结果。
3. 建立独立 r26：20,000 张固定训练子集、完整 5,000 张验证集、15 epochs、3 seeds、A/B/C/D。
4. 后续可补充参数量、GPU 显存、训练吞吐量和推理延迟，形成精度—资源联合分析。
5. r26 只调整数据规模和 epochs，不修改 MoE 路由、辅助损失或推理语义，也不续接 r25。

---

## 8. 主要证据文件

P0：

- `smoke/a1/P0_PRETRAINED_REPORT.md`
- `smoke/a1/p0_pretrained/p0_report.json`
- `smoke/a1/p0_pretrained/validation.json`
- `smoke/a1/p0_pretrained/latency.json`
- `smoke/a1/p0_pretrained/identity.json`

P1：

- `r23-final-audit/P1_FACTORIAL_R23_REPORT.md`
- `r23-final-audit/result_summary.json`
- `r23-final-audit/protocol.json`
- `r23-final-audit/formal_checkpoints.json`
- `r23-final-audit/formal_admission.json`
- `r23-final-audit/routing/hard_top2_512.json`
- `r23-final-audit/residual_activity/routing_probe_512.json`
