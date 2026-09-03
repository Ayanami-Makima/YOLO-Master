# A1｜阶段性报告：预训练基线复现与 MoE × End-to-End 中等规模消融实验

> 课题：A1｜MoE × End-to-End：现有 one-to-one 能否真正免 NMS
> 阶段：P0 / P1（截至 2026-09-02）｜作者：Ayanami-Makima｜状态：阶段报告

---

## TL;DR

1. **P0 已完成。** 使用同一个官方 `yolo26n.pt`，在完整 COCO val2017 上复现了标准 NMS 与 End-to-End NMS-free 两条原生推理路径，并通过真实 suppression kernel 调用审计确认：NMS 路径发生 16 次抑制调用，End-to-End 路径为 0 次。
2. **P1 中等规模正式实验已完成。** 采用 20,000 张 COCO 训练图像、5,000 张验证图像、15 epochs、3 个配对随机种子，完成 A/B/C/D 共 12 个正式单元，并通过恢复感知补充审计 12/12。
3. **P1 核心结果：** Dense 与 MoE 的 mAP50-95 总体相同，MoE 主效应为 `-0.00001`；End-to-End 相对 NMS 的主效应为 `-0.00734`，且三个种子方向一致。
4. **预训练功能得到保持。** 层 4/6/8 使用 `C3k2ResidualFactor`，以冻结的官方 C3k2 为 base，并通过零初始化 gain 保证四格 initializer 保存前和重载后均与 P0 严格等价；每个单元冻结 base 参数为 459,232，正式训练后最大误差为 0。
5. **阶段结论：** 在当前预训练、冻结基座和中等训练预算下，A2C2fMoE factor 尚未带来稳定精度收益；当前主要差异来自检测范式，one-to-one End-to-End 的精度仍低于非 End-to-End + NMS。该结论不外推到从头训练、完整 COCO 长周期训练或其他任务。
6. **交付闭环：** 12 个 ONNX 图均不含 `NonMaxSuppression`，B/D 真实运行时抑制调用为 0；低竞争 CPU 延迟下 B−A 为 −0.37%、D−C 为 −0.22%，可测但很小。严格 raw all-row ONNX 比较仍有 B/D 的 4 个极低分 TopK 尾行差异（A/C 24/24、B/D 20/24），该限制未被阈值语义审计掩盖。

---

## 1. 过去工作

### 1.1 实验工作

| 序号 | 工作 | 状态 |
| ---: | --- | :---: |
| 1 | P0：锁定官方代码 SHA、checkpoint 身份和完整 COCO val2017 评测口径 | ✅ |
| 2 | P0：复现 one-to-many + NMS 与 one-to-one NMS-free 两条原生路径 | ✅ |
| 3 | P0：完成真实 NMS 调用审计及 GPU/CPU Batch-1 延迟测量 | ✅ |
| 4 | P1：建立 Dense / MoE × NMS / End-to-End 的 2×2 等预算设计 | ✅ |
| 5 | 定位 r9 无效负结果：直接替换 C3k2 导致约 21%～23% 参数随机初始化 | ✅ |
| 6 | 引入 YAML 可重建的 `C3k2ResidualFactor`，严格保持并冻结官方预训练 C3k2 base | ✅ |
| 7 | 统一训练与推理路由制度：MoE 从首个训练 batch 起使用硬 Top-2 | ✅ |
| 8 | 完成 initializer、preflight、路由、残差活性、冻结状态和正式准入门禁 | ✅ |
| 9 | 完成 5k/512、5 epochs、3 seeds 的 pilot 实验，用于协议和训练链路验证 | ✅ |
| 10 | 完成 20k/5k、15 epochs、3 seeds 的 r28 中等规模正式实验，共 12 个单元 | ✅ |
| 11 | 针对服务器断连和断点恢复，完成 checkpoint、日志与恢复元数据的补充审计 | ✅ |

### 1.2 团队分工

本课题在与 @delei-kong 同学进行分工合作，上周我们进行了相关讨论和会议，做了简单的合作安排：

1. 前期，我们讨论不同的方案实现，各自尝试不同方案下的一个效果和情况，借鉴相关的结论和方法；
2. 后期，根据工作量分工合作对于机制分析研究和下游任务迁移的工作。

## 2. 实验设计与口径

### 2.1 P0：官方预训练推理基线

P0 不训练模型，而是使用同一个官方 `yolo26n.pt` 比较两条原生检测路径：

| 单元 | Head 路径 | 后处理 | 回答的问题 |
| --- | --- | --- | --- |
| A | one-to-many | 标准 NMS | 官方预训练模型的传统检测基线 |
| B | one-to-one | NMS-free | 现有 End-to-End 路径能否正确免 NMS，以及精度和延迟代价 |

P0 锁定信息：

- 官方基线代码 SHA：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- checkpoint：`yolo26n.pt`
- checkpoint SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- 参数量：2,572,280
- 数据集：完整 COCO val2017
- `imgsz=640`、`batch=16`、`conf=0.001`、`iou=0.7`、`max_det=300`、`half=False`

### 2.2 P1：2×2 因子实验

P1 将问题拆分为两个独立变量：

| 变量 | 取值 | 回答的问题 |
| --- | --- | --- |
| 因子分支 | Dense A2C2f / A2C2fMoE | MoE 相对 Dense 是否带来收益 |
| 检测范式 | 非 End-to-End + NMS / End-to-End one-to-one | 免 NMS 的精度代价，以及它与 MoE 的交互 |

四格定义如下：

| 格 | 因子分支 | 检测范式 | 主要路径 |
| --- | --- | --- | --- |
| A | Dense A2C2f | 非 End-to-End | one-to-many + NMS |
| B | Dense A2C2f | End-to-End | one-to-one，NMS-free |
| C | A2C2fMoE | 非 End-to-End | one-to-many + NMS |
| D | A2C2fMoE | End-to-End | one-to-one，NMS-free |

主要比较量：

- NMS 条件下的 MoE 效应：`C - A`
- End-to-End 条件下的 MoE 效应：`D - B`
- Dense 条件下的 End-to-End 效应：`B - A`
- MoE 条件下的 End-to-End 效应：`D - C`
- 交互效应：`(D - C) - (B - A)`

### 2.3 P1 正式训练配置

| 项目 | 锁定值 | 项目 | 锁定值 |
| --- | --- | --- | --- |
| 训练图像 | 20,000 | 验证图像 | 5,000 |
| epochs | 15 | imgsz | 640 |
| batch | 4 | seeds | 260829 / 260830 / 260831 |
| optimizer | SGD | lr0 | 0.0001 |
| AMP | False | deterministic | True |
| workers | 0 | save_period | 1 |
| mosaic | 0.0 | mixup / copy_paste | 0.0 / 0.0 |
| 训练层 | 4 / 6 / 8 / 23 | 其他模型层 | 全部冻结 |
| BatchNorm | 参数和统计量全部冻结 | C3k2 base | 全部冻结 |

Residual gain 使用独立优化设置：`lr=0.01`、`weight_decay=0.0`、无 warmup。四格除模型因子与 `end2end` 设计变量外，数据列表、预算、优化器、增强、随机种子和冻结策略保持一致。

## 3. 实验操作与配置

### 3.1 预训练保持方案

官方 `yolo26n.pt` 的第 4、6、8 层为 C3k2，检测头为第 23 层。早期 r9 直接将 C3k2 重建成 A2C2f/A2C2fMoE，造成大比例参数随机初始化，得到的低 mAP 不能代表有效的 Dense/MoE 比较。

r12 起采用以下残差因子结构：

```text
C3k2ResidualFactor(x) = base(x) + gain × factor(base(x))
```

- `base`：官方预训练 C3k2，训练中始终冻结；
- `factor`：A/B 使用 Dense A2C2f，C/D 使用 A2C2fMoE；
- `gain`：初始化为精确的 0；
- 初始化门禁：保存前最大误差 `0.0`，重载后最大误差 `0.0`；
- 每单元冻结的官方 C3k2 base 参数：`459232`。

该设计使四格在训练开始前都严格继承 P0 功能，同时允许新因子分支通过 gain 逐步参与优化。

### 3.2 MoE 路由设置与诊断

第 4、6、8 层的专家数分别为 4、8、16，均使用硬 Top-2：

| 路由项 | 值 |
| --- | --- |
| progressive_sparsity | False |
| top_k | 2 |
| warmup_steps | 0 |
| temperature | 1.0 |
| expert_dropout_rate | 0.0 |
| 推理/验证噪声 | 0 |
| 路由制度 | hard Top-2 from step zero |

早期 routing probe 暴露出训练阶段渐进稀疏仍接近 Top-12、审计和推理却直接切到 Top-2 的制度不一致。后续将训练、验证和推理的稀疏语义统一，并使用只作用于训练期 dispatch 的短程私有探索；balance loss 与 z loss 始终使用 clean logits，验证和推理保持 clean hard Top-2。

固定样本中个别专家零命中被保留为诊断项，而真正的路由崩塌由最大选择比例和归一化熵判断。正式准入所用硬 Top-2 路由审计和残差活性审计均已通过。

### 3.3 正式完整性与恢复审计

r28 共 12 个正式单元，均满足：

- `results.csv` 含完整 15 个正式 epoch，指标为有限值；
- A/C 的模型及检测头 `end2end=False`，B/D 为 `True`；
- 三处官方 C3k2 base 逐张量不变，最大误差为 0；
- 其他冻结层与 BatchNorm 参数、统计量全部不变；
- 三处 residual gain 均非零；
- A/B 不含路由器；
- C/D 三处 factor 的路由参数与专家参数均发生更新；
- 请求、initializer 和最终 checkpoint 的 SHA256 与首次审计记录一致。

服务器断连后训练从 checkpoint 恢复。原审计器只读取固定位置的单段日志，并将恢复段的局部 microbatch 计数当作全程计数，因此给出元数据不兼容失败。新增的恢复感知补充审计只读绑定原审计、请求、initializer、最终 checkpoint 和恢复记录，最终通过 12/12；实验权重、锁定协议和仓库语义未被修改。

## 4. 实验结果

### 4.1 P0：完整 COCO val2017

| 单元 | 推理路径 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | one-to-many + NMS | 0.659 | 0.515 | 0.562 | **0.402** |
| B | one-to-one，NMS-free | 0.656 | 0.505 | 0.550 | **0.395** |

End-to-End 相对 NMS 的 mAP50-95 变化为 `-0.007`。

运行时路径审计：

| 单元 | 检测数 | suppression kernel 调用 | 实际路径 |
| --- | ---: | ---: | --- |
| A | 77 | 16 | `nms` |
| B | 76 | 0 | `end2end_score_filter` |

Batch-1 端到端延迟（不含磁盘 I/O 和模型加载）：

| 设备 | A 平均延迟 | B 平均延迟 | B 相对 A |
| --- | ---: | ---: | ---: |
| GPU 0 | 3.289 ms | 3.284 ms | -0.14% |
| CPU | 20.246 ms | 20.757 ms | +2.52% |

### 4.2 P1 r28：20k/5k、15 epochs

以下为各单元第 15 轮 `metrics/mAP50-95(B)`：

| Seed | A Dense+NMS | B Dense+E2E | C MoE+NMS | D MoE+E2E |
| ---: | ---: | ---: | ---: | ---: |
| 260829 | 0.40192 | 0.39512 | 0.40247 | 0.39452 |
| 260830 | 0.40177 | 0.39481 | 0.40182 | 0.39492 |
| 260831 | 0.40232 | 0.39472 | 0.40236 | 0.39451 |
| **Mean** | **0.40200** | **0.39488** | **0.40222** | **0.39465** |
| 样本标准差 | 0.00028 | 0.00021 | 0.00035 | 0.00023 |

### 4.3 因子效应

| 效应 | mAP50-95 差值 |
| --- | ---: |
| NMS 下 MoE 效应 `C - A` | +0.00021 |
| End-to-End 下 MoE 效应 `D - B` | -0.00023 |
| Dense 的 End-to-End 效应 `B - A` | -0.00712 |
| MoE 的 End-to-End 效应 `D - C` | -0.00757 |
| MoE 主效应 | -0.00001 |
| End-to-End 主效应 | -0.00734 |
| MoE × End-to-End 交互 | -0.00045 |

## 5. 分析

### 分析点 1：P0 与 P1 对 End-to-End 代价给出一致方向

P0 使用官方 checkpoint 和完整 COCO val2017，End-to-End 相对 NMS 低 `0.007` mAP50-95；P1 在经过训练的 Dense/MoE 两种因子下，End-to-End 主效应为 `-0.00734`。两阶段实验口径不同，绝对指标不能直接合并，但差异方向和量级接近，说明当前 one-to-one 路径的精度代价具有稳定性。

### 分析点 2：MoE 在当前预算下没有形成稳定收益

在 NMS 条件下，MoE 相对 Dense 为 `+0.00021`；在 End-to-End 条件下为 `-0.00023`，三种子汇总后的 MoE 主效应接近 0。当前证据不支持“A2C2fMoE factor 明显优于 Dense A2C2f factor”的结论，也没有显示明显的负面主效应。

### 分析点 3：结果不是初始化失败或整体路由崩塌

四格 initializer 与 P0 的保存前、重载后误差均为 0；冻结 C3k2 base、其他冻结层和 BatchNorm 在训练后保持完全不变；所有 gain 均非零，C/D 的 router 和 expert 参数均获得更新。正式准入的路由集中度、归一化熵和残差活性审计已通过，因此当前“MoE 无显著增益”是有效实验结果，不是 r9 式随机初始化错误或整体路由崩塌。

### pipeline 成熟度评估

P0 已具备 checkpoint 身份、精度、真实 NMS 调用与延迟证据；P1 已形成 initializer 等价性、preflight、冻结状态、路由、残差活性、正式准入、断点恢复、运行时路径、ONNX、资源与 CPU/GPU 延迟的完整证据链。r28 的 12 个正式单元均可由锁定数据列表、种子、协议和 checkpoint 追溯；严格 raw ONNX 尾部差异作为交付限制保留。

## 6. 下一步计划

### 近期

1. 封存 r28 的 initializer、checkpoint、协议和闭环证据，不覆盖或续训已有正式单元。
2. 若进一步研究 ONNX 精确数值等价性，应单独定位 B/D 4 个极低分 TopK 尾行的浮点排名差异；不能把 `conf=0.001` 的语义通过写成严格逐行 100% 一致。
3. 对 one-to-one 路径的召回率、分类误差和匹配质量进行分解，定位约 `0.0073` mAP50-95 差距的主要来源。
4. 固化 r28 为 A1 P1 的中等规模可复现实验，不覆盖现有 initializer、checkpoint、协议和审计证据。

### 中期

5. 若算力周期允许，另建全新实验目录进行完整 COCO train2017 长周期验证；不从 r28 续训，不改变 P1 已锁定结论。
6. 在保持等预算原则的前提下，研究路由负载、专家专业化和稀疏计算是否能转化为可测的速度或任务收益。
7. 根据机制分析结果，再决定是否进入检测结构改进或 seg / pose 等下游任务迁移。

---

## 附录

### A. 关键证据文件

| 内容 | 路径 |
| --- | --- |
| P0 阶段报告 | `smoke/a1/P0_PRETRAINED_REPORT.md` |
| P0 汇总数据 | `smoke/a1/p0_pretrained/p0_report.json` |
| P1 r28 最终审计 | `smoke/a1/P1_FACTORIAL_MEDIUM_R28_FINAL_AUDIT.md` |
| P1 r28 运行时/导出/资源闭环 | `smoke/a1/P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md` |
| P1 r28 closure 汇总与 CPU 样本 | `smoke/a1/p1_factorial_medium_r28/closure_r1/` |
| P1 r28 结果汇总 | `smoke/a1/p1_factorial_medium_r28/final_result_summary.json` |
| 原正式 checkpoint 审计 | `audits/formal_checkpoints.json` |
| 恢复感知补充审计 | `audits/formal_checkpoints_recovery_aware_v2.json` |
| 正式准入 | `audits/formal_admission.json` |
| 硬 Top-2 路由审计 | `audits/routing/hard_top2_5000.json` |
| 残差活性审计 | `audits/residual_activity/routing_probe_512.json` |

### B. 结论边界

- P0 是同一官方预训练 checkpoint 的两条推理路径复现，不是 MoE 消融。
- P1 r28 是冻结官方 C3k2 base 的中等规模微调实验，不等同于从头训练。
- P0 与 P1 的数据、训练状态和评价目的不同，不能将绝对 mAP 直接作训练收益比较。
- 当前结论只适用于 20k/5k、15 epochs、3 seeds 和已锁定超参数；完整 COCO 长周期训练需作为独立实验重新立项和审计。

### C. 当前可复现结论

1. 官方 `yolo26n.pt` 的标准 NMS 与 End-to-End NMS-free 路径均已被真实运行时证据确认。
2. 在 P1 中等预算下，MoE 主效应接近 0，未观察到稳定精度收益。
3. End-to-End 相对 NMS 稳定低约 `0.00734` mAP50-95。
4. 该结果建立在预训练保持、冻结一致、路由有效和 12/12 恢复感知审计通过的基础上。
