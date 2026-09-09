# A1 P2 阶段交付报告与证据边界

> **P2 收尾状态（2026-09-09）**：证据型负结果路线已完成。已完成效率、MoE筛选、梯度通路、匹配/网格、单侧目标、完整PR/高IoU、val5000配对复评、500次bootstrap及seg pilot；pose不是任务书必选项。新增的正确生效 `tal_topk=7→10` 配对训练与固定 val512 审计也已完成，未发现稳定收益。当前不再启动新训练，剩余仅为最终审阅、证据归档与是否开展后续单变量干预的研究决策。

> 2026-09-08进行中：已启动现有 D alpha=0/0.1 权重在固定 COCO val5000 上的方形/矩形配对复评（GPU1、batch=1、FP32、conf=0.001、max_det=300），输出 `p2_gradient_bridge_pilot_r3/pr_native_capture_val5000_r1`。该过程只读，不覆盖 512 图证据；完成后再计算配对 bootstrap 区间。

> 2026-09-08最新：完整PR/高IoU/逐类分数排序复核已完成，四组标准mAP复现误差均0。矩形相对方形AP90下降1.6165/1.1987 pp（D alpha=0/0.1）；D1低/中IoU的分数区分指标并未全面下降，不能直接归因为分类分数问题。详见[PR与排序报告](p2_gradient_bridge_pilot_r3/PR_RANKING_REPORT.md)。下一步建议固定checkpoint在val5000上配对复评并估计差值不确定性，暂不新增长训。

> 2026-09-08完成：固定 checkpoint 的 val5000 配对复评与500次 bootstrap 已完成。mAP 矩形−方形中位数为 D0 -0.1389 pp（95% CI [-0.3835,+0.0803]）、D1 -0.1616 pp（[-0.3889,+0.0434]）；AP90 区间均完全为负。P2 机制证据已基本齐备，剩余为归档、最终复核和是否需要单变量干预的研究决策。

> 2026-09-08单侧检出诊断完成：D alpha=0.1的118个仅方形检出目标中，114个在矩形仍有同类别IoU≥0.5框但分数<0.25；其中5例同时有高分错误类别重叠。降低conf增加召回也显著增加FP。该结果解释固定阈值下的现象，尚不能解释完整mAP差距，详见 [单侧目标诊断](p2_gradient_bridge_pilot_r3/SINGLE_SIDE_ERROR_REPORT.md)。下一步为完整PR和高IoU/类别分数排序分析。

> 2026-09-08更新：下文第6节第1—2项已完成本轮修复和复评，详见 [同GT/实际网格复核报告](p2_gradient_bridge_pilot_r3/SAME_GT_GRID_AUDIT_REPORT.md)。512图、3536GT正确配对；实际anchor/stride/decode误差均0，逆变换最大误差0.00004503像素。D alpha=0/0.1在矩形下净少检出19/31个GT；共同GT的IoU差约-0.000714/-0.001123。旧产物保持其无效/有偏状态，新证据位于same_gt_grid_r2。根因尚未确定，但本报告已完成 P2 证据型结论与归档说明。

整理日期：2026-09-09。作者：Ayanami-Makima（@delei-kong）。
状态：证据型负结果路线已收尾，可提交导师/评审审阅；不宣称已证明全部精度差距的唯一根因。
公共基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`。

## 1. 任务标准与当前完成情况

任务书 A1 P2 原文为“扩展到 seg/pose，或给出有证据的负结果：梯度稀疏、匹配冲突、路由坍塌等机制分析”。
两条路线可选，不要求同时完成 seg 和 pose，也不要求每个候选涨点。增量补充细则要求锁定后可追溯的机制、实验或工具贡献；重复旧表格不单独构成 P2。

| 工作项 | 当前状态 | 可交付内容及限制 |
| --- | --- | --- |
| 四组同设备 batch=1 效率及耗时分解 | 已完成测量 | 延迟、吞吐、显存、内部耗时；forward 与完整端到端分开报告 |
| 后层 MoE 小规模筛选 | 已完成一轮 pilot | 2/4 experts × Top-1/2；无配置通过效率筛选；Dense 仅按活动 MLP 计算量匹配 |
| End-to-End 精度与监督机制研究 | 已完成诊断 | 已有梯度测量、训练干预、完整 PR/AP 和 val5000 bootstrap；未获得稳定提升，结论限定在锁定预算与评测协议 |
| seg 扩展 | 已完成单 seed pilot | A/B/D train/val/ONNX export；随机 mask/proto 初始化与短预算限制精度解释 |
| pose 扩展 | 未开展 | 原文不要求 seg、pose 两项全做 |
| 正确生效的候选预算复评 | 已完成 | B/D 各自 `topk=7` 对照 `topk=10` 处理，固定 seed、5 epoch、train5000/val512；运行时确认 `topk/topk2=10/1` |
| 最终机制负结果包 | 已收尾 | 已包含梯度通路、匹配/网格、路由效率、单侧目标、完整 PR/高 IoU、val5000 bootstrap 和 seg pilot；待导师/评审审阅，不能宣称已证明全部精度差距的唯一根因 |

## 2. 效率结果与筛选决策

r28 同一 GPU0、batch=1、640 输入、50次预热、200次采样、三个种子的内部 profiling：

| 组 | model-forward 均值 ms | 吞吐 img/s | 峰值显存 MiB |
| --- | ---: | ---: | ---: |
| A Dense + NMS | 5.686 | 175.9 | 686.6 |
| B Dense + End-to-End | 6.154 | 162.5 | 686.6 |
| C MoE + NMS | 13.180 | 75.9 | 695.5 |
| D MoE + End-to-End | 14.078 | 71.0 | 696.1 |

该表不包含完整预处理/后处理；完整端到端性能沿用 P1 closure 证据。内部测量的 Router 约1.502 ms，分发/索引/聚合剩余约2.600 ms，已选专家计算约0.778 ms，shared expert约0.229 ms。Router与分发/聚合约占 MoE 子模块计时的80%。这是当前实现的测量，不外推为所有 MoE 实现结论，也不单凭这些数字断言存在大规模 CPU/GPU 数据往返。

效率筛选只在层8加入 MoE，层4/6保留 Dense residual factor。六组使用 train5000/val512、1 epoch、batch4、640、SGD lr0=1e-4、seed260829、AMP关闭；GPU1 batch1 测量30次预热、100次采样。
Dense ratio4/6 延迟6.242–6.369 ms，MoE四组9.094–9.405 ms，慢约46%–48%；mAP50-95为0.426750–0.427500。
结论：此轮没有值得长训的效率候选。1 epoch 精度接近不等于已证明精度非劣。
Dense 对照匹配活动 MLP 计算量，不是严格全模型等参数或等 FLOPs 对照；若要发表严格公平对照结论，须另补设计。

证据：[效率筛选报告](p2_efficiency_screen_r1/RESULT_SUMMARY.md)、[性能原始汇总](p2_efficiency_screen_r1/benchmark/benchmark_evidence.json)。

## 3. End-to-End 主线：已有有效证据

分支梯度诊断覆盖 r28 的三个 seed。原生 one-to-one 检测损失到共享 factor/router 的梯度为零，one-to-many 检测损失可以传递梯度；这与已有 detach 设计一致，不是新发现的程序 bug，也不表示 router 的总训练梯度为零。
诊断中接入 `x.detach() + 0.1 * (x - x.detach())`，前向保持等价，共享反向通路生效。三个 seed 的32张相同图像上，D router 与 one-to-many 梯度 cosine 均值约0.548–0.706，负 cosine 图数1–4/32。局部反向不等于全局有害冲突，也不能据此解释 mAP 差距。

随后完成 B/D × alpha=0/0.1 四格梯度桥 pilot：train5000/val512、5 epochs、batch4、imgsz640、seed260829、SGD lr0=1e-4、lrf=0.2、momentum=0.9、weight_decay=0.0005、nbs=16、warmup_epochs=0.5、AMP=False、workers=0；训练层4/6/8/23，其余层、BN参数/统计和459232个官方base参数冻结。四格从各自原始 initializer 开始，不使用 preflight 权重。

| D 比较口径 | alpha=0 mAP50-95 | alpha=0.1 mAP50-95 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| 训练内末轮 | 42.3230% | 42.1720% | -0.1510 |
| 独立方形，batch1，未融合 | 42.4310% | 42.5695% | +0.1385 |
| 独立方形，batch1，融合 | 42.5180% | 42.5727% | +0.0547 |
| 独立矩形，batch1，未融合 | 42.3868% | 42.2024% | -0.1843 |

独立方形融合下 B 为42.5601%→42.5214%（-0.0387个百分点）。目前可交付的负结果是：在该单seed短预算下，开放0.1倍共享检测梯度未获得对评测设置稳定的提升。不能推导为“梯度桥普遍无效”或“detach导致全部精度下降”。

证据：[梯度研究及协议](P2_REQUIREMENTS_AND_RESEARCH_REVIEW_20260905.md)、[pilot协议](p2_gradient_bridge_pilot_r3/protocol.json)、[独立复评](p2_gradient_bridge_pilot_r3/reevaluation_r1/evidence.json)。

## 3.1 候选预算复评：`tal_topk=7→10`（r5 corrected）

为补齐历史 r5 因错误 checkout 导致的无效结论，重新从原始 B/D initializer 启动四个单元：
`B-control/D-control` 使用 `topk=7`，`B-candidate10/D-candidate10` 使用 `topk=10`；
其余数据、seed、优化器、冻结策略、5 epoch 预算和 `topk2=1` 全部锁定。运行时文件确认四组均加载当前
`YOLO-Master-r28-medium`，处理组实际为 `topk=10`。

固定 val512、batch=1 的独立评估（使用各组 `last.pt`）如下：

| 组别 | topk | mAP50-95 | candidate GT coverage | final GT coverage |
| --- | ---: | ---: | ---: | ---: |
| B-control | 7 | 0.425601 | 99.859% | 99.604% |
| B-candidate10 | 10 | 0.426193 | 99.859% | 99.604% |
| D-control | 7 | 0.425896 | 99.859% | 99.604% |
| D-candidate10 | 10 | 0.424645 | 99.887% | 99.576% |

`topk=10` 将候选正样本由约 45.6 增至 63.1 个/图，但最终正样本仍约 6.93 个/图；冲突锚点由约
1.28 增至 2.19–2.23 个/图。B 仅有约 0.00059 的单次上升，D 下降约 0.00125，平均处理效应约
`-0.00033`，不构成稳定收益。候选 GT 覆盖在 `topk=7` 时已近饱和，说明当前瓶颈更可能位于冲突消解、
质量排序或分类/定位校准，而不是候选数量不足。该结果支持封存 `topk=10` 候选预算方向，不启动更大
`topk` 或长训。

原始证据：`p2_e2e_candidate_r5_corrected_20260909/eval/evidence.json`；训练协议与请求位于
`configs/a1/p2_e2e_candidate_r5_corrected_20260909/`。

## 4. 诊断结果的保留、撤回与待修正

| 项目 | 处理 | 理由 |
| --- | --- | --- |
| r5 topk=10 训练结论 | 撤回 | 实际加载旧代码，干预未生效；配对权重相同 |
| 原生匹配数量与加权IoU | 保留描述性结果 | 方形/矩形匹配GT数几乎相同，未见总体正样本大量丢失；不能据此排除全部匹配问题 |
| masked Router | 保留本次干预无明显恢复 | D alpha=0.1矩形仅回升0.0106个百分点；不支持泛化的根因排除 |
| 最低分辨率通道统计归一化 | 否决这一具体干预 | D alpha=0/0.1分别下降2.1052/1.8803个百分点；不能证明“破坏空间语义”这一机制 |
| anchor grid逐图配对 | 无效，待修正 | 按顺序zip，只有1/512正确配对；总体平均比例可描述，不代表数量或实际解码正确性 |
| inverse-letterbox定位误差均值 | 有偏，待修正 | 无匹配图被填0；双方匹配对象也不固定，不能视为同GT定位差 |
| 最低分辨率尺度为主要根因 | 撤回 | 全局通道余弦不是空间对齐检验，且统计受padding面积影响 |
| 已证明训练/矩形分布不一致是根因 | 撤回 | 尚无受控训练证据；只能作为候选解释 |

对现有逐图文件只读复算，双方都有匹配的图像中，D alpha=0为485图、alpha=0.1为486图。其每图平均IoU差分别约-0.002621、-0.000385。该复算只揭示填0偏差，不作为已修复的同GT因果比较。
高于0.01置信度的已选候选中心均位于有效内容区，仅支持“该候选集合中未见区外中心”；它不覆盖所有解码前anchor，也不排除padding对特征的影响。

原始输出保留：[anchor审计](p2_gradient_bridge_pilot_r3/anchor_grid_r1/evidence.json)、[定位审计](p2_gradient_bridge_pilot_r3/inverse_letterbox_r1/evidence.json)、[统计归一化](p2_gradient_bridge_pilot_r3/scale_norm_r2/evidence.json)。本轮只整理证据，未声称修复或重跑这些脚本。

## 5. seg 扩展及边界

seg A/B/D pilot使用train5000/val512、5 epochs、batch4、640、seed260829、SGD lr0=1e-4、AMP关闭，冻结base与BN。
A/B/D的mask mAP50-95分别为5.2940%、4.3418%、2.8643%，三格均完成ONNX导出。该结果支持扩展链路可运行；由于mask/proto随机初始化及短预算，不将低mask精度解释成MoE跨任务失败。
证据：[seg评测](p2_seg_r1/evaluation_evidence.json)。

## 6. 收尾核对结果

1. anchor逐图配对已修复并复核：512/512 图像、3536/3536 GT；实际 anchor/stride/decode 误差为0。
2. 定位比较已改为按同图、同GT统计，漏检与共同匹配目标分开，不再用0填充缺失项，并完成坐标变换校验。
3. 梯度桥已用固定 val5000 做标准 PR/AP 复评及500次图像级 bootstrap；mAP 区间跨0，AP90区间均为负，保留为有限条件下的负结果。
4. 证据、复现命令、源码/权重/数据哈希和局限已汇总到本报告及链接报告。无需为完成 P2 强行补 pose、增加 epoch 或继续盲目调参。
5. r5 corrected 的四组训练、运行时参数审计和固定 val512 分配统计已归档；`topk=10` 未显示跨 B/D 的稳定收益，候选预算方向封存。
