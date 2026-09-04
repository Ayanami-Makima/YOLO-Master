# A1 P2 方案：MoE × End-to-End NMS-free 闭环与机制验证

## 1. 目标与边界

### 核心问题

在固定 YOLO-Master 公共基线、数据、训练预算和评测口径下，优先解释并缩小 one-to-one
End-to-End 相对 NMS 路径的精度差距，重点分析召回率、匹配质量、分类误差和定位误差；MoE
路由只作为兼容性和机制辅助变量，不预设其必须带来收益。

这不是“再造一个检测头”。任务是核验已有 one-to-many / one-to-one、训练、推理、导出和评测
路径是否构成真正的 NMS-free 闭环，并定位 MoE 路由与一对一监督是否发生冲突。

### 不可变验收边界

| 项目 | 规则 |
| --- | --- |
| 公共基线 | `acce839c7e895d6b179de7f7093fa879e237cc7b` |
| 主任务 | COCO detect，MoE on/off × NMS on/off 2×2 |
| 共同变量 | 数据 manifest、imgsz、batch、epoch、优化器、增强、设备、seed、评测脚本 |
| 主指标 | mAP50-95、GPU/CPU batch=1 端到端延迟及标准差 |
| P2 主目标 | detect 侧 End-to-End 精度诊断与 one-to-one assigner/loss 改进 |
| P2 备选目标 | seg/pose 扩展，或机制级、可复现的负结果 |
| 禁止项 | 仅写 `nms=False` 不能证明 NMS-free；必须证明 one-to-one end-to-end 输出未隐式回落到 NMS |

`YOLO-Master-v26.08` 可以说明版本来源；代码差异、成员贡献与最终验收均以以上 SHA 为起点。

## 2. 研究假设与成功标准

### H1：NMS-free 路径成立

`end2end=True` 时，one-to-one 分支可以完成 train、val、predict 和 export。验证、预测和导出
都使用 end-to-end 输出及其 postprocess；不出现重复框失控，也不调用常规 NMS。

### H2：MoE 与 one-to-one 兼容

MoE on + NMS off 的 mAP50-95 相对同一 MoE 模型的 NMS-on 对照，下降不超过 2 个点；三 seed
趋势一致，或能用匹配、路由和梯度证据解释差异。

### H3：去除 NMS 有真实收益

CPU batch=1 的完整 preprocess + inference + postprocess 总延迟有可重复改善。不能只报告模型
forward 时间，必须报告 mean、std、p50、p90、p99 和逐次样本。

### P2 判定

P1 通过后，优先完成 detect 侧 End-to-End 精度主线；资源或时间不足时，再选择以下备选主线：

1. 在固定预算下定位并改善 one-to-one 的召回率/匹配/分类/定位差距；
2. 将 detect 侧已验证的 MoE + NMS-free 方案扩展到 `seg` 或 `pose`，并完成最小对照；或
3. 形成机制级负结果：定位梯度稀疏、匹配冲突、路由坍塌、推理路由漂移或输出重复的具体条件，
   提供原始数据、最小反例和至少一个受控缓解/无效验证。

## 3. 2×2 模型矩阵

所有配置都放入 `configs/a1/`。派生 YAML 的文件头必须记录父配置、`end2end` 值和 SHA-256。

| ID | MoE | NMS | 定义 | 必须验证 |
| --- | --- | --- | --- | --- |
| A | off | on | native YOLO26 常规检测基线 | 常规 NMS 实际执行 |
| B | off | off | native one-to-one end-to-end 基线 | `end2end=True`，不走 NMS |
| C | on | on | MoE 对常规检测的影响 | router 正常，NMS 实际执行 |
| D | on | off | A1 核心实验 | one-to-one、router、导出和 NMS-free 后处理均通过 |

候选父配置为：

```text
ultralytics/cfg/models/26/yolo26.yaml
ultralytics/cfg/models/26/yolo26-master-n.yaml
```

在任何长训练前，运行配置审计并写入 `artifacts/a1/config_audit.json`。审计内容包括参数量、
head 类型、`end2end`、`one2one_cv2/one2one_cv3`、训练输出分支、推理输出形状和 postprocess
调用链。未满足矩阵定义的配置不得纳入结果表。

## 4. 数据、预算与随机性

### 数据分层

| 阶段 | 数据 | 用途 | 可否作为正式结论 |
| --- | --- | --- | --- |
| Smoke | COCO8 | 环境、模型构建和单 epoch 链路 | 否 |
| P0/P1 pilot | COCO-mini 固定 manifest | 快速消融与故障定位 | 仅 pilot |
| P1/P2 主实验 | COCO train / COCO val，或声明的固定子集 | detect 最终指标 | 可以 |
| P2 扩展 | COCO-seg 或 COCO-pose | 任务扩展 | 可以 |

COCO-mini 必须是可追踪对象：提交 `train.txt`、`val.txt`、生成脚本、类别/实例统计和 SHA-256。
四格与全部 seed 只能使用同一 manifest。

### 统一训练预算

初始使用单卡 24 GB。统一基础配置锁定 imgsz、batch、epoch、optimizer、学习率日程、增强、AMP、
workers 和预训练策略。除 MoE 与 NMS/end-to-end 所需开关外，A/B/C/D 不得改变任何变量。

主实验固定 seed `{0, 1, 2}`。资源不足时可先跑一个 seed 作为 pilot，但不得把 pilot 当作 P1/P2
结论；应缩小数据规模而不是改变矩阵变量。

## 5. 分阶段计划

### Phase S：Smoke（已完成）

- 锁定官方 SHA，建立隔离环境，运行 `yolo checks`。
- 运行 Agent quick 套件和 COCO8、1 epoch 的 end-to-end detect。
- 保存环境日志、36/36 Agent 结果、训练日志、`args.yaml`、`results.csv` 和权重路径。

通过标准是命令无人工交互、CUDA 可用、train → val → best checkpoint 产物完整。

### Phase P0：原生 detect 与 one-to-one 基线（已完成）

1. 固定 COCO-mini manifest 和统一配置。
2. 跑 A、B 各一个 seed，定位检测头、assigner、loss、predict、validator、export 的输入输出。
3. 对 B 记录 end-to-end 输出形状、候选数和同类高 IoU 重复框率，确认没有 NMS 调用。
4. 对 A、B 测 GPU/CPU batch=1：预热 50 次、正式 200 次；分开保存 preprocess、inference、
   postprocess、total 的原始样本。

实际 P0 使用同一个 `yolo26n.pt` 预训练 checkpoint，在完整 COCO val2017 的 5,000 张图片上
复现 A（one-to-many + NMS）与 B（one-to-one，NMS-free）。A/B mAP50-95 分别为 0.402/0.395，
B 下降 0.7 AP；固定 16 图运行时追踪中 A 调用 16 次真实 NMS kernel，B 为 0。GPU/CPU batch=1
逐样本延迟、mean/std/p50/p90/p99 和原生验证日志均已保存。

P0 通过标准：固定版本上的 A/B 可复现，有 COCO-mini 或 COCO val mAP50-95、非空检测下的
NMS 路径证据，以及 CPU/GPU batch=1 延迟原始数据。

### Phase P1：完整 2×2 与闭环

1. 对 C、D 先跑同预算 pilot；检查 MoE auxiliary loss、expert load、top-k、router entropy、
   Gini 和 expert utilization。
2. 对 2×2 跑完整 COCO-mini 矩阵。D 达到收敛门槛后，对 A/B/C/D 跑三个 seed 主实验。
3. 每格完成 train、val、predict 和 ONNX export；报告导出成功/失败、输出形状及 PyTorch/ONNX
   一致性。
4. 汇总 mAP、延迟、显存、参数量、FLOPs、router 指标与置信区间，禁止只挑单次最佳结果。

P1 目标：D 相对 C 的 mAP50-95 掉点不超过 2；CPU batch=1 total latency 有可测改善；四格均有
完整复现实验包。

### Phase P2-A：seg 或 pose 扩展（后续候选）

只有 D 的 detect 结论跨 seed 稳定后才开始：

1. 选择仓库支持更完整、数据准备更稳定的 `seg` 或 `pose`。
2. 复用 detect 的 one-to-one/NMS-free 实现，不重写一套不可比较的分支。
3. 完成 A/B/D 的最小矩阵：先单 seed pilot，再重复关键格。
4. 报告 box mAP 与 mask mAP 或 pose mAP；单独计时任务特定后处理和导出。
5. 比较 detect 与扩展任务的 router 分布、匹配数量、重复框率，判断结论是否跨任务稳定。

### Phase P2-E：detect End-to-End 精度主线（当前优先）

P1 已显示 B/D 的 End-to-End 主效应约为 `-0.00734 mAP50-95`，因此 P2 先不扩展任务，
直接分析 detect 侧 one-to-one 监督与后处理造成的精度差距：

1. 固定 r28 数据、checkpoint、seed、输入尺寸和评测口径，建立 A/B/C/D 的 recall、每图正样本数、
   matched IoU、未匹配 GT、分类置信度、分类/box/DFL loss 及尺度分档误差基线。
2. 对比 one-to-many 与 one-to-one assigner 输出，统计正样本数量、匹配重叠、低 IoU 匹配、未匹配 GT
   和分类置信度分布，先判断差距来自召回、匹配、分类还是定位。
3. 只修改 one-to-one assigner/loss 的一个因素进行短程受控 pilot；MoE、数据、优化器、预算和
   NMS-free 推理路径保持不变。
4. 用固定 seed 的 on/off 对照验证是否改善 recall、匹配质量和定位误差，不能只看单次 mAP。
5. 只有 pilot 显示明确方向且不破坏 NMS-free 闭环，才扩大到三 seed；否则记录为无效缓解并保留原始证据。

### Phase P2-B：机制级负结果（辅助/备用主线）

若 D 不收敛、掉点超过阈值或无速度收益，停止盲目扫参，转入下表的最小诊断：

| 现象 | 必须保存的证据 | 受控验证 |
| --- | --- | --- |
| 匹配冲突 | 每图正样本数、匹配重叠率、loss 曲线 | 固定 router；one-to-many 主分支 + one-to-one 辅助分支 |
| 路由坍塌 | expert load、entropy、top-k、Gini、aux loss | 温度、aux loss、capacity、warmup 的一次 on/off |
| 梯度稀疏 | backbone/router/head 梯度范数和零比例 | detach 边界和 loss 权重的最小对照 |
| 推理漂移 | train/eval expert id、权重和输出差异 | 固定路由或 dense fallback 对照 |
| 输出重复 | 候选数、同类高 IoU 重复率、NMS 前后差 | 检查 end-to-end postprocess、top-k、max_det |
| 无加速 | 各阶段 latency 与 profiler | 排除数据复制、Python 后处理和 I/O 主导 |

负结果必须提供失败复现命令、原始曲线、最小反例和已验证的缓解或无效措施。仅报告“mAP 低”
不构成 P2 负结果。

## 6. 指标与性能协议

### 准确率与统计

- 主指标：`metrics/mAP50-95(B)`；同时报告 mAP50、precision、recall、class-wise AP。
- 每格报告三个 seed 的 mean ± std、单次原始值、训练时长与 checkpoint 选择规则。
- 对“掉点 ≤2”同时报告绝对差、相对差、seed 方差和置信区间。

### 延迟

- 固定 batch=1、设备、CPU 线程、输入尺寸、输入来源、预热次数和计时次数。
- GPU 计时以 `torch.cuda.synchronize()` 包围；CPU 使用相同输入和固定线程数。
- 必报 total latency，同时拆分 preprocess、model inference、postprocess；不能只报 forward。
- 原始逐次样本保存为 `latency_samples.csv`，汇总 n、mean、std、p50、p90、p99、min、max。

### 路由和 one-to-one 诊断

主实验逐 epoch 保存 router entropy、expert load、top-k 分布、Gini、aux loss、梯度范数；验证阶段
保存候选数和重复框率。原始数据用 CSV/JSON，图表脚本只能读取这些原始文件。

## 7. 代码、证据与提交结构

```text
configs/a1/                 # 四格 YAML、数据 manifest、配置审计
scripts/reproduce/          # 训练、验证、延迟、聚合脚本
scripts/a1/                 # 匹配、路由、NMS-free、导出诊断
artifacts/a1/               # manifest、CSV、JSON、图表、日志摘要
reports/a1/                 # 2×2 表、P2 报告、limitations.md
smoke/a1/                   # 已完成的准入 Smoke 和原始轻量证据
```

每个运行目录至少有：`command.txt`、`git_state.json`、`environment.json`、`args.yaml`、
`dataset_manifest.json`、`results.csv`、`latency_samples.csv`、`router_stats.csv`、
`export_report.json`。大 checkpoint 保留在服务器，manifest 记录路径、大小和 SHA-256；不强制提交到 Git。

## 8. Go / No-Go

| 决策点 | Go 条件 | No-Go / 降级 |
| --- | --- | --- |
| P0 | A/B 可训可测，NMS-free 断言成立 | 先修复 head/validator/postprocess，不进入 MoE |
| P1 pilot | D 收敛且 router 无明显坍塌 | 转入 P2-E 精度诊断，不扩大数据 |
| P1 main | 精度差与延迟收益达到门槛 | 形成差距归因与受控缓解对照 |
| P2-E | recall/匹配/分类/定位差距有可复现归因 | 记录无效缓解，转备选 P2-A/P2-B |
| P2-A/B | P2-E 无明确改进方向或资源不足 | 仅在 P2-E 证据封存后选择扩展或机制负结果 |
| 发布 | diff、证据、限制、脚本齐全 | 标为 experimental，不宣称生产可用 |

## 9. 最终交付清单

- [ ] `BASE_REF`、`FINAL_REF`、`git diff` 和新增 commit 清单
- [ ] A/B/C/D 配置、配置审计和固定数据 manifest
- [ ] 三 seed 原始结果与准确率/延迟/显存汇总表
- [ ] ONNX 导出与 PyTorch/ONNX 一致性报告
- [ ] router、匹配、重复框、梯度诊断原始数据及绘图脚本
- [ ] seg/pose 扩展结果，或机制级负结果包
- [ ] `README.md`、`limitations.md` 与独立复现说明

## 10. 当前下一步

P0 与 P1 r28 已完成并封存。P1 的效率 profiling 和 dispatch 原型表明，MoE 的主要开销来自
路由同步、Python dispatch、索引和聚合；当前没有可直接合入的加速实现。

P2 当前主线已切换为 detect 侧 End-to-End 精度诊断：`smoke/a1/p2_e2e_precision_r1/`。
首轮固定 r28 A/B/C/D checkpoint 和 pilot val512，不改模型，只采集 recall、one-to-one 正样本数、
匹配 IoU/重叠、未匹配 GT、分类置信度以及 box/DFL/class 误差，确认差距来源后再进行单因素
assigner/loss pilot。r1 已在 GPU1 完成三 seed、四格、固定 val512 的只读诊断；MoE 路由机制目录
`smoke/a1/p2_mechanism_r1/` 保留为辅助证据，不能替代 P2-E 的精度主线，也不能把单次死专家写成
全局路由坍塌。

随后发现首版 `p2_e2e_loss_r2` 的脚本导入了远端另一个 editable checkout，导致 one-to-one
分类增益处理组实际上与中性对照完全相同；该目录已封存并标记为无效。修正版
`p2_e2e_loss_r2_corrected` 已强制当前仓库根目录位于 `sys.path[0]`，并通过单 batch 梯度审计
确认 `gain=1.0` 与 `gain=1.2` 真正产生不同 loss/梯度，正在按同一四组序列重新运行。
修正版 r2-c 已完成。固定 val512 的 B 处理组 mAP50-95 相对对照为 `-0.000756`，D 处理组为
`+0.000311`；该单 seed、5 epoch 的微小差异不足以支持正式收益声明。下一步应封存 r2-c，
优先选择新的 one-to-one assigner/loss 单因素，而不是直接扩大训练预算或宣称增益有效。

P2-E r3 已完成：以 `topk2=1`/`topk2=2` 对照 B、D 两组，固定 seed=260829、5 epoch、
pilot val512。`topk2=2` 没有形成预期的正样本增加（每图正样本仍约 6.87，重复/重叠均为
0），但分类损失上升约 19%～23%，预测数和 FP 增加约 29%～32%，mAP50-95 分别下降
0.032789（B）和 0.023616（D）。匹配 IoU 略升、box/DFL 未恶化，因此主要机制是分类分数
校准受到扰动，而不是定位匹配变差。完整诊断见 `smoke/a1/p2_e2e_assigner_r3/`；该处理组
被拒绝，当前保留 `topk2=1` 基线。下一步若继续，只增加 assigner 中间量审计并做单因素候选
生成/冲突消解实验，不直接长训。

### P2-F：小规模效率筛选（r1，已完成）

为判断 MoE 是否值得进入下一轮长训，固定 GPU1、batch=1 性能口径，新增仅在 backbone 层 8
使用 MoE 的六组筛选：Dense ratio 4/6 等活动 MLP 计算对照，以及 2/4 experts × Top-1/Top-2。
六组均从同一零增益 initializer 进行 1 epoch、5000/512 pilot，并在固定 val512 上复评。

固定 val512 mAP50-95 为 `0.426750–0.427500`，相对匹配 Dense 的最大绝对差仅 `0.000424`，
故没有可辨识的精度差异；但 GPU1 E2E batch=1 中 Dense 为 `6.242–6.369 ms`，MoE 为
`9.094–9.405 ms`，慢 `45.7%–47.7%`，吞吐下降约三分之一，峰值显存仅增加 `0.3–1.3 MiB`。
2→4 experts 几乎不改善延迟，Top-2 只额外增加约 `0.22–0.26 ms`。这与 P1 内部 profiling
一致，表明 Router、索引/分发和聚合是主要成本，而非专家矩阵乘法。

因此 r1 的效率门禁结论为：Dense 对照保留；四个 late-layer MoE 变体全部拒绝进入长训。
该结论是效率筛选结论，不把 1 epoch 的近似等精度写成 MoE 正式收益；原始证据见
`smoke/a1/p2_efficiency_screen_r1/`。若后续继续 MoE，必须先做设备端路由融合/编译优化并
重新通过同一 benchmark，否则 P2 继续 one-to-one assigner/loss 精度主线。
