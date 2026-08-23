# A1 P2 方案：MoE × End-to-End NMS-free 闭环与机制验证

## 1. 目标与边界

### 核心问题

在固定 YOLO-Master 公共基线、数据、训练预算和评测口径下，验证 MoE 路由能否与 one-to-one
监督稳定共存，使 end-to-end 检测不依赖 NMS 仍保持可接受精度，并取得可重复的端到端延迟收益。

这不是“再造一个检测头”。任务是核验已有 one-to-many / one-to-one、训练、推理、导出和评测
路径是否构成真正的 NMS-free 闭环，并定位 MoE 路由与一对一监督是否发生冲突。

### 不可变验收边界

| 项目 | 规则 |
| --- | --- |
| 公共基线 | `acce839c7e895d6b179de7f7093fa879e237cc7b` |
| 主任务 | COCO detect，MoE on/off × NMS on/off 2×2 |
| 共同变量 | 数据 manifest、imgsz、batch、epoch、优化器、增强、设备、seed、评测脚本 |
| 主指标 | mAP50-95、GPU/CPU batch=1 端到端延迟及标准差 |
| P2 目标 | seg/pose 扩展，或机制级、可复现的负结果 |
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

P1 通过后，完成以下任一条主线：

1. 将 detect 侧已验证的 MoE + NMS-free 方案扩展到 `seg` 或 `pose`，并完成最小对照；或
2. 形成机制级负结果：定位梯度稀疏、匹配冲突、路由坍塌、推理路由漂移或输出重复的具体条件，
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

### Phase P0：原生 detect 与 one-to-one 基线

1. 固定 COCO-mini manifest 和统一配置。
2. 跑 A、B 各一个 seed，定位检测头、assigner、loss、predict、validator、export 的输入输出。
3. 对 B 记录 end-to-end 输出形状、候选数和同类高 IoU 重复框率，确认没有 NMS 调用。
4. 对 A、B 测 GPU/CPU batch=1：预热 50 次、正式 200 次；分开保存 preprocess、inference、
   postprocess、total 的原始样本。

P0 通过标准：固定版本上的 A/B 可复现，有 COCO-mini mAP50-95 与 CPU/GPU 延迟原始数据。

### Phase P1：完整 2×2 与闭环

1. 对 C、D 先跑同预算 pilot；检查 MoE auxiliary loss、expert load、top-k、router entropy、
   Gini 和 expert utilization。
2. 对 2×2 跑完整 COCO-mini 矩阵。D 达到收敛门槛后，对 A/B/C/D 跑三个 seed 主实验。
3. 每格完成 train、val、predict 和 ONNX export；报告导出成功/失败、输出形状及 PyTorch/ONNX
   一致性。
4. 汇总 mAP、延迟、显存、参数量、FLOPs、router 指标与置信区间，禁止只挑单次最佳结果。

P1 目标：D 相对 C 的 mAP50-95 掉点不超过 2；CPU batch=1 total latency 有可测改善；四格均有
完整复现实验包。

### Phase P2-A：seg 或 pose 扩展（优先）

只有 D 的 detect 结论跨 seed 稳定后才开始：

1. 选择仓库支持更完整、数据准备更稳定的 `seg` 或 `pose`。
2. 复用 detect 的 one-to-one/NMS-free 实现，不重写一套不可比较的分支。
3. 完成 A/B/D 的最小矩阵：先单 seed pilot，再重复关键格。
4. 报告 box mAP 与 mask mAP 或 pose mAP；单独计时任务特定后处理和导出。
5. 比较 detect 与扩展任务的 router 分布、匹配数量、重复框率，判断结论是否跨任务稳定。

### Phase P2-B：机制级负结果（备用主线）

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
| P1 pilot | D 收敛且 router 无明显坍塌 | 转入 P2-B 诊断，不扩大数据 |
| P1 main | 精度差与延迟收益达到门槛 | 形成负结果与受控缓解对照 |
| P2-A | detect 结论跨 seed 稳定 | 不扩展任务，完成 P2-B 负结果包 |
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

已完成：官方基线锁定、隔离环境、Agent quick 36/36、COCO8 单 epoch end-to-end detect Smoke。

下一项工作是建立 `configs/a1/` 的 A/B/C/D 派生配置和 COCO-mini manifest，先完成 P0 的 A/B
原生对照及 CPU/GPU batch=1 端到端延迟脚本，再决定是否进入 MoE 2×2。
