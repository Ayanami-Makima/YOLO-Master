# A1 P2 要求复核与下一步研究方案

日期：2026-09-05。当前优先完成检测侧监督机制的有效证据链，再决定是否做新的短训。
本次审计撤回 r5 的训练消融结论，并完成三 seed 的分支梯度诊断。

2026-09-06 执行更新：下文为9月5日预先制定的方案；现已完成不重叠32图三seed复核、
四格极小preflight及实际保存/恢复检查。r2因框架自动减batch已停止；修订后在GPU1启动真实batch4
的配对pilot r3。最新执行状态以 `p2_gradient_bridge_pilot_r3/README.md` 和
`A1_P0_P1_PROGRESS_REPORT.md` 第6.16节为准。

## 1 任务书要求

依据《腾讯犀牛鸟_YOLO-Master_实战课题任务书_细化版_20260822》第 6—7 页 A1 节。
源文件 SHA-256：`cf724551dad32a83075edc9465c0e1178722ec184e3ba767c3b087c20afc5012`。

> P2｜理想：扩展到 seg/pose，或给出有证据的负结果：梯度稀疏、匹配冲突、路由坍塌等机制分析。

核心问题是现有训练、推理、导出、评测能否形成 NMS-free 闭环，以及稀疏 MoE 路由是否与一对一
监督冲突。P2 是理想目标；seg/pose 扩展与有证据的机制负结果属于可选择的方向。
“必须提升 mAP”“每个专家都必须被选中”“必须 Top-2”“必须完成 seg 和 pose 两项”均非上述条款。
统一实验要求是同数据、同预算、同增广、同评测；提升/下降原则上至少三 seed，否则明确单 seed 局限。

另核对《历史成果、基线锁定与增量验收补充细则》第 3—4 页：P2 必须形成超出原有能力边界的
增量，例如机制分析、跨架构/跨后端矩阵、工具、性能恢复或高质量负结果。重跑旧表、重排报告
不能单独构成 P2。因此“发现源码已有 detach”本身也不是创新；应交付 MoE 条件下的梯度分布、
可复现冲突案例、受控干预结果及其适用边界。

| 条目 | 来源 | 本项目处理 |
| --- | --- | --- |
| seg/pose 扩展或机制分析 | A1 P2 原文 | 保留 seg pilot，当前研究检测侧机制 |
| ≥3 seed 或单 seed 局限 | A1 交付证据及共同红线 | 已有 r28 三 seed；新 pilot 单独声明范围 |
| 掉点 ≤2、CPU 延迟可测改善 | A1 P1 目标 | 不挪作每轮 P2 改动必须涨点的门槛 |
| one-to-one assigner/loss 精度主线 | 用户确认的研究方向 | 作为机制分析和受控缓解的具体实现 |
| 459232 base 参数、BN 冻结、硬 Top-2 | 本项目已锁定实验设计 | 新对照继续保持，避免同时改变多个因素 |
| 无死专家、entropy ≥0.5 等 | 早期内部路由筛选 | 不能冒充 A1 原文或作为当前唯一失败依据 |

## 2 已有证据与缺口

- r28：已有检测 2×2、多 seed 和效率材料；其正式结论与后续 pilot 分开引用。
- 效率筛选：支持当前实现存在额外路由/分发开销；不等于 MoE 在所有实现或预算下无收益。
- r2 corrected、r3、r4：有监督侧试验和诊断记录；本次没有逐项重审，沿用时仍须确认 runtime 证据。
- seg r1：已有单 seed A/B/D 的 train/val/export pilot。随机 mask/proto 初始化和短预算限制了
  跨任务精度解释；不能仅据低 mask mAP 宣称正式跨任务负结果或 P2 已验收。
- r5：**训练干预无效**，原先的“拒绝 topk=10”结论撤回。

## 3 r5 错误与修复

训练入口 `run_p1_bn_frozen.py` 未固定仓库路径，命令运行时由 editable 安装加载旧的
`YOLO-Master/ultralytics`。旧 loss 不识别 `A1_E2E_O2O_TAL_TOPK`，声明 10 时实际仍为 7。
B 对照/处理的 927 个状态张量、D 的 1636 个状态张量分别完全相同。

事后评估使用新代码，因此能观察 topk=10 的候选数增加，但它对比的是相同权重。
assigner 只用于监督诊断，不能通过事后切换 assigner 证明该训练改动没有精度收益。
原文件保留为历史数据，状态为 `invalid_training_intervention`。

已修复训练入口的源码导入，并新增实际 criterion 的 `topk/topk2` 与请求不一致时立即失败的检查。
新增源码文件路径和 SHA 记录；回归测试覆盖 7/10 正常实例、干预未生效、错误 checkout。
正式恢复实验还需要真实 first-batch 日志及受控梯度差异，不能只凭 dry-run 放行。

另外，原汇总中约 6.93 个最终正样本/图的均值排除了 4 张背景图，分母是 508。
不能直接和按全部 512 图汇总的约 6.88 数字比较。后续汇总须明确全图/有 GT 图分母，
有 GT 行缺失诊断数据必须报错，不能自动当作 0。

## 4 本轮新增机制证据

第一项使用 r28 seed260829 的 B/D checkpoint、固定最前 8 张 train 图、640、batch=1、CPU，
关闭增强、冻结 BN/base、无 optimizer 更新：新代码实际 topk=7/10 的 head 梯度差在该 8 图均为 0。
该事实仅说明候选预算对这组图无影响，不能代替修复后的完整训练实验。

同一测试分开回传 one-to-many 和 one-to-one 检测损失：one-to-one 对 factor/router 的梯度为 0，
one-to-many 对 D router 的梯度非零。源码 head 的 `detach()` 正好对应这一现象。
这属于原生设计：YOLOv10 官方实现也在 one-to-one 输入处 detach，不能认定是本项目的程序错误。
参考：[YOLOv10 官方检测头实现](https://github.com/THU-MIG/yolov10/blob/main/ultralytics/nn/modules/head.py)。

第二项只在诊断内替换 one-to-one 输入的梯度路径：

```python
bridged = x.detach() + 0.1 * (x - x.detach())
```

数值前向保持原样，反向对共享特征开放 0.1 倍梯度。0.1 是预先选定的探针强度，不代表最优超参。
使用相同 checkpoint、固定最前 32 张 train 图（264 GT）；分别测 native、bridge 与 one-to-many。
不包含路由辅助损失；无优化器更新，所有 checkpoint 参数和 buffer 保持不变。

seed260829 的结果：

| 测量 | B Dense | D MoE |
| --- | ---: | ---: |
| native one-to-one → factor 梯度 L2 均值 | 0 | 0 |
| bridge one-to-one → factor 梯度 L2 均值 | 0.817649 | 0.642310 |
| bridge 与 one-to-many 的 factor 梯度 cosine 均值 | 0.666563 | 0.641816 |
| factor 梯度 cosine <0 图数 | 1/32 | 3/32 |
| bridge 与 one-to-many 的 router 梯度 cosine 均值 | 不适用 | 0.684941 |
| router 梯度 cosine <0 图数 | 不适用 | 2/32 |
| 前向 boxes/scores 最大误差 | 0.0 | 0.0 |

暂时支持的解释：native one-to-one 不直接训练 router，少量接通后大多数样本梯度同向，存在少数
方向相反的样本。这既不能证明全局梯度冲突，也不能证明 bridge 会涨点。cosine 为负只是局部
梯度方向关系；不能直接等同于有害训练冲突。
seed260830/260831 已按同一 32 图协议重复完成。三 seed 使用相同图像，不构成独立的 96 图样本。

| seed | B factor cosine 均值 | D factor cosine 均值 | D router cosine 均值 | D router 负 cosine 图数 |
| --- | ---: | ---: | ---: | ---: |
| 260829 | 0.666563 | 0.641816 | 0.684941 | 2/32 |
| 260830 | 0.610337 | 0.619456 | 0.548093 | 4/32 |
| 260831 | 0.652379 | 0.651306 | 0.706256 | 1/32 |

共 192 个 seed×格×图的配对测量，只有 32 张独立图像。三 seed 的 native one-to-one 到 factor/router
梯度均为零；bridge 的前向误差均为 0，模型 state_dict 无变化。结果支持继续检验梯度通路这一因素，
但未证明 detach 导致了 mAP 差距，也未证明训练中没有其他形式的冲突。

## 5 下一步执行顺序

1. 三 seed 梯度复核已完成，已保存每图原始梯度范数、cosine、前向误差和权重不变证明。
   另取固定、互不重叠的 train 图子集验证样本选择敏感性。若解释不稳定，就继续诊断而非宣布机制。
2. 在明确操作确实生效后，准备独立的 bridge pilot：B/D × alpha=0/0.1，共四格。
   固定 r28 原始 initializer；train5000/val512、5 epochs、batch4、640、seed260829、SGD lr0=1e-4、
   lrf=0.2、momentum=0.9、weight_decay=0.0005、nbs=16、warmup_epochs=0.5、AMP=False、workers=0；
   训练层4/6/8/23、其余/BN/base冻结，MoE 参数和全部数据顺序保持配对。必须在新目录独立开始。
3. pilot 的执行门禁：alpha=0 与 native 的前向及梯度等价；alpha=0.1 前向仍等价、预期共享梯度
   非零；head 梯度和损失未被意外缩放；frozen base/BN 不变；resume 后 coefficient 和源码 SHA 不漂移。
   此处属于内部正确性检查，不是任务书额外要求。
4. pilot 比较固定 val512 的全 PR/mAP、recall、尺寸分档、分类/定位误差和固定 confidence=0.25
   的 FP/FN，同时统计 router 辅助损失与检测梯度的相对量级。用同一 checkpoint 选择规则；
   事先声明本轮筛选条件，训练后不改门槛。当前建议只有 D mAP 至少改善 0.001、recall不降，
   且 B mAP 回退不超过 0.001、NMS-free/导出有效时，才投入三 seed 正式验证。
   这些是预算筛选线而非显著性标准；三 seed 后仍需报告均值、方差和局限。
5. 若 bridge 无收益，封存具体条件、最小反例和受控无效缓解，形成 P2 机制报告；
   若有收益，再扩大 seed 和验证集，完整复核 NMS-free、导出和部署一致性。当前不再增加专家数。

## 6 当前执行状态与交付边界

CPU 源码/权重审计及全部三个 seed 的梯度诊断已完成。
新的 bridge 训练尚未启动，未修改检测头默认 forward 或发布算法改进。
本次通过脚本审计和原文逐项核对修正了研究路线，P2 仍在研究阶段，不能声明验收完成。

证据目录：`p2_r5_validity_20260905/`、`p2_gradient_bridge_r1/`。
研究过程的模块检查采用仓库 Agent Skill 允许的研究脚本路径，以便读取 CLI 不直接暴露的分支梯度。

验证：新增和关联测试 7/7 通过，五个变更 Python 文件的 Ruff lint/format、py_compile 和序列
脚本语法检查通过。全仓库检查仍有 2843 条 Ruff 问题、126 个文件需要格式化；本轮未批量修改。
远端未安装 codespell，未通过该项检查。以上研究脚本验证不代表全仓库 CI 已通过。
