# A1 P2-E End-to-End 精度诊断 r1

## 目标

P1 r28 显示 B/D（one-to-one、NMS-free）的 End-to-End 主效应约为 `-0.00734 mAP50-95`。
本轮不改模型、不重训，只用固定验证集和 r28 正式 checkpoint，拆解差距来自召回率、one-to-one
匹配质量、分类误差还是定位误差，为下一轮单因素 assigner/loss pilot 提供依据。

## 固定边界

- 公共基线：`acce839c7e895d6b179de7f7093fa879e237cc7b`
- 来源实验：`p1_factorial_medium_r28`，A/B/C/D 三个 seed 的正式 `last.pt`
- 数据：固定 pilot val512 manifest，不改图像顺序、标签和类别映射
- 设备：单卡 GPU1；输入 640×640；eval；batch=1
- 检测路径：A/C 使用 one-to-many + NMS，B/D 使用 one-to-one + NMS-free
- 本轮禁止：修改 assigner/loss、修改 router、增加专家、改变训练预算或把诊断结果写成新的 mAP 结论

## 必采指标

1. 总体与分格：mAP50-95、mAP50、precision、recall、每图 GT/预测数。
2. 召回归因：每图 TP/FP/FN、未匹配 GT 数量、按目标尺寸（small/medium/large）分档 recall。
3. 匹配质量：one-to-one 正样本数、matched IoU 均值/分位数、低 IoU 匹配比例、匹配重叠/冲突率。
4. 分类误差：TP/FP 分类置信度分布、类别混淆统计、分类 loss（若验证路径可取）。
5. 定位误差：TP 的 IoU 分布、box loss/DFL loss（若验证路径可取）、small/medium/large 定位分档。
6. 可复现性：checkpoint SHA-256、manifest SHA-256、逐图原始 CSV/JSON、命令、环境和 git 状态。

## 判定与下一步

- r1 只做归因，不以一次 pilot 的 mAP 变化判定成功。
- 若差距主要由 recall/匹配造成，r2 只改变一个 one-to-one assigner 因素；若主要由分类或定位造成，
  r2 只改变对应 loss 权重/形式。MoE、数据、优化器、预算和 NMS-free 推理保持不变。
- r2 先单 seed 短程受控 pilot；只有 recall、匹配质量或定位误差出现一致改善且闭环仍为 NMS-free，
  才扩大到三 seed。
- MoE 路由统计继续作为并行诊断，不能替代本协议的精度归因，也不能由单次死专家推出全局坍塌。
