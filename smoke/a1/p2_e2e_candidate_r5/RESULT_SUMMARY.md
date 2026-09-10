# P2-E r5：训练消融无效，结论已撤回（2026-09-05 审计更正）

原 r5 的四组训练结束、checkpoint 和固定 val512 指标仍作为原始记录保留，但不能据此判定
`tal_topk=10` 无效。重放训练脚本的导入环境发现，它加载的是
`/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master/ultralytics/utils/loss.py`，
该版本不识别 `A1_E2E_O2O_TAL_TOPK`，设置 10 时实际 assigner 仍为 7。
B 对照/处理的全部 927 个状态张量、D 的全部 1636 个状态张量分别完全相同。

原评估脚本加载了正确的新 checkout，因此候选统计确实因 topk 改变；
但这只是对同一组权重的事后 assigner 审计。assigner 不参与推理，故相同 mAP 并不能证明
“正确训练的 topk=10 没有收益”。原“拒绝 topk=10、无效缓解已完成”的结论撤回，
r5 标记为 `invalid_training_intervention`；原始文件不删除、不覆盖。

修复已在训练入口加入仓库路径固定及首 batch 的实际 criterion 参数校验。
补充 CPU 诊断在 r28 B/D seed260829 的相同 8 张训练图上验证：
实际 topk=7/10 的损失和 head 梯度相同（8/8），这里只说明该小样本未受到候选预算改变的影响。
分支梯度审计还确认：native one-to-one 检测损失到 factor/router 的梯度为零，
one-to-many 检测损失可以传到 factor/router。这与检测头 `detach()` 一致，
不能写成“MoE 总梯度为零”或“路由必然坍塌”；MoE 辅助损失未包含在该分支实验中。

当前研究转入固定输入、前向等价的梯度通路诊断，详见
`P2_REQUIREMENTS_AND_RESEARCH_REVIEW_20260905.md`。
审计证据：`p2_r5_validity_20260905/evidence.json`。
