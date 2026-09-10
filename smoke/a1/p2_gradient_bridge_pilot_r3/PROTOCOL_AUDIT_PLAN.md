# 评测口径一致性验证（2026-09-07）

目标：解释训练内验证与独立复评的微小增益变号，不训练、不改权重、不择优选择结果。

四个末轮last.pt保持与reevaluation_r1相同SHA；固定val512、FP32、640、conf0.001、max_det300、workers0。
每个权重做batch={1,8}×rect={False,True}×fuse={False,True}八组合，共32次串行验证。
另统计conf0.25下FP/FN，记录实际输入shape、BN模块数、图像集合SHA及前后checkpoint SHA。

源码证据：训练器验证dataloader batch=训练batch×2=8，DetectionTrainer构建val数据时强制rect=True；
训练内验证不走AutoBackend融合。前次直接创建DetectionValidator采用默认rect=False、batch1，
AutoBackend默认融合。这三个因素需分别对照，不能把差异一概归为随机波动。

对照解释：固定rect/fuse比较batch；固定batch/fuse比较rect；固定batch/rect比较fuse。
rect=True时batch也会改变分组padding，因此方形输入的batch对照用于隔离纯batch效应。
通过BN数量断言融合开关实际生效，通过实际shape断言方形输入为640×640。
最后检查b8/rectTrue/unfused能否复现训练日志；若不能，保留未解释差异，不提前归因。

GPU1启动检查时余量不足，未加载模型；转GPU0统一运行。启动前已有显存<12GiB，自身allocator限额8GiB，
每8batch检查22GiB总显存线，超限只退出自身验证，不操作其他进程。nohup使SSH断连不影响进程，
服务器重启不会自动恢复；各完成单元即时保存。输出目录拒绝覆盖，不自动重训。

远端输出：`/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_pilot_r3/protocol_audit_r1/`
日志：`/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_pilot_r3/protocol_audit_r1_gpu0.log`
状态：输出目录`evidence.json`。此文是启动计划，不代表32项全部完成。

脚本`scripts/a1/audit_p2_validation_protocol.py`按仓库YOLO Agent Skill的研究扩展路径调用原生验证器，
仅在当前进程内临时设置后端fuse参数，不修改训练/模型源码。源码/输入/权重校验和dry-run通过，
脚本ruff、format、py_compile通过。全仓库既有2843项ruff问题、126个待格式化文件，codespell未安装。
