# P2 one-to-one 梯度通路配对 pilot r2

**已停止且不得纳入配对结论。** 启动后发现框架在8GiB allocator限额下触发OOM，并自动把batch4改成2。
此前“恢复/冻结检查通过”只说明这些检查通过，漏查了真实batch；不等于锁定预算通过。
256图preflight实际为batch2（128step），并非512图。首格5000图pilot也变成2500step/epoch。
2026-09-06在首格epoch1尚未完成时主动停止序列和自己的训练进程，未启动其他正式格。
原文件全部保留；新修订r3加入每batch实际预算检查并禁止框架自动减batch，改为11GiB自身限额、
启动前外部占用≤10GiB，总阈值仍为22GiB。下文保留启动时设计和当时判读，实际状态以本更正为准。

日期：2026-09-06。执行根目录：`/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_pilot_r2`。
本目录为第二版执行脚本的 pilot，不替代 r28 P1 或上一轮机制诊断。

20:21:45（北京时间）四格串行训练已启动，首格B alpha0已进入第1/5轮，实际源码、topk7/topk2=1、
alpha0、459232冻结base参数均已核实。四格preflight与恢复测试全部通过，见 `launch_gate.json`。
这是启动快照，实时进展以远端 `train_status.json`、各格 `results.csv` 和日志为准。

## 单因素设计

- 四格串行：B alpha0 → B alpha0.1 → D alpha0 → D alpha0.1。
- alpha0 为原生 detach；alpha0.1 仅把 one-to-one 检测损失到共享特征的梯度开放 0.1 倍。
- one-to-many、one-to-one head 的损失权重、候选匹配 topk7/topk2=1 均不改。
- 默认、推理、export 路径不启用 bridge；本轮仅允许原生 Detect，不扩展到 seg/pose。
- 各格从 r28 seed260829 原始 initializer 独立开始；不使用预检或已训练的权重。

## 锁定预算

train5000/val512，5 epochs，batch4，imgsz640，seed260829，SGD lr0=0.0001，lrf=0.2，
momentum=0.9，weight_decay=0.0005，nbs=16，warmup_epochs=0.5，warmup_bias_lr=0，
warmup_momentum=0.8，AMP=False，workers=0，deterministic=True，patience=0，cos_lr=False。
mosaic/mixup/copy_paste=0，close_mosaic=0；其余增强由相同默认配置固定，完整值见各格 args.yaml。

训练层4/6/8/23，其他层冻结；全部 BN 参数与统计量、459232 个官方 C3k2 base 参数冻结。
继承 initializer 的逐通道 residual gain 优化器策略：初始 lr=0.01（100×）、weight_decay=0、
不参与 warmup。普通参数及 router 初始 lr=0.0001。不得将全模型所有参数统一描述为相同学习率。
MoE 保持原有硬 Top-2、专家数量和辅助损失设置，具体参数及源码/输入哈希见 protocol.json 和 requests。

## 正确性和恢复

- 新增及关联测试共 24 项：alpha0/native 等价、0.1 梯度缩放、head 梯度不变、推理/export-mode 等价、
  checkpoint 序列化、非法系数、向量 gain 记录、冻结状态检测及模型配置回归。
- 四格先用原有极小数据（512 train/128 val）跑 2 epochs；每格在保存 epoch1 后主动退出，
  再检查实际 optimizer/epoch 恢复到 epoch2。预检权重不进入 pilot。
- 每次启动核对源码/输入 SHA、真实 criterion、实际系数；首次 forward 核对 one-to-one 特征是否带梯度。
- 每 epoch 对冻结参数及 BN buffer 做哈希比较；记录三层逐通道 gain 的形状和范数。
- 后台序列不依赖 SSH 连接；每 epoch 保存 last.pt。显存/OOM等可识别中断有有效 checkpoint 时自动恢复，
  不从 initializer 自动重跑失败格。正确性错误停止序列，不继续下一格。
- 恢复是“最近已保存 epoch”级别，不保证与从未中断运行数值完全一致；中断前未保存的 epoch 需重算。
  服务器重启不会自动重启该后台进程，可重新运行相同序列入口；已完成格按完成回执跳过。

GPU1，最多一格；启动时已有占用须不高于 13312 MiB，自身 CUDA allocator 限额 8 GiB，
每15秒检查设备总占用，超过22528 MiB只终止本序列的训练子进程。有其他任务突增或CUDA非allocator
内存时仍可能短暂越线，不能把轮询声称为硬件级绝对上限。不会终止或更改其他人的任务。

## 结果判读

主对照统一使用最后 epoch 的 EMA checkpoint；训练结束自动产生的 best 验证指标只作为次要信息，
不可混用为四格最后 epoch 对照。后续补 val512 的 PR、recall、分类/定位、固定conf0.25 FP/FN及路由
辅助损失与检测梯度量级；真实导出后端仍需复核，export-mode 单测不是 ONNX 验收。
预先提出的三 seed 扩展线：D ΔmAP≥0.001且recall不降，B ΔmAP≥-0.001，再结合导出检查决定。
这只是内部预算筛选线，不是显著性阈值或任务书硬性门槛。当前不宣称精度收益或 P2 完成。

## 历史与检查边界

`p2_gradient_bridge_pilot_r1` 的首格在 epoch 日志记录处因把向量 gain 转 scalar 而失败，未启动正式
5000图训练。原日志和目录保留；修复后使用全新 r2，未覆盖/续接失败权重。
研究遵循仓库 YOLO Agent Skill 的研究脚本路径；普通 CLI 不覆盖受控梯度通路与分支审计。
doctor 确认正确 checkout 激活。新研究脚本 Ruff lint/format 通过；head.py 原有24条 lint 问题未批量
修改；全仓库仍有2843条 Ruff问题、126个待格式化文件，codespell不可用，不宣称全仓库 CI 通过。
