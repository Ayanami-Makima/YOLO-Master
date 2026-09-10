# P2 梯度通路配对 pilot r3

2026-09-06。根目录 `/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_pilot_r3`。
本轮为正确性和资源守护修订，不是新的超参搜索：唯一研究因素仍为B/D×one-to-one共享梯度alpha0/0.1。
旧r1因gain日志形状错误停止；r2因自身显存限额诱发框架自动减batch而停止，均不纳入配对结果。

## 最新状态：训练及统一复评完成

四格训练已于2026-09-06 21:55:50完成；随后在GPU0以FP32、batch1统一复评四个末轮last.pt。
B alpha0/0.1 mAP50-95=0.42560062/0.42521384；D=0.42518009/0.42572750。
D小幅改善但未达到内部扩展线，且conf0.25下FN增加4；不启动扩大训练。
详见 `REEVALUATION_REPORT.md` 与 `reevaluation_r1/evidence.json`。D alpha0有一次epoch级恢复；
训练内与独立复评差值变号，不能宣称稳定收益。以下启动快照仅为历史记录。

## 启动快照

20:32:17（北京时间）已在GPU1启动首格B alpha0。已核对1/5轮、1250step/epoch、实际batch4，
未触发自动减batch；自身约9608MiB、设备总占用约19399MiB。四格真实batch4预检、保存/退出/恢复
及冻结门禁通过，记录在 `launch_gate.json`，24项测试见 `test_results.xml`。
远端实时状态为 `train_status.json`，序列日志为 `logs/train_sequence.log`；
当前训练日志为 `logs/train_b_alpha0p0_seed260829.log`，各格epoch记录为 `train/<格名>/results.csv`。
该段是启动快照，不表示四格已完成。

精确源码快照 `source_snapshot.tar.gz` 的SHA256为
`f5c46069043338a73f620a4a3286534e9e902db25ba034fb1a83dac98f85331f`，包含协议锁定文件与门禁/测试源码。

## 固定设置

四格串行 B alpha0 → B alpha0.1 → D alpha0 → D alpha0.1，各从r28 seed260829原始initializer独立开始。
5000train/512val，5epochs，batch4（每轮1250step），640，seed260829，SGD lr0=1e-4，lrf=0.2，
momentum=0.9，weight_decay=0.0005，nbs=16，warmup_epochs=0.5，warmup_bias_lr=0，
warmup_momentum=0.8，AMP=False，workers=0，deterministic=True，patience=0，cos_lr=False。
mosaic/mixup/copy_paste=0；其余增强保持相同默认值，完整配置见各格args.yaml。
训练层4/6/8/23，其余层、BN参数/统计、459232个官方C3k2 base参数冻结。继承逐通道gain策略：
初始lr=0.01（100×）、weight_decay=0、不参与warmup；普通参数及router初始lr=1e-4。
MoE硬Top-2、专家数、辅助损失不改；one-to-one候选topk7/topk2=1、loss gain不改。

## 资源与预算保护

GPU1，每次最多一格；启动前设备已有显存≤10240MiB，自身CUDA allocator限额11GiB。
每15秒检查设备总占用，超过22528MiB只终止本序列的子进程。该轮询不是绝对硬件上限，
其他任务突增时仍可能短暂越线；不操作其他人的任务。

每个batch核对trainer.args.batch、trainer.batch_size、dataloader.batch_size都等于4，且实际训练图数
等于协议；预检记录真实每轮步数。通过设置引擎已检查的OOM重试计数上限，禁止自动减batch；
遇OOM直接失败，不能偷偷改变实验预算。256train/128val的2epoch极小预检使用batch4、64step/epoch，
在保存epoch1后主动退出并恢复epoch2；这些权重全部丢弃。

SSH断连不影响nohup序列，epoch级保存。资源类中断有有效checkpoint时重跑该入口并从已存epoch恢复；
无checkpoint或正确性错误停止。服务器重启后不会自动拉起进程，需重新运行同一序列命令。
恢复不等于逐batch无损或与未中断训练数值完全相同；中断回退的额外计算必须在结论中披露。

## 验证与结果口径

不重叠图像三seed复核见 `../p2_gradient_bridge_holdout_20260906/`。24项单测/配置回归覆盖前向、
梯度缩放、head梯度不变、eval/export-mode等价、checkpoint序列化、gain向量、冻结和真实batch漂移。
每次启动锁定源码/输入SHA并检查实际criterion及alpha，每epoch核对冻结状态。

统一最后epoch EMA作为主对照；训练结束返回的best指标仅次要参考。后续还需固定val512 PR/recall、
分类/定位、conf0.25 FP/FN、router辅助损失与检测梯度量级，以及真实导出后端验证。
内部三seed扩展线：D ΔmAP≥0.001且recall不降、B ΔmAP≥-0.001并检查导出；不是A1标准或显著性标准。
没有证据前不宣称提高精度或P2完成。

研究脚本采用仓库YOLO Agent Skill允许的Python API路径。新增脚本lint/format通过；head.py既有lint
问题未批量改动，全仓库CI不宣称通过。旧r2源码快照及invalid状态仍保留，以便追溯配置偏差。
