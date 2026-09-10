# 同图 padding 路由诊断

2026-09-07，评测专用，不训练、不改权重。研究假设：空间均值汇聚可能使MoE路由对padding敏感；
目前不是已证实的因果机制，不能据此直接修改Router。

对D alpha0/alpha0.1两个末轮last.pt分别进行方形/矩形val512验证，共4次；统一GPU1、FP32、batch1、
640、禁止融合、原生Top-2、不加噪声、不干预专家选择。记录以下证据：

- 每图实际输入尺寸、padding比例、原图内容裁剪后的像素SHA（先排除缩放差异）；
- 每层Router的空间均值logits、实际Top-2专家ID及归一化权重、输入特征通道均值；
- 专家集合切换数与按专家ID对齐的权重L1差异，不把专家输出顺序改变当成集合切换；
- 同图conf0.25下TP/FP变化，与路由切换是否同时发生；标准mAP另保留；
- 按图像路径配对，不依赖rect dataloader的排序；checkpoint前后SHA不变。

源码依据：EfficientSpatialRouter先avg_pool2d，再卷积/BN投影，最后对空间维取均值得到global_logits。
hook只读取张量，返回None，不改变模型forward结果。更改padding也会影响backbone/attention等，
因此即使路由切换与检测变化同时出现，也不能直接宣布它是唯一原因；必要时另做受控路由干预。

新增脚本：`scripts/a1/audit_p2_padding_routes.py`；测试：`tests/test_p2_padding_routes.py`。
远端输出：`/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_pilot_r3/padding_routes_r1/`；
日志为同级`padding_routes_r1.log`。启动采用nohup；SSH断连不影响，服务器重启不会自动恢复。
自身allocator限额4GiB，按继承验证器定期检查22GiB总显存线；超限退出自身验证，不操作其他任务。

本文件是启动计划，不代表已完成。训练日志完全复现属于后续审计，本脚本不声称覆盖该项。
