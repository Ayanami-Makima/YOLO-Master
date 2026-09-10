# P2 梯度通路诊断

状态：三 seed 诊断完成；没有 optimizer 更新，没有新训练的精度结果。

使用 r28 B/D、seed260829/260830/260831 的 last.pt，在相同前 32 张训练图上分开回传
one-to-many 和 one-to-one 的检测损失，再在诊断内为 one-to-one 输入加入 0.1 倍梯度通路：

```python
bridged = x.detach() + 0.1 * (x - x.detach())
```

| seed | B factor 梯度 cosine | D factor 梯度 cosine | D router 梯度 cosine | D router 负 cosine 图数 |
| --- | ---: | ---: | ---: | ---: |
| 260829 | 0.666563 | 0.641816 | 0.684941 | 2/32 |
| 260830 | 0.610337 | 0.619456 | 0.548093 | 4/32 |
| 260831 | 0.652379 | 0.651306 | 0.706256 | 1/32 |

cosine 是 bridge one-to-one 梯度与 one-to-many 梯度的方向相似度，范围 -1 到 1。
正值表示此处大体同向，负值表示局部反向；不等同于最终训练收益或损害。
原生 one-to-one 检测梯度在三个 seed 上都未传到 factor/router；bridge 后相应梯度非零，
前向 boxes/scores 误差始终为 0，所有 state_dict 张量不变。

这些结果暂不支持“普遍的直接梯度冲突”这一解释；原生路径已有 detach，
one-to-many 检测梯度及 MoE 辅助损失仍能训练相关参数。辅助损失不在本次分支测量内。
不能将本结果描述为“router 没有梯度”或“detach 已证实导致 mAP 下降”。

配置：CPU、2 threads、640、batch1、无增强、所有 BN/base 冻结；264 GT/32 图。
192 个 seed×格×图配对测量重复使用同一 32 图集合；没有独立的 192 图样本。
样本是训练集合开头的固定子集，需另取互斥子集检查选择偏差；不构成 COCO 性能评估。

## 复现

远端仓库 `/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium`，Python 使用同级 `.venv/bin/python`。

```bash
/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python scripts/a1/audit_p2_gradient_bridge.py \
  --root /data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28 \
  --data configs/a1/p1_pretrained/pilot_data/coco.yaml \
  --output /absolute/new-output/evidence_seed260829.json --images 32 --seed 260829
```

分别替换 seed 为 260830、260831，三个 JSON 写到同一新目录。脚本拒绝覆盖现有输出。
随后用 `scripts/a1/summarize_p2_gradient_bridge.py --input-root <目录> --output <新summary.json>` 聚合。

`evidence_seed*.json` 保存 checkpoint 与图像哈希、逐图梯度和 state 不变检查；
`summary.json` 保存配对汇总。结果汇总前核验三 seed 的图像哈希一致和前向/权重不变断言。

下一研究方案见 `../P2_REQUIREMENTS_AND_RESEARCH_REVIEW_20260905.md`。
