# A1: MoE × End-to-End NMS-free Smoke

This directory records the admission smoke and pretrained P0 baseline for the Tencent Rhino-Bird A1 task:
**MoE × End-to-End: whether the existing one-to-one path can form an NMS-free
loop**.

## Scope and baseline

The original COCO8 run is an admission smoke, not an accuracy result. The later
pretrained COCO val2017 A/B comparison is the P0 accuracy and latency baseline;
neither run is a P1 MoE conclusion.
The code baseline is the public YOLO-Master commit:

```text
acce839c7e895d6b179de7f7093fa879e237cc7b
```

The smoke uses the native end-to-end configuration
`ultralytics/cfg/models/26/yolo26.yaml`, whose `end2end` option is enabled.
The one-to-one detection head is implemented in
`ultralytics/nn/modules/head.py`; the detect train/validation/prediction
entrypoints are under `ultralytics/models/yolo/detect/`, and the standard
assigner is `ultralytics/utils/tal.py:TaskAlignedAssigner`.

## Verified environment

| Item | Value |
| --- | --- |
| Host | `scut-server` |
| GPU | 2 × NVIDIA GeForce RTX 4090 (24 GB each) |
| Python | 3.10.20 |
| PyTorch | 2.1.2+cu121 |
| CUDA | 12.1 |
| NumPy / OpenCV | 1.26.4 / 4.10.0 |
| Ultralytics / YOLO-Master | 8.4.101 |

`yolo checks` passed. The Agent quick suite also passed **36/36** cases.

## Reproduce

On `scut-server`, use the isolated A1 environment:

```bash
cd /data/data2/TuJiajun/A1-smoke/YOLO-Master
bash scripts/reproduce/run_a1_smoke.sh
```

Or invoke the two smoke stages directly:

```bash
../.venv/bin/yolo checks
../.venv/bin/python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
../.venv/bin/yolo train \
  model=ultralytics/cfg/models/26/yolo26.yaml data=coco8.yaml \
  epochs=1 imgsz=64 batch=1 device=0 workers=0 plots=False \
  project=/data/data2/TuJiajun/A1-smoke/runs name=detect_e2e_coco8_1ep
```

## Result

The real GPU smoke completed one COCO8 epoch and validated the generated
`best.pt`. It produced `best.pt`, `last.pt`, `last_healthy.pt`, `args.yaml`,
and `results.csv`. Validation ran over 4 images / 17 instances. The recorded
GPU inference time was **1.7 ms per image**.

The mAP50 and mAP50-95 values are both zero because this smoke trains a random
initialization for one epoch. They are expected for this test and must not be
reported as an A1 baseline accuracy claim.

The raw logs, command arguments, and result CSV are under `evidence/`.
Checkpoint files are intentionally excluded: they are reproducible generated
artifacts, not source evidence needed to review this smoke.

## P0 pretrained baseline (completed)

P0 uses the same `yolo26n.pt` checkpoint for both cells and changes only the
native detection-head route: A selects one-to-many outputs followed by NMS;
B selects the checkpoint's one-to-one end-to-end outputs. No training or
checkpoint modification is performed.

| Cell | Route | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | one-to-many + NMS | 0.659 | 0.515 | 0.562 | 0.402 |
| B | one-to-one, NMS-free | 0.656 | 0.505 | 0.550 | 0.395 |

B is **0.7 AP points** below A on all 5,000 COCO val2017 images. On 16 fixed
validation images, A produced 77 detections and executed 16 real suppression
kernels; B produced 76 detections and executed zero suppression kernels.

Batch-1 total wall latency, excluding model loading and disk I/O:

| Device | A mean ± std | B mean ± std | B vs A |
| --- | ---: | ---: | ---: |
| RTX 4090 GPU 0 | 3.289 ± 0.089 ms | 3.284 ± 0.098 ms | -0.14% |
| CPU, one Torch/OpenCV thread | 20.246 ± 0.379 ms | 20.757 ± 0.569 ms | +2.52% |

The accuracy drop is within the two-AP target, and the runtime trace proves the
native B path is NMS-free. The speed result is neutral on GPU and slightly
slower on CPU, so P0 establishes correctness but does not claim a latency win.
See [`P0_PRETRAINED_REPORT.md`](P0_PRETRAINED_REPORT.md), the compact evidence
under [`p0_pretrained/`](p0_pretrained/), and the reproducer
[`scripts/a1/evaluate_p0_pretrained.py`](../../scripts/a1/evaluate_p0_pretrained.py).

## P1 pretrained 2×2 pilot (executed, acceptance failed)

The locked pretrained P1 pilot has now run A/B/C/D through preflight, 5-epoch
training, full COCO validation, NMS tracing, latency, routing, and ONNX audit.
B/D execute zero suppression kernels, so the NMS-free route is real. However,
all four full-COCO mAP50-95 values are effectively zero because the custom
A2C2f/A2C2fMoE blocks are mostly or partly randomly initialized and the short
low-LR fine-tuning budget cannot train them. A/B/C also have at least one ONNX
agreement failure. Therefore the planned pilot is complete, but P1 is **not
accepted or complete as a task milestone**.

See [`P1_PRETRAINED_PILOT_REPORT.md`](P1_PRETRAINED_PILOT_REPORT.md) and the
compact evidence under [`p1_pretrained/`](p1_pretrained/).

## Limits and next step

P1 pilot 的当前结果和原始证据索引见 [`P1_PILOT_REPORT.md`](P1_PILOT_REPORT.md)。

完整 COCO2017 已在 `scut-server` 验收完成（118,287 train / 5,000 val），就绪标记为
`/data/data2/TuJiajun/COCO2017/coco/READY.json`。从随机初始化开始的 r4 full-COCO
30-epoch 训练已于 2026-08-29 按计划暂停：C 的失败证据和 D 已完成的第一个 epoch、
checkpoint、日志均保留在 `/data/data2/TuJiajun/A1-smoke-r4/p1_full_r4/`，没有把它们
计为 P0/P1 结论。旧 r1/r2/r3 诊断产物也继续保留。
当前诊断、复现命令与实时日志入口见 [`P1_RECOVERY_20260828.md`](P1_RECOVERY_20260828.md)。

The Smoke admission gate and pretrained P0 are complete. P1 now has a complete
single-seed negative pilot, but still requires a warm-start strategy that
preserves useful pretrained behavior, followed by a valid-accuracy 2×2 rerun
and passing export agreement before making any MoE compatibility claim.
