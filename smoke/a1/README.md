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

## Historical invalid P1 pilot (r9)

The early pretrained 2×2 pilot directly rebuilt the official C3k2 layers as
custom A2C2f/A2C2fMoE blocks. That left a material part of the checkpoint
randomly initialized, so its near-zero mAP results are retained only as a
diagnostic record, not as a P1 conclusion. See
[`P1_PRETRAINED_PILOT_REPORT.md`](P1_PRETRAINED_PILOT_REPORT.md) and the
compact evidence under [`p1_pretrained/`](p1_pretrained/).

## P1 r28 warm-start 2×2 factorial (completed)

The valid P1 experiment uses `C3k2ResidualFactor`: the official pretrained
C3k2 base at layers 4/6/8 is retained and frozen, while the residual factor
and detection head are trained. It completed A/B/C/D × seeds 260829/260830/
260831 on 20,000 COCO train images and 5,000 val images for 15 epochs. Mean
mAP50-95 is A 0.40200, B 0.39488, C 0.40222 and D 0.39465; the MoE main effect
is -0.00001 and the End-to-End main effect is -0.00734.

All 12 runtime traces distinguish NMS (A/C) from the native NMS-free
score-filter route (B/D), and all exported ONNX graphs contain zero
`NonMaxSuppression` nodes. Strict raw-output ONNX parity remains partial: A/C
pass 24/24 inputs and B/D pass 20/24; the four B/D differences are
very-low-score TopK-tail members. The deployment-relevant `conf=0.001`
semantic comparison passes B/D 24/24, but does not replace that strict
limitation.

Controlled low-contention CPU timing records B−A = -18.900 ms (-0.37%) and
D−C = -16.183 ms (-0.22%): measurable but too small to claim a material
speedup. See [`P1_FACTORIAL_MEDIUM_R28_FINAL_AUDIT.md`](P1_FACTORIAL_MEDIUM_R28_FINAL_AUDIT.md),
[`P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md`](P1_FACTORIAL_MEDIUM_R28_CLOSURE_REPORT.md),
and the [closure evidence](p1_factorial_medium_r28/closure_r1/).

The smoke admission gate, P0, and the valid warm-start P1 r28 factorial
experiment are complete. The r9 and early pretrained pilots remain preserved
as invalid or negative historical evidence. P1 r28 supports a constrained
accuracy conclusion, not a general claim of MoE benefit or material latency
acceleration; any full-COCO long-schedule follow-up must start as an
independent protocol.
