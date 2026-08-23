# A1: MoE × End-to-End NMS-free Smoke

This directory records the admission smoke for the Tencent Rhino-Bird A1 task:
**MoE × End-to-End: whether the existing one-to-one path can form an NMS-free
loop**.

## Scope and baseline

This is an admission smoke, not a P0 accuracy result or a research conclusion.
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

## Limits and next step

This completes only the Smoke/admission gate. P0 still requires a COCO-mini or
COCO-val run, mAP50-95, GPU and CPU batch=1 latency with standard deviation,
and a documented reproducibility package. The subsequent A1 work must also
perform the MoE on/off × NMS on/off 2×2 ablation before making any NMS-free or
MoE compatibility claim.
