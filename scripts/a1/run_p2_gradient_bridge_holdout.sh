#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
OUT=/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_holdout_20260906
mkdir -p "$OUT"
exec 9>"$OUT/sequence.lock"
flock -n 9 || exit 1
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$ROOT"
for seed in 260829 260830 260831; do
  "$PY" scripts/a1/audit_p2_gradient_bridge.py \
    --root /data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28 \
    --data "$ROOT/configs/a1/p1_pretrained/pilot_data/coco.yaml" \
    --output "$OUT/evidence_seed${seed}.json" --seed "$seed" --offset 128 --images 32
done
"$PY" scripts/a1/summarize_p2_gradient_bridge.py --input-root "$OUT" --output "$OUT/summary.json"
