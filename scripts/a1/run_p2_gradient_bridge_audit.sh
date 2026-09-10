#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
OUT=/data/data2/TuJiajun/A1-smoke-r4/p2_gradient_bridge_r1
mkdir -p "$OUT"
exec 9>"$OUT/audit.lock"
flock -n 9 || exit 1
cd "$ROOT"
for seed in 260830 260831; do
  evidence="$OUT/evidence_seed${seed}.json"
  if [[ -e "$evidence" ]]; then
    "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "completed"' "$evidence"
    continue
  fi
  "$PY" scripts/a1/audit_p2_gradient_bridge.py \
    --root /data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28 \
    --data configs/a1/p1_pretrained/pilot_data/coco.yaml \
    --output "$evidence" --images 32 --seed "$seed"
done
printf '%s\n' 'All remaining seed gradient audits completed.'
