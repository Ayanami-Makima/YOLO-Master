#!/usr/bin/env bash
set -euo pipefail

PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
CONFIG=$ROOT/configs/a1/p2_seg_r1
RUN_ROOT=/data/data2/TuJiajun/A1-smoke-r4/p2_seg_r1
LOG_ROOT=$RUN_ROOT/logs
mkdir -p "$LOG_ROOT"

for cell in a b d; do
  request="$CONFIG/${cell}_request.json"
  log="$LOG_ROOT/${cell}_seg_pilot_seed260829_5ep.log"
  if [[ -f "$RUN_ROOT/train/${cell}_seg_pilot_seed260829_5ep/weights/last.pt" && -f "$RUN_ROOT/train/${cell}_seg_pilot_seed260829_5ep/results.csv" ]]; then
    echo "[$(date -Is)] skip existing $cell" | tee -a "$LOG_ROOT/sequence.log"
    continue
  fi
  echo "[$(date -Is)] start $cell" | tee -a "$LOG_ROOT/sequence.log"
  "$PY" "$ROOT/scripts/a1/run_p1_bn_frozen.py" --request "$request" 2>&1 | tee "$log"
  echo "[$(date -Is)] complete $cell" | tee -a "$LOG_ROOT/sequence.log"
done
echo "[$(date -Is)] sequence complete" | tee -a "$LOG_ROOT/sequence.log"
