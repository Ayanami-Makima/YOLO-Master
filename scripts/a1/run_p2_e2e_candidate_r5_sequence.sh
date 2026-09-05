#!/usr/bin/env bash
set -euo pipefail

PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
CONFIG=$ROOT/configs/a1/p2_e2e_candidate_r5
RUN_ROOT=/data/data2/TuJiajun/A1-smoke-r4/p2_e2e_candidate_r5
LOG_ROOT=$RUN_ROOT/logs
mkdir -p "$LOG_ROOT"

run_one() {
  local name="$1" topk="$2"
  local request="$CONFIG/requests/${name}.json"
  local run="$RUN_ROOT/train/${name}_seed260829_5ep"
  local log="$LOG_ROOT/${name}.log"
  if [[ -f "$run/weights/last.pt" && -f "$run/results.csv" ]]; then
    echo "[$(date -Is)] skip existing $name" | tee -a "$LOG_ROOT/sequence.log"
    return 0
  fi
  echo "[$(date -Is)] start $name topk=$topk" | tee -a "$LOG_ROOT/sequence.log"
  cd "$ROOT"
  A1_E2E_O2O_TAL_TOPK="$topk" A1_E2E_O2O_TAL_TOPK2=1 A1_TAL_ASSIGNMENT_AUDIT=1 \
    "$PY" "$ROOT/scripts/a1/run_p1_bn_frozen.py" --request "$request" 2>&1 | tee "$log"
  echo "[$(date -Is)] complete $name" | tee -a "$LOG_ROOT/sequence.log"
}

run_one b_control 7
run_one b_candidate10 10
run_one d_control 7
run_one d_candidate10 10
echo "[$(date -Is)] sequence complete" | tee -a "$LOG_ROOT/sequence.log"
