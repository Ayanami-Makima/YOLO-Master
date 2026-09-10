#!/usr/bin/env bash
set -euo pipefail

REPO="/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium"
PYTHON="/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python"
RUN_ROOT="/data/data2/TuJiajun/A1-smoke-r4/p2_efficiency_screen_r1"
CONFIG_ROOT="$REPO/configs/a1/p2_efficiency_screen_r1"
RUNNER="$REPO/scripts/a1/run_p2_efficiency_screen_pilot.py"
LOG_ROOT="$RUN_ROOT/logs"
cd "$REPO"
mkdir -p "$LOG_ROOT"

variants=(
  late_dense_eq_top1
  late_dense_eq_top2
  late_moe_2e_top1
  late_moe_2e_top2
  late_moe_4e_top1
  late_moe_4e_top2
)

for variant in "${variants[@]}"; do
  request="$CONFIG_ROOT/${variant}_request.json"
  run_dir="$RUN_ROOT/pilot/${variant}_seed260829_1ep"
  done_marker="$run_dir/.screen.done"
  log="$LOG_ROOT/${variant}.log"
  if [ -f "$done_marker" ]; then
    echo "[$variant] already complete"
    continue
  fi

  resume_args=()
  if [ -s "$run_dir/weights/last.pt" ]; then
    resume_args=(--resume-from "$run_dir/weights/last.pt")
    echo "[$variant] resuming from $run_dir/weights/last.pt"
  else
    echo "[$variant] starting from initializer"
  fi
  "$PYTHON" "$RUNNER" --request "$request" "${resume_args[@]}" > "$log" 2>&1
  touch "$done_marker"
  echo "[$variant] complete"
done

touch "$RUN_ROOT/.sequence.done"
echo "efficiency screen sequence complete"
