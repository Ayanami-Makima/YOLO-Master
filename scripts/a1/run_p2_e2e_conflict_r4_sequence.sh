#!/usr/bin/env bash
set -euo pipefail

PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
FORMAL=/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28
PROJECT=$FORMAL/p2_e2e_conflict_r4
DATA=$ROOT/configs/a1/p1_pretrained/pilot_data/coco.yaml
SCRIPT=$ROOT/scripts/a1/run_p2_e2e_cls_gain_pilot.py
EVAL=$ROOT/scripts/a1/evaluate_p2_e2e_conflict_r4.py
B_INIT=$FORMAL/initializers/seed260829/b_residual_factor_init.pt
D_INIT=$FORMAL/initializers/seed260829/d_residual_factor_init.pt

run_one() {
  local model="$1" name="$2" metric="$3" log="$4"
  if [[ -f "$PROJECT/$name/p2_status.json" ]]; then
    echo "[$name] existing run directory; refusing implicit overwrite" >&2
    return 1
  fi
  cd "$ROOT"
  "$PY" "$SCRIPT" --model "$model" --data "$DATA" --project "$PROJECT" \
    --name "$name" --seed 260829 --epochs 5 --gain 1.0 --topk2 1 \
    --conflict-metric "$metric" --device 1 > "$FORMAL/$log" 2>&1
  printf '%s\n' "[$name] complete"
}

run_one "$B_INIT" b_conflict_overlap_seed260829_5ep overlap p2_e2e_conflict_r4_b_overlap.log
run_one "$B_INIT" b_conflict_align_seed260829_5ep align p2_e2e_conflict_r4_b_align.log
run_one "$D_INIT" d_conflict_overlap_seed260829_5ep overlap p2_e2e_conflict_r4_d_overlap.log
run_one "$D_INIT" d_conflict_align_seed260829_5ep align p2_e2e_conflict_r4_d_align.log

cd "$ROOT"
"$PY" "$EVAL" --project "$PROJECT" --data "$DATA" \
  --output "$PROJECT/eval_fixed_val512" --device cuda:1 > "$FORMAL/p2_e2e_conflict_r4_eval.log" 2>&1
printf '%s\n' "P2-E r4 sequence completed: $(date -Is)" > "$FORMAL/p2_e2e_conflict_r4.sequence.done"
