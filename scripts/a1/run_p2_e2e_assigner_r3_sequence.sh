#!/usr/bin/env bash
set -euo pipefail

PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
FORMAL=/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28
PROJECT=$FORMAL/p2_e2e_assigner_r3
DATA=$ROOT/configs/a1/p1_pretrained/pilot_data/coco.yaml
SCRIPT=$ROOT/scripts/a1/run_p2_e2e_cls_gain_pilot.py
EVAL=$ROOT/scripts/a1/evaluate_p2_e2e_assigner_r3.py

run_one() {
  local model="$1" name="$2" topk2="$3" log="$4"
  cd "$ROOT"
  "$PY" "$SCRIPT" --model "$model" --data "$DATA" --project "$PROJECT" \
    --name "$name" --seed 260829 --epochs 5 --gain 1.0 --topk2 "$topk2" --device 1 \
    > "$FORMAL/$log" 2>&1
}

run_one "$FORMAL/initializers/seed260829/b_residual_factor_init.pt" \
  b_topk2_1_seed260829_5ep 1 p2_e2e_assigner_r3_b_topk2_1.log
run_one "$FORMAL/initializers/seed260829/b_residual_factor_init.pt" \
  b_topk2_2_seed260829_5ep 2 p2_e2e_assigner_r3_b_topk2_2.log
run_one "$FORMAL/initializers/seed260829/d_residual_factor_init.pt" \
  d_topk2_1_seed260829_5ep 1 p2_e2e_assigner_r3_d_topk2_1.log
run_one "$FORMAL/initializers/seed260829/d_residual_factor_init.pt" \
  d_topk2_2_seed260829_5ep 2 p2_e2e_assigner_r3_d_topk2_2.log

cd "$ROOT"
"$PY" "$EVAL" --project "$PROJECT" --data "$DATA" \
  --output "$PROJECT/eval_fixed_val512" --device cuda:1 \
  > "$FORMAL/p2_e2e_assigner_r3_eval.log" 2>&1
printf '%s\n' "P2-E r3 sequence completed: $(date -Is)" > "$FORMAL/p2_e2e_assigner_r3.sequence.done"
