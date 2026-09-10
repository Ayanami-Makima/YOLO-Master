#!/usr/bin/env bash
set -euo pipefail

PY=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
ROOT=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r28-medium
FORMAL=/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_medium_r28
PROJECT=$FORMAL/p2_e2e_loss_r2
DATA=$ROOT/configs/a1/p1_pretrained/pilot_data/coco.yaml
SCRIPT=$ROOT/scripts/a1/run_p2_e2e_cls_gain_pilot.py
EVAL=$ROOT/scripts/a1/evaluate_p2_e2e_loss_r2.py

run_one() {
  local model="$1" name="$2" gain="$3" log="$4"
  cd "$ROOT"
  "$PY" "$SCRIPT" --model "$model" --data "$DATA" --project "$PROJECT" \
    --name "$name" --seed 260829 --epochs 5 --gain "$gain" --device 1 \
    > "$FORMAL/$log" 2>&1
}

# B gain=1.0 is started separately. Wait for its child process to exit so the
# same GPU is never oversubscribed after an SSH disconnect.
while ps -p 3385271 >/dev/null 2>&1; do sleep 30; done
run_one "$FORMAL/initializers/seed260829/b_residual_factor_init.pt" \
  b_gain1p2_seed260829_5ep 1.2 p2_e2e_loss_r2_b_gain1p2.log
run_one "$FORMAL/initializers/seed260829/d_residual_factor_init.pt" \
  d_gain1p0_seed260829_5ep 1.0 p2_e2e_loss_r2_d_gain1p0.log
run_one "$FORMAL/initializers/seed260829/d_residual_factor_init.pt" \
  d_gain1p2_seed260829_5ep 1.2 p2_e2e_loss_r2_d_gain1p2.log
$PY "$EVAL" --project "$PROJECT" --data "$DATA" \
  --output "$PROJECT/eval_fixed_val512" --device cuda:1 \
  > "$FORMAL/p2_e2e_loss_r2_eval.log" 2>&1
printf '%s\n' "P2-E r2 sequence completed: $(date -Is)" > "$FORMAL/p2_e2e_loss_r2.sequence.done"
