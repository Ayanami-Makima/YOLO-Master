#!/usr/bin/env bash
set -euo pipefail

REPO=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r24-full
PYTHON=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
PROTOCOL="$REPO/configs/a1/p1_factorial_full_r24/protocol.json"
RUN_ROOT=/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_full_r24
AUDITS="$RUN_ROOT/audits"
GPU0_UUID=GPU-601f13ba-c60e-08b2-0bc4-043afb12ac6c
GPU1_UUID=GPU-4e204b8b-4c8c-ad2a-e693-0a26428bd214

cd "$REPO"

timestamp() {
    date --iso-8601=seconds
}

wait_for_both_gpus() {
    local stage="$1"
    while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
        | grep -E -q "^(${GPU0_UUID}|${GPU1_UUID})$"; do
        echo "$(timestamp) waiting_for_idle_gpus stage=$stage"
        sleep 60
    done
    echo "$(timestamp) idle_gpus_confirmed stage=$stage"
}

echo "$(timestamp) r24_pipeline_started"

wait_for_both_gpus preflight
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r24.py \
    --protocol "$PROTOCOL" --stage preflight
"$PYTHON" scripts/a1/audit_p1_checkpoints_r24.py \
    --protocol "$PROTOCOL" --stage preflight \
    --output "$AUDITS/preflight_checkpoints.json"

wait_for_both_gpus routing_probe
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r24.py \
    --protocol "$PROTOCOL" --stage routing_probe
"$PYTHON" scripts/a1/audit_p1_checkpoints_r24.py \
    --protocol "$PROTOCOL" --stage routing_probe \
    --output "$AUDITS/routing_probe_checkpoints.json"

wait_for_both_gpus routing_audit
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/a1/audit_p1_routing_r24.py \
    --protocol "$PROTOCOL" \
    --preflight-audit "$AUDITS/preflight_checkpoints.json" \
    --probe-audit "$AUDITS/routing_probe_checkpoints.json" \
    --device 0
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/a1/audit_p1_residual_activity_r24.py \
    --protocol "$PROTOCOL" \
    --probe-checkpoint-audit "$AUDITS/routing_probe_checkpoints.json" \
    --routing-audit "$AUDITS/routing/hard_top2_512.json" \
    --output "$AUDITS/residual_activity/routing_probe_512.json" \
    --device 0

"$PYTHON" scripts/a1/audit_p1_formal_admission_r24.py \
    --protocol "$PROTOCOL" \
    --preflight-audit "$AUDITS/preflight_checkpoints.json" \
    --probe-audit "$AUDITS/routing_probe_checkpoints.json" \
    --routing-audit "$AUDITS/routing/hard_top2_512.json" \
    --residual-audit "$AUDITS/residual_activity/routing_probe_512.json" \
    --migration-audit "$AUDITS/migration_r23_to_r24.json" \
    --output "$AUDITS/formal_admission.json"

wait_for_both_gpus formal
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r24.py \
    --protocol "$PROTOCOL" --stage formal
"$PYTHON" scripts/a1/audit_p1_checkpoints_r24.py \
    --protocol "$PROTOCOL" --stage formal \
    --output "$AUDITS/formal_checkpoints.json"

echo "$(timestamp) r24_pipeline_completed"
