#!/usr/bin/env bash
set -euo pipefail

REPO=/data/data2/TuJiajun/A1-smoke-r4/YOLO-Master-r25-full
PYTHON=/data/data2/TuJiajun/A1-smoke-r4/.venv/bin/python
PROTOCOL="$REPO/configs/a1/p1_factorial_full_r25/protocol.json"
RUN_ROOT=/data/data2/TuJiajun/A1-smoke-r4/p1_factorial_full_r25
AUDITS="$RUN_ROOT/audits"
MIN_FREE_MEMORY_MIB=12000

cd "$REPO"

timestamp() {
    date --iso-8601=seconds
}

wait_for_gpu_memory() {
    local stage="$1"
    local free0 free1
    while true; do
        free0="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        free1="$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if (( free0 >= MIN_FREE_MEMORY_MIB && free1 >= MIN_FREE_MEMORY_MIB )); then
            break
        fi
        echo "$(timestamp) waiting_for_gpu_memory stage=$stage gpu0_free_mib=$free0 gpu1_free_mib=$free1"
        sleep 60
    done
    echo "$(timestamp) shared_gpu_memory_confirmed stage=$stage gpu0_free_mib=$free0 gpu1_free_mib=$free1"
}

echo "$(timestamp) r25_pipeline_started"

wait_for_gpu_memory preflight
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r25.py \
    --protocol "$PROTOCOL" --stage preflight
"$PYTHON" scripts/a1/audit_p1_checkpoints_r25.py \
    --protocol "$PROTOCOL" --stage preflight \
    --output "$AUDITS/preflight_checkpoints.json"

wait_for_gpu_memory routing_probe
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r25.py \
    --protocol "$PROTOCOL" --stage routing_probe
"$PYTHON" scripts/a1/audit_p1_checkpoints_r25.py \
    --protocol "$PROTOCOL" --stage routing_probe \
    --output "$AUDITS/routing_probe_checkpoints.json"

wait_for_gpu_memory routing_audit
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/a1/audit_p1_routing_r25.py \
    --protocol "$PROTOCOL" \
    --preflight-audit "$AUDITS/preflight_checkpoints.json" \
    --probe-audit "$AUDITS/routing_probe_checkpoints.json" \
    --device 0
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/a1/audit_p1_residual_activity_r25.py \
    --protocol "$PROTOCOL" \
    --probe-checkpoint-audit "$AUDITS/routing_probe_checkpoints.json" \
    --routing-audit "$AUDITS/routing/hard_top2_512.json" \
    --output "$AUDITS/residual_activity/routing_probe_512.json" \
    --device 0

"$PYTHON" scripts/a1/audit_p1_formal_admission_r25.py \
    --protocol "$PROTOCOL" \
    --preflight-audit "$AUDITS/preflight_checkpoints.json" \
    --probe-audit "$AUDITS/routing_probe_checkpoints.json" \
    --routing-audit "$AUDITS/routing/hard_top2_512.json" \
    --residual-audit "$AUDITS/residual_activity/routing_probe_512.json" \
    --migration-audit "$AUDITS/migration_r23_to_r25.json" \
    --output "$AUDITS/formal_admission.json"

wait_for_gpu_memory formal
"$PYTHON" scripts/a1/run_p1_factorial_multiseed_2gpu_r25.py \
    --protocol "$PROTOCOL" --stage formal
"$PYTHON" scripts/a1/audit_p1_checkpoints_r25.py \
    --protocol "$PROTOCOL" --stage formal \
    --output "$AUDITS/formal_checkpoints.json"

echo "$(timestamp) r25_pipeline_completed"
