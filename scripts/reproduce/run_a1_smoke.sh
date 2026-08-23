#!/usr/bin/env bash
# Reproduce the A1 admission smoke on scut-server.
set -euo pipefail

EXPECTED_REF="acce839c7e895d6b179de7f7093fa879e237cc7b"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${A1_RUN_ROOT:-/data/data2/TuJiajun/A1-smoke}"
PYTHON_BIN="${A1_PYTHON_BIN:-${RUN_ROOT}/.venv/bin/python}"
YOLO_BIN="${A1_YOLO_BIN:-${RUN_ROOT}/.venv/bin/yolo}"
LOG_DIR="${RUN_ROOT}/logs"
RUN_NAME="${A1_RUN_NAME:-detect_e2e_coco8_1ep}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${YOLO_BIN}" ]]; then
  echo "A1 virtual environment is unavailable. Set A1_RUN_ROOT or A1_PYTHON_BIN/A1_YOLO_BIN." >&2
  exit 2
fi

actual_ref="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ "${actual_ref}" != "${EXPECTED_REF}" ]]; then
  echo "Refusing to run: expected ${EXPECTED_REF}, got ${actual_ref}." >&2
  exit 3
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

"${YOLO_BIN}" checks 2>&1 | tee "${LOG_DIR}/yolo_checks.log"
"${PYTHON_BIN}" agent/scripts/validate_yolo_master_skill.py \
  --suite quick --pretty --summary-only 2>&1 | tee "${LOG_DIR}/agent_quick.log"
"${YOLO_BIN}" train \
  model=ultralytics/cfg/models/26/yolo26.yaml \
  data=coco8.yaml epochs=1 imgsz=64 batch=1 device=0 workers=0 plots=False \
  project="${RUN_ROOT}/runs" name="${RUN_NAME}" 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
