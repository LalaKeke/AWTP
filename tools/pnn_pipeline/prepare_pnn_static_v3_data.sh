#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_NEW="${PNN_BASE_NEW_DATA:-${HIPAD_ROOT}/data/pnn/static_v1/train_new.pt}"
OUT_DIR="${PNN_STATIC_V3_DATA_DIR:-${HIPAD_ROOT}/data/pnn/static_v3}"
OUTPUT="${OUT_DIR}/train_new_with_hipad_plan.pt"

if [[ ! -s "${BASE_NEW}" ]]; then
  echo "[static-v3-data] missing base tensor: ${BASE_NEW}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/augment_static_dataset_with_hipad_plan.py" \
  --base-new-data "${BASE_NEW}" \
  --output "${OUTPUT}" \
  --chunk-dir "${PNN_CHUNK_DIR:-${HIPAD_ROOT}/outputs/inference_chunks}" \
  --workers "${PNN_CONVERT_WORKERS:-4}"

echo "[static-v3-data] output=${OUTPUT}"
