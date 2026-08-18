#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${PNN_STATIC_V31_DATA_DIR:-${HIPAD_ROOT}/data/pnn/static_v31}"
OUTPUT="${PNN_SOLID_LANE_SUPERVISION:-${OUT_DIR}/solid_lane_supervision.pt}"
CANDIDATE_MASK="${PNN_SOLID_LANE_CANDIDATE_MASK:-${OUT_DIR}/lane_candidate_mask_15m.pt}"
OLD_DATA="${PNN_OLD_DATA:-${HIPAD_ROOT}/data/pnn/static_v1/train_old.pt}"
NEW_DATA="${PNN_NEW_DATA:-${HIPAD_ROOT}/data/pnn/static_v3/train_new_with_hipad_plan.pt}"

mkdir -p "${OUT_DIR}"

if [[ -s "${CANDIDATE_MASK}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "[static-v3.1-data] reusing ${CANDIDATE_MASK}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_static_lane_candidate_mask.py" \
    --old-data "${OLD_DATA}" \
    --new-data "${NEW_DATA}" \
    --output "${CANDIDATE_MASK}" \
    --distance "${PNN_SOLID_LANE_CANDIDATE_DISTANCE:-15.0}"
fi

if [[ -s "${OUTPUT}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "[static-v3.1-data] reusing ${OUTPUT}"
else
  cd "${HIPAD_ROOT}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_static_solid_lane_supervision.py" \
    --output "${OUTPUT}" \
    --workers "${PNN_CONVERT_WORKERS:-12}" \
    --candidate-mask "${CANDIDATE_MASK}"
fi

echo "[static-v3.1-data] candidate_mask=${CANDIDATE_MASK}"
echo "[static-v3.1-data] solid_lane=${OUTPUT}"
