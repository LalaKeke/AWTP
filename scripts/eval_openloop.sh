#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

HIPAD_CKPT="${HIPAD_CKPT:-${PROJECT_ROOT}/checkpoints/hipad_stage2.pth}"
PNN_CONTROL_CKPT="${PNN_CONTROL_CKPT:-${PROJECT_ROOT}/checkpoints/pnn_control.pth}"
PNN_STATS_PATH="${PNN_STATS_PATH:-${PROJECT_ROOT}/checkpoints/pnn_stats.pt}"
GPUS="${GPUS:-1}"

for required in "${HIPAD_CKPT}" "${PNN_CONTROL_CKPT}" "${PNN_STATS_PATH}"; do
  [[ -s "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done

export HIPAD_PNN_ROOT="${PROJECT_ROOT}"
export PNN_OPENLOOP_CONTROL_CKPT="${PNN_CONTROL_CKPT}"
export PNN_OPENLOOP_STATS="${PNN_STATS_PATH}"
export PNN_OPENLOOP_USE_WEIGHT_NET=0
export PNN_COORD_CONVENTION="${PNN_COORD_CONVENTION:-pnn_xy}"
export PNN_STATS_QUANTILE_LOW="${PNN_STATS_QUANTILE_LOW:-0.005}"
export PNN_STATS_QUANTILE_HIGH="${PNN_STATS_QUANTILE_HIGH:-0.995}"
export PNN_CLAMP_NORMALIZED_INPUTS="${PNN_CLAMP_NORMALIZED_INPUTS:-1}"

exec bash tools/dist_test_hipad_pnn_openloop.sh \
  projects/configs/hipad_b2d_stage2.py "${HIPAD_CKPT}" "${GPUS}" \
  --pnn-control-ckpt "${PNN_CONTROL_CKPT}" \
  --pnn-stats "${PNN_STATS_PATH}" \
  --pnn-no-weight --pnn-planning-only --eval bbox \
  --pnn-max-batches "${PNN_MAX_BATCHES:-0}"
