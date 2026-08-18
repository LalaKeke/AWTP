#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export HIPAD_PNN_ROOT="${HIPAD_PNN_ROOT:-${PROJECT_ROOT}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export PNN_OLD_DATA="${PNN_OLD_DATA:-${PROJECT_ROOT}/data/pnn/time_aligned_v3/train_old.pt}"
export PNN_NEW_DATA="${PNN_NEW_DATA:-${PROJECT_ROOT}/data/pnn/static_v3/train_new_with_hipad_plan.pt}"
export PNN_SUPERVISION_DATA="${PNN_SUPERVISION_DATA:-${PROJECT_ROOT}/data/pnn/static_v31/solid_lane_supervision.pt}"
export PNN_SAVE_DIR="${PNN_SAVE_DIR:-${PROJECT_ROOT}/outputs/pnn_physical_joint_scratch_v2}"

exec bash "${PROJECT_ROOT}/tools/pnn_pipeline/run_pnn_physical_joint_scratch_v2_train.sh"
