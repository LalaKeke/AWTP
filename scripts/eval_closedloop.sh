#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export WORK_DIR="${PROJECT_ROOT}"
export HIPAD_PNN_ROOT="${PROJECT_ROOT}"
export HIPAD_CKPT="${HIPAD_CKPT:-${PROJECT_ROOT}/checkpoints/hipad_stage2.pth}"
export PNN_CONTROL_CKPT="${PNN_CONTROL_CKPT:-${PROJECT_ROOT}/checkpoints/pnn_control.pth}"
export PNN_STATS_PATH="${PNN_STATS_PATH:-${PROJECT_ROOT}/checkpoints/pnn_stats.pt}"
export PNN_USE_WEIGHT_NET=0
export PNN_ROUTE_SOURCE=hipad_plan
export PNN_COORD_CONVENTION=pnn_xy

: "${CARLA_ROOT:?Set CARLA_ROOT to a CARLA 0.9.15 installation}"
export CARLA_ROOT

exec bash bench2drive/leaderboard/scripts/run_evaluation_multi_hipad_pnn.sh
