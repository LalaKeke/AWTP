#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MAP_ROOT="${MAP_ROOT:-$(dirname "${HIPAD_ROOT}")}"
cd "${HIPAD_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_hzy}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RUNNER="${RUNNER:-${HIPAD_ROOT}/tools/pnn_pipeline/run_pnn_train_b2d.py}"

export PNN_GPUS="${PNN_GPUS:-0,1,2,3}"
export PNN_BATCH_SIZE="${PNN_BATCH_SIZE:-48}"
# Stage-1 runs on a fully materialized tensor dataset. Keep workers at 0 by
# default to avoid 4 ranks x 4 worker copies and expensive worker restarts.
export PNN_NUM_WORKERS="${PNN_STAGE1_NUM_WORKERS:-1}"
export PNN_PLANNER_MAX_ITERATIONS="${PNN_PLANNER_MAX_ITERATIONS:-10}"
export PNN_CONTROL_CKPT="${PNN_CONTROL_CKPT:-}"
export PNN_SAVE_DIR="${PNN_SAVE_DIR:-${HIPAD_ROOT}/outputs/pnn_stage1}"
export PNN_OLD_DATA="${PNN_OLD_DATA:-${HIPAD_ROOT}/data/pnn/time_aligned_v3/train_old.pt}"
export PNN_NEW_DATA="${PNN_NEW_DATA:-${HIPAD_ROOT}/data/pnn/static_v3/train_new_with_hipad_plan.pt}"
export PNN_COORD_CONVENTION="${PNN_COORD_CONVENTION:-pnn_xy}"
export PNN_REFERENCE_FORWARD_OFFSET="${PNN_REFERENCE_FORWARD_OFFSET:-0.0}"
export PNN_OUTPUT_FORWARD_OFFSET="${PNN_OUTPUT_FORWARD_OFFSET:-0.0}"
# Stage 1: fixed default weights, no WeightNet update, no trajectory-supervised WeightNet.
export PNN_CONTROL_WEIGHT_START_EPOCH="${PNN_CONTROL_WEIGHT_START_EPOCH:-1000000}"
export PNN_WEIGHT_UPDATE_START_EPOCH="${PNN_WEIGHT_UPDATE_START_EPOCH:-1000000}"
export PNN_LAMBDA_WEIGHT_TRAJ="${PNN_LAMBDA_WEIGHT_TRAJ:-0}"
export PNN_LAMBDA_GT_REFERENCE_LANE="${PNN_LAMBDA_GT_REFERENCE_LANE:-0.05}"
# These positive penalty coefficients must not be minimized jointly with the
# policy; doing so collapsed the control-rate regularization in the old run.
export PNN_TRAIN_SOFT_CONSTRAINT_LAMBDAS="${PNN_TRAIN_SOFT_CONSTRAINT_LAMBDAS:-0}"
export PNN_LAMBDA_ROLLOUT_COMFORT="${PNN_LAMBDA_ROLLOUT_COMFORT:-0.5}"
export PNN_COMFORT_ACC_THRESHOLD="${PNN_COMFORT_ACC_THRESHOLD:-2.40}"
export PNN_COMFORT_MIN_LON_ACCEL="${PNN_COMFORT_MIN_LON_ACCEL:--4.05}"
export PNN_COMFORT_LAT_ACCEL_THRESHOLD="${PNN_COMFORT_LAT_ACCEL_THRESHOLD:-4.89}"
export PNN_COMFORT_JERK_THRESHOLD="${PNN_COMFORT_JERK_THRESHOLD:-4.13}"
export PNN_COMFORT_YAW_RATE_THRESHOLD="${PNN_COMFORT_YAW_RATE_THRESHOLD:-0.95}"
export PNN_COMFORT_YAW_ACCEL_THRESHOLD="${PNN_COMFORT_YAW_ACCEL_THRESHOLD:-1.93}"
# Robust normalization must be identical during training and inference.
export PNN_STATS_QUANTILE_LOW="${PNN_STATS_QUANTILE_LOW:-0.005}"
export PNN_STATS_QUANTILE_HIGH="${PNN_STATS_QUANTILE_HIGH:-0.995}"
export PNN_CLAMP_NORMALIZED_INPUTS="${PNN_CLAMP_NORMALIZED_INPUTS:-1}"
export PNN_EVAL_EACH_EPOCH="${PNN_EVAL_EACH_EPOCH:-false}"
export PNN_DEFAULT_COST_WEIGHTS="${PNN_DEFAULT_COST_WEIGHTS:-1.0,2.0,0.6,2.0,3.0,2.0,1.2,10.0}"
export PNN_PROGRESS_OVERSHOOT_WEIGHT="${PNN_PROGRESS_OVERSHOOT_WEIGHT:-0.3}"
export PNN_EGO_TRACK_TIME_WEIGHTS="${PNN_EGO_TRACK_TIME_WEIGHTS:-1.8,2.5,1.8,1.2,0.8,0.5}"
export PNN_LAMBDA_ROUTE_SPEED_EXCESS="${PNN_LAMBDA_ROUTE_SPEED_EXCESS:-1.5}"
export PNN_ROUTE_SPEED_MARGIN="${PNN_ROUTE_SPEED_MARGIN:-0.5}"
export PNN_ROUTE_SPEED_BRAKE_TRIGGER_MARGIN="${PNN_ROUTE_SPEED_BRAKE_TRIGGER_MARGIN:-0.5}"
export PNN_ROUTE_SPEED_POS_ACCEL_THRESHOLD="${PNN_ROUTE_SPEED_POS_ACCEL_THRESHOLD:-0.2}"
export PNN_ROUTE_SPEED_BRAKE_WEIGHT="${PNN_ROUTE_SPEED_BRAKE_WEIGHT:-0.4}"
export PNN_LAMBDA_LANE_CLEARANCE="${PNN_LAMBDA_LANE_CLEARANCE:-0.5}"
export PNN_LANE_CLEARANCE_MARGIN="${PNN_LANE_CLEARANCE_MARGIN:-0.8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-50}"
LOG_DIR="${LOG_DIR:-${HIPAD_ROOT}/outputs/logs}"
mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}" "${PNN_SAVE_DIR}/checkpoints"

LAST_CKPT="${PNN_SAVE_DIR}/checkpoints/last.pth"
INITIAL_CKPT="${PNN_INITIAL_RESUME_CKPT:-}"

last_epoch() {
  local checkpoint="${LAST_CKPT}"
  if [[ ! -f "${checkpoint}" && -n "${INITIAL_CKPT}" && -f "${INITIAL_CKPT}" ]]; then
    checkpoint="${INITIAL_CKPT}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "-1"
    return
  fi
  CHECKPOINT="${checkpoint}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
ckpt = torch.load(os.environ["CHECKPOINT"], map_location="cpu")
print(int(ckpt.get("epoch", -1)))
PY
}

while true; do
  CURRENT_EPOCH="$(last_epoch)"
  NEXT_START=$((CURRENT_EPOCH + 1))
  if (( NEXT_START >= TOTAL_EPOCHS )); then
    echo "[control-pretrain] done: current_epoch=${CURRENT_EPOCH}, total_epochs=${TOTAL_EPOCHS}"
    break
  fi

  NEXT_END=$((NEXT_START + SEGMENT_EPOCHS))
  if (( NEXT_END > TOTAL_EPOCHS )); then
    NEXT_END="${TOTAL_EPOCHS}"
  fi

  export PNN_EPOCHS="${NEXT_END}"
  if [[ -f "${LAST_CKPT}" ]]; then
    export PNN_RESUME_CKPT="${LAST_CKPT}"
  elif [[ -n "${INITIAL_CKPT}" && -f "${INITIAL_CKPT}" ]]; then
    export PNN_RESUME_CKPT="${INITIAL_CKPT}"
  else
    unset PNN_RESUME_CKPT || true
  fi

  export MASTER_PORT="${MASTER_PORT:-$((29710 + NEXT_START % 80))}"
  LOG_FILE="${LOG_DIR}/pnn_stage1_control_pretrain_${NEXT_START}_to_$((NEXT_END - 1)).log"

  echo "[control-pretrain] segment ${NEXT_START}..$((NEXT_END - 1))"
  echo "[control-pretrain] PNN_SAVE_DIR=${PNN_SAVE_DIR}"
  echo "[control-pretrain] PNN_COORD_CONVENTION=${PNN_COORD_CONVENTION:-<infer-from-data>}"
  echo "[control-pretrain] PNN_OLD_DATA=${PNN_OLD_DATA}"
  echo "[control-pretrain] PNN_NEW_DATA=${PNN_NEW_DATA}"
  echo "[control-pretrain] PNN_SUPERVISION_DATA=${PNN_SUPERVISION_DATA:-<none>}"
  echo "[control-pretrain] PNN_RESUME_CKPT=${PNN_RESUME_CKPT:-<scratch>}"
  echo "[control-pretrain] PNN_TEACHER_CKPT=${PNN_TEACHER_CKPT:-<none>}"
  echo "[control-pretrain] normalization=q${PNN_STATS_QUANTILE_LOW}..q${PNN_STATS_QUANTILE_HIGH} clamp=${PNN_CLAMP_NORMALIZED_INPUTS}"
  echo "[control-pretrain] fixed_actor_safety=1 lambda_gt_reference_lane=${PNN_LAMBDA_GT_REFERENCE_LANE}"
  echo "[control-pretrain] ego_object_safety=${PNN_LAMBDA_EGO_OBJECT_SAFETY:-1.0} control_lr=${PNN_LR_CONTROL:-2e-5} override_resume_lr=${PNN_OVERRIDE_RESUME_LR:-0}"
  echo "[control-pretrain] metric_safety=margin:${PNN_METRIC_SAFETY_MARGIN:-1.5},topk:${PNN_METRIC_SAFETY_TOPK:-3},risk_gain:${PNN_RISK_SAFETY_GAIN:-0.0} teacher_trust=${PNN_LAMBDA_TEACHER_TRUST:-0.0}"
  echo "[control-pretrain] metric_lane=${PNN_LAMBDA_METRIC_LANE:-0.0}@${PNN_METRIC_LANE_MARGIN:-0.05} disable_pred_lane=${PNN_DISABLE_PREDICTED_LANE_LOSSES:-0} pnn_only_gain=${PNN_PNN_ONLY_SAFETY_GAIN:-0.0} shared_gain=${PNN_SHARED_SAFETY_GAIN:-0.0}"
  echo "[control-pretrain] comfort=${PNN_LAMBDA_ROLLOUT_COMFORT} thresholds=lon:[${PNN_COMFORT_MIN_LON_ACCEL},${PNN_COMFORT_ACC_THRESHOLD}],lat:${PNN_COMFORT_LAT_ACCEL_THRESHOLD},jerk:${PNN_COMFORT_JERK_THRESHOLD},yaw_rate:${PNN_COMFORT_YAW_RATE_THRESHOLD},yaw_accel:${PNN_COMFORT_YAW_ACCEL_THRESHOLD}"
  echo "[control-pretrain] tensorboard_logdir=${PNN_SAVE_DIR}/tb"
  echo "[control-pretrain] LOG_FILE=${LOG_FILE}"

  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${RUNNER}" > "${LOG_FILE}" 2>&1
done
