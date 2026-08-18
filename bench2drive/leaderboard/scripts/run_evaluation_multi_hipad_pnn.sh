#!/bin/bash
set -euo pipefail

BASE_PORT=${BASE_PORT:-32000}
BASE_TM_PORT=${BASE_TM_PORT:-52000}
IS_BENCH2DRIVE=True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=${WORK_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
MAP_ROOT=${MAP_ROOT:-$(dirname "${WORK_DIR}")}
CARLA_ROOT=${CARLA_ROOT:-${WORK_DIR}/carla}
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${WORK_DIR}"

CONFIG_NAME=${CONFIG_NAME:-hipad_b2d_pnn_control_epoch0150_0725}
HIPAD_CONFIG=${HIPAD_CONFIG:-${WORK_DIR}/projects/configs/hipad_b2d_stage2.py}
HIPAD_CKPT=${HIPAD_CKPT:-${WORK_DIR}/checkpoints/hipad_stage2.pth}

TEAM_AGENT=${TEAM_AGENT:-bench2drive/leaderboard/team_code/hipad_pnn_b2d_agent.py}
TEAM_CONFIG=${TEAM_CONFIG:-${HIPAD_CONFIG}+${HIPAD_CKPT}+${CONFIG_NAME}}

PLANNER_TYPE=traj
BASE_ROUTES=${BASE_ROUTES:-bench2drive/leaderboard/data/splits16/bench2drive220}

SAVE_PATH=${SAVE_PATH:-evaluation/${CONFIG_NAME}}
BASE_CHECKPOINT_ENDPOINT=${BASE_CHECKPOINT_ENDPOINT:-evaluation/${CONFIG_NAME}/${CONFIG_NAME}}

export MAP_ROOT
export CARLA_ROOT
export PNN_ROUTE_SOURCE=${PNN_ROUTE_SOURCE:-hipad_plan}
if [ "${PNN_ROUTE_SOURCE}" = "navigation" ]; then
    : "${PNN_STATS_PATH:?navigation mode requires an explicit PNN_STATS_PATH}"
    : "${PNN_CONTROL_CKPT:?navigation mode requires an explicit PNN_CONTROL_CKPT}"
    export PNN_STATS_PATH PNN_CONTROL_CKPT
else
    export PNN_STATS_PATH=${PNN_STATS_PATH:-${WORK_DIR}/checkpoints/pnn_stats.pt}
    export PNN_CONTROL_CKPT=${PNN_CONTROL_CKPT:-${WORK_DIR}/checkpoints/pnn_control.pth}
fi
export PNN_WEIGHT_CKPT=${PNN_WEIGHT_CKPT:-}
export PNN_USE_WEIGHT_NET=${PNN_USE_WEIGHT_NET:-0}
export PNN_HIPAD_PLAN_KEY=${PNN_HIPAD_PLAN_KEY:-plan_temp_2hz}
export PNN_NAV_MIN_SPEED=${PNN_NAV_MIN_SPEED:-1.0}
export PNN_NAV_MAX_SPEED=${PNN_NAV_MAX_SPEED:-15.0}
export PNN_NAV_DISTANCE_SCALE=${PNN_NAV_DISTANCE_SCALES:-${PNN_NAV_DISTANCE_SCALE:-1.0}}
export PNN_NAV_INTERPOLATION=${PNN_NAV_INTERPOLATION:-spline}
export PNN_PID_WAYPOINT_TIME=${PNN_PID_WAYPOINT_TIME:-0.5}
export PNN_OUTPUT_FORWARD_OFFSET=${PNN_OUTPUT_FORWARD_OFFSET:-0.0}
export PNN_REFERENCE_FORWARD_OFFSET=${PNN_REFERENCE_FORWARD_OFFSET:-0.0}
export PNN_COORD_CONVENTION=${PNN_COORD_CONVENTION:-pnn_xy}
export PNN_STATS_QUANTILE_LOW=${PNN_STATS_QUANTILE_LOW:-0.005}
export PNN_STATS_QUANTILE_HIGH=${PNN_STATS_QUANTILE_HIGH:-0.995}
export PNN_CLAMP_NORMALIZED_INPUTS=${PNN_CLAMP_NORMALIZED_INPUTS:-1}
export PNN_VISUALIZE=${PNN_VISUALIZE:-0}
mkdir -p "${SAVE_PATH}"

echo -e "************** HiP-AD + PNN Bench2Drive closed-loop **************"
echo -e "TEAM_AGENT: ${TEAM_AGENT}"
echo -e "TEAM_CONFIG: ${TEAM_CONFIG}"
echo -e "PNN_CONTROL_CKPT: ${PNN_CONTROL_CKPT}"
echo -e "PNN_USE_WEIGHT_NET: ${PNN_USE_WEIGHT_NET} (0 means trajectory inference uses ControlNet only)"
if [ "${PNN_USE_WEIGHT_NET}" != "0" ]; then
    echo -e "PNN_WEIGHT_CKPT: ${PNN_WEIGHT_CKPT}"
fi
echo -e "PNN_STATS_PATH: ${PNN_STATS_PATH}"
echo -e "PNN_HIPAD_PLAN_KEY: ${PNN_HIPAD_PLAN_KEY}"
echo -e "PNN_ROUTE_SOURCE: ${PNN_ROUTE_SOURCE}"
echo -e "PNN_NAV_SPEED_RANGE: ${PNN_NAV_MIN_SPEED}-${PNN_NAV_MAX_SPEED} m/s"
echo -e "PNN_NAV_DISTANCE_SCALE: ${PNN_NAV_DISTANCE_SCALE}"
echo -e "PNN_NAV_INTERPOLATION: ${PNN_NAV_INTERPOLATION}"
echo -e "PNN_PID_WAYPOINT_TIME: ${PNN_PID_WAYPOINT_TIME}"
echo -e "PNN_OUTPUT_FORWARD_OFFSET: ${PNN_OUTPUT_FORWARD_OFFSET}"
echo -e "PNN_REFERENCE_FORWARD_OFFSET: ${PNN_REFERENCE_FORWARD_OFFSET}"
echo -e "PNN_COORD_CONVENTION: ${PNN_COORD_CONVENTION}"

if [ "${PNN_ROUTE_SOURCE}" = "navigation" ]; then
    echo -e "PNN navigation mode requires a ControlNet trained with pnn_step2_navigation data."
fi

# GPU_RANK is a GPU index, not the number of GPUs.
GPU_RANK_LIST_STR=${GPU_RANK_LIST:-0 1 2 3}
TASK_LIST_STR=${TASK_LIST:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}
read -r -a GPU_RANKS <<< "${GPU_RANK_LIST_STR}"
read -r -a TASK_IDS <<< "${TASK_LIST_STR}"
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-4}
USE_WATCHDOG=${USE_WATCHDOG:-1}
export CONFIG_NAME TEAM_AGENT TEAM_CONFIG PLANNER_TYPE BASE_ROUTES SAVE_PATH PYTHON_BIN
export BASE_CHECKPOINT_ENDPOINT BASE_PORT BASE_TM_PORT
export TASK_LIST="${TASK_LIST_STR}"
export GPU_RANK_LIST="${GPU_RANK_LIST_STR}"

for required_path in \
    "${HIPAD_CONFIG}" \
    "${HIPAD_CKPT}" \
    "${PNN_STATS_PATH}" \
    "${PNN_CONTROL_CKPT}" \
    "${WORK_DIR}/hipad_pnn_adapter.py" \
    "${CARLA_ROOT}/CarlaUE4.sh" \
    "${TEAM_AGENT}"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ "${PNN_USE_WEIGHT_NET}" != "0" ] && [ ! -e "${PNN_WEIGHT_CKPT}" ]; then
    echo "ERROR: PNN weight checkpoint does not exist: ${PNN_WEIGHT_CKPT}" >&2
    exit 1
fi

for split in "${TASK_IDS[@]}"; do
    if [ ! -s "${BASE_ROUTES}_${split}.xml" ]; then
        echo "ERROR: route split does not exist: ${BASE_ROUTES}_${split}.xml" >&2
        exit 1
    fi
done

echo -e "TASK_LIST: ${TASK_IDS[*]}"
echo -e "GPU_RANK_LIST: ${GPU_RANKS[*]}"
echo -e "MAX_PARALLEL_TASKS: ${MAX_PARALLEL_TASKS}"
echo -e "USE_WATCHDOG: ${USE_WATCHDOG}"
echo -e "\033[36m***********************************************************************************\033[0m"

if [ "${PNN_PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "[preflight] all closed-loop paths and route splits are valid"
    exit 0
fi

if [ "${USE_WATCHDOG}" = "1" ]; then
    # A watchdog owns GPU scheduling, CARLA restart, per-route retries and resume.
    exec bash bench2drive/leaderboard/scripts/watchdog_hipad_eval.sh
fi

length=${#TASK_IDS[@]}
for ((i=0; i<length; i++ )); do
    PORT=$((BASE_PORT + i * 200))
    TM_PORT=$((BASE_TM_PORT + i * 200))
    ROUTES="${BASE_ROUTES}_${TASK_IDS[$i]}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_IDS[$i]}.json"
    GPU_RANK=${GPU_RANKS[$((i % ${#GPU_RANKS[@]}))]}
    LOG_FILE="${BASE_CHECKPOINT_ENDPOINT}_${TASK_IDS[$i]}.log"

    echo -e "TASK_ID: $i"
    echo -e "PORT: $PORT"
    echo -e "TM_PORT: $TM_PORT"
    echo -e "ROUTES: $ROUTES"
    echo -e "CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT"
    echo -e "GPU_RANK: $GPU_RANK"
    echo -e "LOG_FILE: $LOG_FILE"
    echo -e "\033[36m***********************************************************************************\033[0m"

    bash -e bench2drive/leaderboard/scripts/run_evaluation.sh \
        "$PORT" "$TM_PORT" "$IS_BENCH2DRIVE" "$ROUTES" "$TEAM_AGENT" "$TEAM_CONFIG" \
        "$CHECKPOINT_ENDPOINT" "$SAVE_PATH" "$PLANNER_TYPE" "$GPU_RANK" \
        > "$LOG_FILE" 2>&1 &
    sleep 5

    if [ $(((i + 1) % MAX_PARALLEL_TASKS)) -eq 0 ]; then
        wait
    fi
done
wait
