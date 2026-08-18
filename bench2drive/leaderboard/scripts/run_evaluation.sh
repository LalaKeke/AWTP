#!/bin/bash
# Must set CARLA_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export MAP_ROOT="${MAP_ROOT:-$(dirname "${WORK_DIR}")}"
export WORK_DIR="${WORK_DIR:-${MAP_ROOT}/HiP-AD}"
export CARLA_ROOT="${CARLA_ROOT:-${MAP_ROOT}/carla}"
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# Match the released HiP-AD evaluator. Low quality changes the camera domain.
export CARLA_SERVER_ARGS="${CARLA_SERVER_ARGS:--RenderOffScreen -nosound -quality-level=Epic}"
export CARLA_REQUIRE_EXACT_PORT="${CARLA_REQUIRE_EXACT_PORT:-1}"
export CARLA_VULKAN_PREFLIGHT="${CARLA_VULKAN_PREFLIGHT:-warn}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_EGG_CACHE="${PYTHON_EGG_CACHE:-${HOME}/.cache/Python-Eggs}"
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PYTHON_ENV_ROOT="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
export LD_LIBRARY_PATH="${PYTHON_ENV_ROOT}/lib:${CARLA_ROOT}/CarlaUE4/Binaries/Linux:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/bench2drive/scenario_runner
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/bench2drive/scenario_runner

export LEADERBOARD_ROOT=${WORK_DIR}/bench2drive/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

if [[ "${CARLA_SERVER_ARGS}" == *"-RenderOffScreen"* ]] \
    && [ "${CARLA_VULKAN_PREFLIGHT}" != "off" ] \
    && command -v vulkaninfo >/dev/null 2>&1; then
    VULKAN_OUTPUT="$(
        LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}" \
        VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" \
        vulkaninfo 2>&1 || true
    )"
    if ! grep -q "NVIDIA" <<< "${VULKAN_OUTPUT}"; then
        echo "WARNING: vulkaninfo could not confirm an NVIDIA Vulkan device." >&2
        if [ "${CARLA_VULKAN_PREFLIGHT}" = "strict" ]; then
            echo "ERROR: strict Vulkan preflight is enabled; refusing to start CARLA." >&2
            exit 1
        fi
        echo "Continuing because CARLA_VULKAN_PREFLIGHT=${CARLA_VULKAN_PREFLIGHT}." >&2
    fi
fi

CUDA_VISIBLE_DEVICES=${GPU_RANK} ${PYTHON_BIN} ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
