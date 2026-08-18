#!/bin/bash
set -u

# HiP-AD Bench2Drive watchdog.
# - Starts pending splits on available GPUs.
# - Restarts a split when its evaluator/CARLA process exits before completion.
# - Optional: set STALL_SECONDS>0 to restart when log/checkpoint files stop updating.
# - Uses atomic lock directories to avoid running the same split twice.

BASE_PORT="${BASE_PORT:-30000}"
BASE_TM_PORT="${BASE_TM_PORT:-50000}"
PORT_STRIDE="${PORT_STRIDE:-200}"
IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-True}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MAP_ROOT="${MAP_ROOT:-$(dirname "${WORK_DIR}")}"

CONFIG_NAME="${CONFIG_NAME:-hipad_b2d_stage2_pnn}"
TEAM_AGENT="${TEAM_AGENT:-bench2drive/leaderboard/team_code/hipad_pnn_b2d_agent.py}"
TEAM_CONFIG="${TEAM_CONFIG:-${WORK_DIR}/projects/configs/hipad_b2d_stage2.py+${WORK_DIR}/checkpoints/hipad_stage2.pth+${CONFIG_NAME}}"
PLANNER_TYPE="${PLANNER_TYPE:-traj}"
BASE_ROUTES="${BASE_ROUTES:-bench2drive/leaderboard/data/splits16/bench2drive220}"
SAVE_PATH="${SAVE_PATH:-evaluation/${CONFIG_NAME}}"
BASE_CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT:-evaluation/${CONFIG_NAME}/${CONFIG_NAME}}"

TASK_LIST_STR="${TASK_LIST:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
GPU_RANK_LIST_STR="${GPU_RANK_LIST:-0 1 2 3}"

CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
START_DELAY="${START_DELAY:-5}"
STALL_SECONDS="${STALL_SECONDS:-900}"
CARLA_START_GRACE="${CARLA_START_GRACE:-240}"
STARTUP_RETRY_LIMIT="${STARTUP_RETRY_LIMIT:-5}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-20000}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
LOCK_ROOT="${LOCK_ROOT:-${SAVE_PATH}/watchdog_locks}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${SAVE_PATH}/watchdog.log}"
PYTHON_JSON_BIN="${PYTHON_JSON_BIN:-${PYTHON_BIN:-python}}"
LOG_APPEND="${LOG_APPEND:-1}"
AUTO_SKIP_CRASH_ROUTES="${AUTO_SKIP_CRASH_ROUTES:-1}"
CRASH_SKIP_THRESHOLD="${CRASH_SKIP_THRESHOLD:-3}"
SKIPPED_ROUTES_LOG="${SKIPPED_ROUTES_LOG:-${SAVE_PATH}/watchdog_skipped_routes.jsonl}"
ROUTE_STATE_TOOL="${ROUTE_STATE_TOOL:-bench2drive/leaderboard/scripts/watchdog_route_state.py}"

mkdir -p "${SAVE_PATH}" "${LOCK_ROOT}"

read -r -a TASK_LIST <<< "${TASK_LIST_STR}"
read -r -a GPU_RANK_LIST <<< "${GPU_RANK_LIST_STR}"

if [ "${#TASK_LIST[@]}" -eq 0 ] || [ "${#GPU_RANK_LIST[@]}" -eq 0 ]; then
    echo "ERROR: TASK_LIST and GPU_RANK_LIST must not be empty" >&2
    exit 2
fi
if [ ! -f "${ROUTE_STATE_TOOL}" ]; then
    echo "ERROR: watchdog helper is missing" >&2
    echo "ROUTE_STATE_TOOL=${ROUTE_STATE_TOOL}" >&2
    exit 2
fi
if ! command -v "${PYTHON_JSON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: watchdog Python executable is missing" >&2
    echo "PYTHON_JSON_BIN=${PYTHON_JSON_BIN}" >&2
    exit 2
fi

log_msg() {
    local msg="$1"
    local ts
    ts="$(date '+%F %T')"
    echo "[${ts}] ${msg}" | tee -a "${WATCHDOG_LOG}"
}

task_port() {
    local idx="$1"
    echo $((BASE_PORT + idx * PORT_STRIDE))
}

task_tm_port() {
    local idx="$1"
    echo $((BASE_TM_PORT + idx * PORT_STRIDE))
}

checkpoint_file() {
    local split="$1"
    echo "${BASE_CHECKPOINT_ENDPOINT}_${split}.json"
}

log_file() {
    local split="$1"
    echo "${BASE_CHECKPOINT_ENDPOINT}_${split}.log"
}

split_lock_dir() {
    local split="$1"
    echo "${LOCK_ROOT}/split_${split}.lock"
}

gpu_lock_dir() {
    local gpu="$1"
    echo "${LOCK_ROOT}/gpu_${gpu}.lock"
}

gpu_has_capacity() {
    local gpu="$1"
    local free_mb
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    free_mb="$(
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" \
            2>/dev/null | sed -n '1p' | tr -d ' '
    )"
    if ! [[ "${free_mb}" =~ ^[0-9]+$ ]]; then
        return 0
    fi
    [ "${free_mb}" -ge "${MIN_GPU_FREE_MB}" ]
}

is_pid_alive() {
    local pid="$1"
    [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1
}

read_lock_pid() {
    local lock_dir="$1"
    [ -f "${lock_dir}/pid" ] && sed -n '1p' "${lock_dir}/pid"
}

release_lock() {
    local lock_dir="$1"
    rm -rf "${lock_dir}"
}

kill_process_tree() {
    local pid="$1"
    local child

    if ! is_pid_alive "${pid}"; then
        return
    fi

    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_process_tree "${child}"
    done

    kill -TERM "${pid}" >/dev/null 2>&1 || true
}

kill_process_tree_force() {
    local pid="$1"
    local child

    if ! is_pid_alive "${pid}"; then
        return
    fi

    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_process_tree_force "${child}"
    done

    kill -KILL "${pid}" >/dev/null 2>&1 || true
}

acquire_lock() {
    local lock_dir="$1"
    local split="$2"
    local gpu="$3"

    if mkdir "${lock_dir}" 2>/dev/null; then
        echo "$$" > "${lock_dir}/owner"
        echo "${split}" > "${lock_dir}/split"
        echo "${gpu}" > "${lock_dir}/gpu"
        return 0
    fi

    local pid
    pid="$(read_lock_pid "${lock_dir}")"
    if ! is_pid_alive "${pid}"; then
        release_lock "${lock_dir}"
        if mkdir "${lock_dir}" 2>/dev/null; then
            echo "$$" > "${lock_dir}/owner"
            echo "${split}" > "${lock_dir}/split"
            echo "${gpu}" > "${lock_dir}/gpu"
            return 0
        fi
    fi

    return 1
}

progress_text() {
    local split="$1"
    "${PYTHON_JSON_BIN}" "${ROUTE_STATE_TOOL}" state \
        "$(checkpoint_file "${split}")" "${BASE_ROUTES}_${split}.xml" 2>/dev/null \
        || echo "0 0 unreadable none"
}

is_split_done() {
    local split="$1"
    local cur total status route_id
    read -r cur total status route_id <<< "$(progress_text "${split}")"
    [ "${total}" != "0" ] && [ "${cur}" -ge "${total}" ]
}

task_pid_for_split() {
    local split="$1"
    local lock_dir
    lock_dir="$(split_lock_dir "${split}")"
    read_lock_pid "${lock_dir}"
}

carla_running_for_port() {
    local port="$1"
    pgrep -f "CarlaUE4.*-carla-rpc-port=${port}" >/dev/null 2>&1
}

kill_task_processes() {
    local split="$1"
    local idx="$2"
    local gpu="$3"
    local port tm_port pid
    port="$(task_port "${idx}")"
    tm_port="$(task_tm_port "${idx}")"
    pid="$(task_pid_for_split "${split}")"

    log_msg "Stopping split ${split} on GPU ${gpu} port ${port}"

    if is_pid_alive "${pid}"; then
        kill -TERM "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
        kill_process_tree "${pid}"
        sleep 5
        if is_pid_alive "${pid}"; then
            kill -KILL "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
            kill_process_tree_force "${pid}"
        fi
    fi

    pkill -f "leaderboard_evaluator.py .*--port=${port}" >/dev/null 2>&1 || true
    pkill -f "CarlaUE4.*-carla-rpc-port=${port}" >/dev/null 2>&1 || true
    pkill -f "CarlaUE4.*-traffic-manager-port=${tm_port}" >/dev/null 2>&1 || true
}

start_task() {
    local idx="$1"
    local split="$2"
    local gpu="$3"
    local port tm_port routes checkpoint log split_lock gpu_lock restarts

    if is_split_done "${split}"; then
        log_msg "Skip split ${split}: already complete"
        return 0
    fi

    split_lock="$(split_lock_dir "${split}")"
    gpu_lock="$(gpu_lock_dir "${gpu}")"

    if ! acquire_lock "${split_lock}" "${split}" "${gpu}"; then
        return 1
    fi
    if ! acquire_lock "${gpu_lock}" "${split}" "${gpu}"; then
        release_lock "${split_lock}"
        return 1
    fi

    restarts=0
    [ -f "${LOCK_ROOT}/split_${split}.restarts" ] && restarts="$(sed -n '1p' "${LOCK_ROOT}/split_${split}.restarts")"
    if [ "${MAX_RESTARTS}" -gt 0 ] && [ "${restarts}" -ge "${MAX_RESTARTS}" ]; then
        log_msg "Skip split ${split}: restart limit ${MAX_RESTARTS} reached"
        release_lock "${gpu_lock}"
        release_lock "${split_lock}"
        return 1
    fi
    echo $((restarts + 1)) > "${LOCK_ROOT}/split_${split}.restarts"

    port="$(task_port "${idx}")"
    tm_port="$(task_tm_port "${idx}")"
    routes="${BASE_ROUTES}_${split}.xml"
    checkpoint="$(checkpoint_file "${split}")"
    log="$(log_file "${split}")"

    kill_task_processes "${split}" "${idx}" "${gpu}"
    if [ "${LOG_APPEND}" = "0" ]; then
        : > "${log}"
    fi
    stat -c %s "${log}" 2>/dev/null > "${split_lock}/log_offset" || echo 0 > "${split_lock}/log_offset"
    log_msg "Starting split ${split} on GPU ${gpu}, port ${port}, checkpoint ${checkpoint}"
    setsid bash -e bench2drive/leaderboard/scripts/run_evaluation.sh \
        "${port}" "${tm_port}" "${IS_BENCH2DRIVE}" "${routes}" "${TEAM_AGENT}" \
        "${TEAM_CONFIG}" "${checkpoint}" "${SAVE_PATH}" "${PLANNER_TYPE}" "${gpu}" \
        >> "${log}" 2>&1 &

    echo "$!" > "${split_lock}/pid"
    echo "$!" > "${gpu_lock}/pid"
    echo "${split}" > "${gpu_lock}/split"
    echo "${idx}" > "${split_lock}/idx"
    echo "${idx}" > "${gpu_lock}/idx"
    date +%s > "${split_lock}/started_at"
    date +%s > "${gpu_lock}/started_at"

    sleep "${START_DELAY}"
}

route_started_this_attempt() {
    local split="$1"
    local split_lock log offset
    split_lock="$(split_lock_dir "${split}")"
    log="$(log_file "${split}")"
    offset="$(sed -n '1p' "${split_lock}/log_offset" 2>/dev/null || echo 0)"
    [ -f "${log}" ] || return 1
    tail -c "+$((offset + 1))" "${log}" 2>/dev/null | grep -aq "Preparing RouteScenario_"
}

record_startup_failure() {
    local split="$1"
    local reason="$2"
    local count_file count
    count_file="${LOCK_ROOT}/split_${split}.startup_failures"
    count=0
    [ -f "${count_file}" ] && count="$(sed -n '1p' "${count_file}")"
    count=$((count + 1))
    echo "${count}" > "${count_file}"
    log_msg "Split ${split} startup failure ${count}/${STARTUP_RETRY_LIMIT}: ${reason}"
    if [ "${count}" -ge "${STARTUP_RETRY_LIMIT}" ]; then
        echo "${reason}" > "${LOCK_ROOT}/split_${split}.blocked"
        log_msg "Split ${split} blocked after repeated startup failures; routes were not skipped"
        return 0
    fi
    return 1
}

record_failure_for_attempt() {
    local split="$1"
    local gpu="$2"
    local reason="$3"
    if route_started_this_attempt "${split}"; then
        rm -f "${LOCK_ROOT}/split_${split}.startup_failures"
        record_route_failure "${split}" "${gpu}" "${reason}" || true
    else
        record_startup_failure "${split}" "${reason}" || true
    fi
}

log_is_stale() {
    local split="$1"
    local log checkpoint split_lock now latest log_mtime checkpoint_mtime started_at
    log="$(log_file "${split}")"
    checkpoint="$(checkpoint_file "${split}")"
    split_lock="$(split_lock_dir "${split}")"
    now="$(date +%s)"
    latest=0

    started_at="$(sed -n '1p' "${split_lock}/started_at" 2>/dev/null || echo 0)"
    [ "${started_at}" -gt "${latest}" ] && latest="${started_at}"

    if [ -f "${log}" ]; then
        log_mtime="$(stat -c %Y "${log}" 2>/dev/null || echo 0)"
        [ "${log_mtime}" -gt "${latest}" ] && latest="${log_mtime}"
    fi

    if [ -f "${checkpoint}" ]; then
        checkpoint_mtime="$(stat -c %Y "${checkpoint}" 2>/dev/null || echo 0)"
        [ "${checkpoint_mtime}" -gt "${latest}" ] && latest="${checkpoint_mtime}"
    fi

    [ "${STALL_SECONDS}" -gt 0 ] && [ "${latest}" != "0" ] && [ $((now - latest)) -gt "${STALL_SECONDS}" ]
}

last_preparing_route() {
    local split="$1"
    local log
    log="$(log_file "${split}")"
    [ -f "${log}" ] || return
    grep -a "Preparing RouteScenario_" "${log}" | tail -1 | sed -E 's/.*Preparing (RouteScenario_[^ ]+) \(repetition ([0-9]+)\).*/\1_rep\2/'
}

mark_route_skipped() {
    local split="$1"
    local gpu="$2"
    local route_id="$3"
    local reason="$4"
    local checkpoint routes
    checkpoint="$(checkpoint_file "${split}")"
    routes="${BASE_ROUTES}_${split}.xml"
    "${PYTHON_JSON_BIN}" "${ROUTE_STATE_TOOL}" skip \
        "${checkpoint}" "${routes}" "${SKIPPED_ROUTES_LOG}" \
        "${split}" "${gpu}" "${reason}"
}

record_route_failure() {
    local split="$1"
    local gpu="$2"
    local reason="$3"
    local route_id count_file count cur total status

    [ "${AUTO_SKIP_CRASH_ROUTES}" = "1" ] || return 1
    read -r cur total status route_id <<< "$(progress_text "${split}")"
    if [ -z "${route_id}" ] || [ "${route_id}" = "none" ]; then
        route_id="$(last_preparing_route "${split}")"
    fi
    [ -n "${route_id}" ] || return 1

    count_file="${LOCK_ROOT}/split_${split}_${route_id}.route_failures"
    count=0
    [ -f "${count_file}" ] && count="$(sed -n '1p' "${count_file}")"
    count=$((count + 1))
    echo "${count}" > "${count_file}"

    log_msg "Split ${split} route ${route_id} failure count ${count}/${CRASH_SKIP_THRESHOLD}: ${reason}"

    if [ "${count}" -lt "${CRASH_SKIP_THRESHOLD}" ]; then
        return 1
    fi

    if mark_route_skipped "${split}" "${gpu}" "${route_id}" "${reason}; repeated ${count} times"; then
        log_msg "Split ${split} route ${route_id} skipped after ${count} failures"
        rm -f "${count_file}"
        return 0
    fi

    log_msg "Split ${split} route ${route_id} reached skip threshold, but checkpoint patch failed"
    return 1
}

monitor_task() {
    local idx="$1"
    local split="$2"
    local gpu="$3"
    local pid split_lock gpu_lock cur total status route_id port now started_at age

    split_lock="$(split_lock_dir "${split}")"
    gpu_lock="$(gpu_lock_dir "${gpu}")"

    if [ ! -d "${split_lock}" ]; then
        return
    fi

    pid="$(read_lock_pid "${split_lock}")"
    read -r cur total status route_id <<< "$(progress_text "${split}")"

    if is_split_done "${split}"; then
        log_msg "Split ${split} complete: progress [${cur}/${total}], status=${status}"
        kill_task_processes "${split}" "${idx}" "${gpu}"
        release_lock "${gpu_lock}"
        release_lock "${split_lock}"
        return
    fi

    if ! is_pid_alive "${pid}"; then
        log_msg "Split ${split} process exited before completion: progress [${cur}/${total}], status=${status}"
        record_failure_for_attempt "${split}" "${gpu}" "evaluator exited before route completion"
        release_lock "${gpu_lock}"
        release_lock "${split_lock}"
        [ ! -f "${LOCK_ROOT}/split_${split}.blocked" ] && start_task "${idx}" "${split}" "${gpu}" || true
        return
    fi

    port="$(task_port "${idx}")"
    now="$(date +%s)"
    started_at="$(sed -n '1p' "${split_lock}/started_at" 2>/dev/null || stat -c %Y "${split_lock}" 2>/dev/null || echo "${now}")"
    age=$((now - started_at))
    if [ "${age}" -gt "${CARLA_START_GRACE}" ] && ! carla_running_for_port "${port}"; then
        log_msg "Split ${split} evaluator is alive but CARLA is missing on port ${port} after ${age}s: progress [${cur}/${total}], status=${status}"
        kill_task_processes "${split}" "${idx}" "${gpu}"
        record_failure_for_attempt "${split}" "${gpu}" "CARLA process missing after startup grace"
        release_lock "${gpu_lock}"
        release_lock "${split_lock}"
        [ ! -f "${LOCK_ROOT}/split_${split}.blocked" ] && start_task "${idx}" "${split}" "${gpu}" || true
        return
    fi

    if log_is_stale "${split}"; then
        log_msg "Split ${split} appears stuck for >${STALL_SECONDS}s: progress [${cur}/${total}], status=${status}"
        kill_task_processes "${split}" "${idx}" "${gpu}"
        record_failure_for_attempt "${split}" "${gpu}" "no log/checkpoint update for ${STALL_SECONDS}s"
        release_lock "${gpu_lock}"
        release_lock "${split_lock}"
        [ ! -f "${LOCK_ROOT}/split_${split}.blocked" ] && start_task "${idx}" "${split}" "${gpu}" || true
    fi
}

schedule_pending_tasks() {
    local idx split gpu
    for idx in "${!TASK_LIST[@]}"; do
        split="${TASK_LIST[$idx]}"
        if is_split_done "${split}"; then
            continue
        fi
        if [ -f "${LOCK_ROOT}/split_${split}.blocked" ]; then
            continue
        fi
        if [ -d "$(split_lock_dir "${split}")" ]; then
            continue
        fi

        for gpu in "${GPU_RANK_LIST[@]}"; do
            if [ ! -d "$(gpu_lock_dir "${gpu}")" ] && gpu_has_capacity "${gpu}"; then
                start_task "${idx}" "${split}" "${gpu}" || true
                break
            fi
        done
    done
}

monitor_running_tasks() {
    local idx split gpu_lock gpu locked_split locked_idx
    for gpu in "${GPU_RANK_LIST[@]}"; do
        gpu_lock="$(gpu_lock_dir "${gpu}")"
        [ -d "${gpu_lock}" ] || continue
        locked_split="$(sed -n '1p' "${gpu_lock}/split" 2>/dev/null || true)"
        locked_idx="$(sed -n '1p' "${gpu_lock}/idx" 2>/dev/null || true)"
        [ -n "${locked_split}" ] || { release_lock "${gpu_lock}"; continue; }
        [ -d "$(split_lock_dir "${locked_split}")" ] || { release_lock "${gpu_lock}"; continue; }
        [ -n "${locked_idx}" ] || locked_idx="${locked_split}"
        monitor_task "${locked_idx}" "${locked_split}" "${gpu}"
    done
}

cleanup_finished_locks() {
    local split split_lock gpu gpu_lock
    for split in "${TASK_LIST[@]}"; do
        split_lock="$(split_lock_dir "${split}")"
        [ -d "${split_lock}" ] || continue
        if is_split_done "${split}"; then
            release_lock "${split_lock}"
        fi
    done
    for gpu in "${GPU_RANK_LIST[@]}"; do
        gpu_lock="$(gpu_lock_dir "${gpu}")"
        [ -d "${gpu_lock}" ] || continue
        split="$(sed -n '1p' "${gpu_lock}/split" 2>/dev/null || true)"
        if [ -z "${split}" ] || is_split_done "${split}"; then
            release_lock "${gpu_lock}"
        fi
    done
}

if [ "${STALL_SECONDS}" -gt 0 ]; then
    log_msg "Watchdog started: tasks=(${TASK_LIST_STR}), gpus=(${GPU_RANK_LIST_STR}), stale-log restart=${STALL_SECONDS}s, min GPU free=${MIN_GPU_FREE_MB}MB"
else
    log_msg "Watchdog started: tasks=(${TASK_LIST_STR}), gpus=(${GPU_RANK_LIST_STR}), stale-log restart=disabled, min GPU free=${MIN_GPU_FREE_MB}MB"
fi

all_splits_done() {
    local split
    for split in "${TASK_LIST[@]}"; do
        is_split_done "${split}" || return 1
    done
    return 0
}

cleanup_watchdog() {
    local status=$?
    local gpu gpu_lock split idx
    trap - EXIT
    for gpu in "${GPU_RANK_LIST[@]}"; do
        gpu_lock="$(gpu_lock_dir "${gpu}")"
        [ -d "${gpu_lock}" ] || continue
        split="$(sed -n '1p' "${gpu_lock}/split" 2>/dev/null || true)"
        idx="$(sed -n '1p' "${gpu_lock}/idx" 2>/dev/null || true)"
        if [ -n "${split}" ] && [ -n "${idx}" ]; then
            kill_task_processes "${split}" "${idx}" "${gpu}"
            release_lock "$(split_lock_dir "${split}")"
        fi
        release_lock "${gpu_lock}"
    done
    exit "${status}"
}

any_split_blocked() {
    local split
    for split in "${TASK_LIST[@]}"; do
        [ -f "${LOCK_ROOT}/split_${split}.blocked" ] && return 0
    done
    return 1
}

trap cleanup_watchdog EXIT
trap 'exit 130' INT TERM

while true; do
    cleanup_finished_locks
    monitor_running_tasks
    schedule_pending_tasks
    if all_splits_done; then
        log_msg "All requested splits are complete"
        break
    fi
    if any_split_blocked; then
        log_msg "Watchdog stopped: at least one split is blocked by repeated startup failures"
        exit 2
    fi
    sleep "${CHECK_INTERVAL}"
done
