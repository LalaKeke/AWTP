#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUT_DIR="${PNN_CHUNK_DIR:-$ROOT/outputs/inference_chunks}"
RUN_ONE="$SCRIPT_DIR/run_infer_train_chunk.sh"
MERGE="$SCRIPT_DIR/merge_chunk_outputs.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_CHUNK=${START_CHUNK:-0}
END_CHUNK=${END_CHUNK:-31}
PORT_BASE=${PORT_BASE:-29630}
POLL_SECONDS=${POLL_SECONDS:-60}
AUTO_MERGE=${AUTO_MERGE:-1}

GPUS=${GPUS:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-1}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}

cd "$ROOT"
mkdir -p "$OUT_DIR"

chunk_is_running() {
  local chunk_id="$1"
  pgrep -af "bash .*run_infer_train_chunk.sh[[:space:]]+${chunk_id}([[:space:]]|\$)" >/dev/null 2>&1
}

for chunk_id in $(seq "$START_CHUNK" "$END_CHUNK"); do
  chunk=$(printf "%03d" "$chunk_id")
  out_file="$OUT_DIR/hipad_stage2_b2d_train_chunk_${chunk}.pkl"
  log_file="$OUT_DIR/chunk_${chunk}.log"

  if [ -s "$out_file" ]; then
    echo "[$(date '+%F %T')] chunk ${chunk} already done, skip: $out_file"
    continue
  fi

  while chunk_is_running "$chunk_id"; do
    echo "[$(date '+%F %T')] chunk ${chunk} is already running elsewhere; waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
    if [ -s "$out_file" ]; then
      echo "[$(date '+%F %T')] chunk ${chunk} finished by existing process, skip."
      continue 2
    fi
  done

  port=$((PORT_BASE + chunk_id))
  echo "[$(date '+%F %T')] start chunk ${chunk} on port ${port}" | tee -a "$log_file"
  echo "[$(date '+%F %T')] GPUS=${GPUS}, SAMPLES_PER_GPU=${SAMPLES_PER_GPU}, WORKERS_PER_GPU=${WORKERS_PER_GPU}" | tee -a "$log_file"

  GPUS="$GPUS" \
  SAMPLES_PER_GPU="$SAMPLES_PER_GPU" \
  WORKERS_PER_GPU="$WORKERS_PER_GPU" \
  PNN_CHUNK_DIR="$OUT_DIR" \
  "$RUN_ONE" "$chunk_id" "$port" >> "$log_file" 2>&1

  if [ ! -s "$out_file" ]; then
    echo "[$(date '+%F %T')] chunk ${chunk} finished but output is missing: $out_file" >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] chunk ${chunk} done: $out_file" | tee -a "$log_file"
done

if [ "$AUTO_MERGE" = "1" ]; then
  echo "[$(date '+%F %T')] all chunks done; merging outputs..."
  "$PYTHON_BIN" "$MERGE" --chunk-dir "$OUT_DIR" --out "$ROOT/outputs/hipad_train_outputs.pkl"
fi
