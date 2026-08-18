#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 CHUNK_ID [PORT]" >&2
  exit 2
fi

CHUNK_ID=$1
PORT=${2:-$((29630 + CHUNK_ID))}
CHUNK=$(printf "%03d" "$CHUNK_ID")
GPUS=${GPUS:-4}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-1}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PORT

INFO_DIR="${PNN_INFO_CHUNK_DIR:-${PROJECT_ROOT}/data/infos/chunks}"
OUT_DIR="${PNN_CHUNK_DIR:-${PROJECT_ROOT}/outputs/inference_chunks}"
ANN_FILE="${INFO_DIR}/b2d_infos_train_chunk_${CHUNK}.pkl"
OUT_FILE="${OUT_DIR}/hipad_stage2_b2d_train_chunk_${CHUNK}.pkl"

if [ ! -f "$ANN_FILE" ]; then
  echo "missing chunk ann file: $ANN_FILE" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

bash ./tools/dist_test.sh \
  ./projects/configs/hipad_b2d_stage2.py \
  ./checkpoints/hipad_stage2.pth \
  "$GPUS" \
  --out "$OUT_FILE" \
  --cfg-options \
  data.test.ann_file="$ANN_FILE" \
  data.test.samples_per_gpu="$SAMPLES_PER_GPU" \
  data.workers_per_gpu="$WORKERS_PER_GPU" \
  work_dir=./outputs/work_dirs/hipad_stage2_b2d_train_infer/chunk_${CHUNK}
