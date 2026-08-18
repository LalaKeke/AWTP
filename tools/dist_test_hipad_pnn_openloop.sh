#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29620}
PYTHON_BIN=${PYTHON_BIN:-python3}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH:-} \
"${PYTHON_BIN}" -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$(dirname "$0")/test_hipad_pnn_openloop.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"
