#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PORT=${PORT:-29621}

bash ./tools/dist_test.sh \
  ./projects/configs/hipad_b2d_stage2.py \
  ./checkpoints/hipad_stage2.pth \
  4 \
  --out ./outputs/hipad_train_outputs.pkl \
  --cfg-options \
  data.test.ann_file=data/infos/b2d_infos_train.pkl \
  data.workers_per_gpu=2 \
  work_dir=./outputs/work_dirs/hipad_stage2_b2d_train_infer
