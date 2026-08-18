#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CHUNKS="${NUM_CHUNKS:-32}"
OUT_DIR="${PNN_STATIC_DATA_DIR:-${HIPAD_ROOT}/data/pnn/static_v1}"
INFO_DIR="${PNN_INFO_CHUNK_DIR:-${HIPAD_ROOT}/data/infos/chunks}"
CHUNK_DIR="${PNN_CHUNK_DIR:-${HIPAD_ROOT}/outputs/inference_chunks}"

cd "${HIPAD_ROOT}"

# 清理操作后分块标注可能不存在，先从原始训练 info 恢复。
if [[ ! -s "${INFO_DIR}/b2d_infos_train_chunk_000.pkl" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/split_train_infos.py" \
    --src "${HIPAD_ROOT}/data/infos/b2d_infos_train.pkl" \
    --out-dir "${INFO_DIR}" \
    --num-chunks "${NUM_CHUNKS}"
fi

# 只运行缺失的 HiP-AD 推理 chunk；已有 chunk 会自动跳过。
START_CHUNK=0 \
END_CHUNK="$((NUM_CHUNKS - 1))" \
GPUS="${HIPAD_INFER_GPUS:-4}" \
SAMPLES_PER_GPU="${HIPAD_INFER_BATCH_SIZE:-1}" \
WORKERS_PER_GPU="${HIPAD_INFER_WORKERS:-4}" \
AUTO_MERGE=0 \
PNN_INFO_CHUNK_DIR="${INFO_DIR}" \
PNN_CHUNK_DIR="${CHUNK_DIR}" \
bash "${SCRIPT_DIR}/run_infer_train_all_chunks.sh"

PNN_CONVERT_WORKERS="${PNN_CONVERT_WORKERS:-4}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/convert_hipad_static_chunks_parallel.py" \
  --num-chunks "${NUM_CHUNKS}" \
  --chunk-dir "${CHUNK_DIR}" \
  --info-dir "${INFO_DIR}" \
  --all-info-pkl "${HIPAD_ROOT}/data/infos/b2d_infos_train.pkl" \
  --output-dir "${OUT_DIR}" \
  --old-name train_old.pt \
  --new-name train_new.pt \
  --gt-source true_gt \
  --route-source hipad_plan \
  --coord-convention pnn_xy \
  --score-thr "${PNN_STATIC_SCORE_THR:-0.3}"

STATS_PATH="${OUT_DIR}/pnn_static_inference_stats_q005_q995.pt"
"${PYTHON_BIN}" "${SCRIPT_DIR}/create_pnn_static_inference_stats.py" \
  --input "${OUT_DIR}/train_old.pt" \
  --output "${STATS_PATH}" \
  --q-low 0.005 \
  --q-high 0.995

echo "[static-v1-data] old=${OUT_DIR}/train_old.pt"
echo "[static-v1-data] new=${OUT_DIR}/train_new.pt"
echo "[static-v1-data] stats=${STATS_PATH}"
