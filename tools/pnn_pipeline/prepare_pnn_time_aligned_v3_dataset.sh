#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPAD_ROOT="${HIPAD_PNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CHUNKS="${NUM_CHUNKS:-32}"
OUT_DIR="${PNN_TIME_ALIGNED_DATA_DIR:-${HIPAD_ROOT}/data/pnn/time_aligned_v3}"

cd "${HIPAD_ROOT}"

# 复用已经保存的HiP-AD推理结果；缺少chunk时才补跑感知推理。
missing_chunk=0
for ((chunk=0; chunk<NUM_CHUNKS; chunk++)); do
    path="${PNN_CHUNK_DIR:-${HIPAD_ROOT}/outputs/inference_chunks}/hipad_stage2_b2d_train_chunk_$(printf '%03d' "${chunk}").pkl"
    if [[ ! -s "${path}" ]]; then
        missing_chunk=1
        break
    fi
done
if (( missing_chunk )); then
    START_CHUNK=0 \
    END_CHUNK="$((NUM_CHUNKS - 1))" \
    GPUS="${HIPAD_INFER_GPUS:-4}" \
    SAMPLES_PER_GPU="${HIPAD_INFER_BATCH_SIZE:-1}" \
    WORKERS_PER_GPU="${HIPAD_INFER_WORKERS:-4}" \
    AUTO_MERGE=0 \
    PNN_INFO_CHUNK_DIR="${PNN_INFO_CHUNK_DIR:-${HIPAD_ROOT}/data/infos/chunks}" \
    PNN_CHUNK_DIR="${PNN_CHUNK_DIR:-${HIPAD_ROOT}/outputs/inference_chunks}" \
    bash "${SCRIPT_DIR}/run_infer_train_all_chunks.sh"
else
    echo "[time-aligned-data] reusing ${NUM_CHUNKS} existing HiP-AD chunks"
fi

# 只有动态目标输入发生变化。复用原GT监督和其他场景张量，只重建
# ped_states/veh_states，可避免重复生成1.8 GiB的GT actor boxes。
BASE_DATA_DIR="${PNN_BASE_STATIC_DATA_DIR:-${HIPAD_ROOT}/data/pnn/static_v1}"
OLD_PATH="${OUT_DIR}/train_old.pt"
NEW_PATH="${BASE_DATA_DIR}/train_new.pt"
PNN_CONVERT_WORKERS="${PNN_CONVERT_WORKERS:-4}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/realign_pnn_actor_inputs_parallel.py" \
  --num-chunks "${NUM_CHUNKS}" \
  --chunk-dir "${PNN_CHUNK_DIR:-${HIPAD_ROOT}/outputs/inference_chunks}" \
  --base-old "${BASE_DATA_DIR}/train_old.pt" \
  --output-old "${OLD_PATH}" \
  --coord-convention pnn_xy \
  --score-thr "${PNN_STATIC_SCORE_THR:-0.3}" \
  --actor-motion-source-dt 0.1

STATS_PATH="${OUT_DIR}/pnn_time_aligned_v3_inference_stats_q005_q995.pt"
# 该分支只部署普通ControlNet；静态张量可保留在数据中，但不属于模型输入。
"${PYTHON_BIN}" "${SCRIPT_DIR}/create_pnn_inference_stats.py" \
  --input "${OLD_PATH}" \
  --output "${STATS_PATH}" \
  --q-low 0.005 \
  --q-high 0.995

ALIGNMENT_VERSION="$(PYTHONPATH="${HIPAD_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c 'from pnn_temporal_alignment import ALIGNMENT_VERSION; print(ALIGNMENT_VERSION)')"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_pnn_actor_alignment.py" \
  --input "${OLD_PATH}" \
  --expected-version "${ALIGNMENT_VERSION}"

echo "[time-aligned-data] old=${OLD_PATH}"
echo "[time-aligned-data] new=${NEW_PATH} (reused; GT supervision is unchanged)"
echo "[time-aligned-data] stats=${STATS_PATH}"
