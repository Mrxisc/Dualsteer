#!/usr/bin/env bash
set -euo pipefail
TEXT2IMAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${TEXT2IMAGE_ROOT}"
WORKSPACE_ROOT="$(cd "${TEXT2IMAGE_ROOT}/.." && pwd)"
CACHE_SCRIPT="${REPO_ROOT}/Scripts/collect/collect_activations.py"
CSV_PATH="${CSV_PATH:-${TEXT2IMAGE_ROOT}/Datasets/i2p_no_sexual.csv}"
MODEL_DIR="${MODEL_DIR:-${TEXT2IMAGE_ROOT}/Models/SD/stable-diffusion-3.5-large}"
OUTPUT_ROOT="${TEXT2IMAGE_ROOT}/Activations/i2p_no_sexual_SD"
LOG_DIR="${TEXT2IMAGE_ROOT}/Logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-30}"
declare -a HOOKS_TO_RUN=()

DEFAULT_TEXT_HOOKS=(
    "text_encoder"
    "text_encoder_2"
    "text_encoder_3"
)

if [[ $# -gt 0 ]]; then
    HOOKS_TO_RUN=("$@")
else
    HOOKS_TO_RUN=("${DEFAULT_TEXT_HOOKS[@]}")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/collect_i2p_no_sexual_sd35_textencoder_${TIMESTAMP}.log"
STATUS=0

for hook in "${HOOKS_TO_RUN[@]}"; do
    echo "==== Processing hook ${hook} ====" | tee -a "${LOG_FILE}"
    python "${CACHE_SCRIPT}" \
        --model_name "${MODEL_DIR}" \
        --hook_names "${hook}" \
        --new_cached_activations_path "${OUTPUT_ROOT}" \
        --csv_path "${CSV_PATH}" \
        --csv_prompt_column prompt \
        --csv_category_column categories \
        --guidance_scale 4.0 \
        --num_inference_steps "${NUM_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --seed 42 \
        --denoiser_attr transformer \
        --output_or_diff output \
        2>&1 | tee -a "${LOG_FILE}"

    STATUS=$?
    [[ ${STATUS} -ne 0 ]] && break
done

if [[ ${STATUS} -ne 0 ]]; then
    echo "Script failed; see ${LOG_FILE}" >&2
fi

echo "Logs saved to ${LOG_FILE}"
exit ${STATUS}
