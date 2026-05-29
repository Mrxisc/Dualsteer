#!/usr/bin/env bash
set -euo pipefail

# Local helper for caching activations from the i2p_no_sexual CSV.
# Update CSV_PATH and MODEL_DIR to match your workspace before running.

TEXT2IMAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${TEXT2IMAGE_ROOT}"
WORKSPACE_ROOT="$(cd "${TEXT2IMAGE_ROOT}/.." && pwd)"
CACHE_SCRIPT="${REPO_ROOT}/Scripts/collect/collect_activations.py"
CSV_PATH="${CSV_PATH:-${TEXT2IMAGE_ROOT}/Datasets/i2p_no_sexual.csv}"
MODEL_DIR="${MODEL_DIR:-${TEXT2IMAGE_ROOT}/Models/FLUX.1-dev}"
FLUX_CONFIG="${REPO_ROOT}/_Backup_code/configs/flux1_dev.json"
OUTPUT_ROOT="${TEXT2IMAGE_ROOT}/Activations/i2p_no_sexual_flux"
LOG_DIR="${TEXT2IMAGE_ROOT}/Logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-30}"

DEFAULT_HOOKS=(
    text_encoder
)

ALL_HOOKS=(
    text_encoder
)
for idx in {0..5}; do
    ALL_HOOKS+=("flux_model.double_blocks.${idx}")
done
for idx in {0..11}; do
    ALL_HOOKS+=("flux_model.single_blocks.${idx}")
done

if [[ ${1:-} == "--all-hooks" ]]; then
    HOOKS=("${ALL_HOOKS[@]}")
    shift || true
elif [[ $# -gt 0 ]]; then
    HOOKS=("$@")
else
    HOOKS=("${DEFAULT_HOOKS[@]}")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/collect_i2p_no_sexual_flux_textencoder_${TIMESTAMP}.log"
STATUS=0

for hook in "${HOOKS[@]}"; do
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
        --use_flux \
        --flux_config_path "${FLUX_CONFIG}" \
        --output_or_diff output \
        2>&1 | tee -a "${LOG_FILE}"

    STATUS=$?
    [[ ${STATUS} -ne 0 ]] && break
done

STATUS=$?

if [[ ${STATUS} -ne 0 ]]; then
    echo "Script failed; see ${LOG_FILE}" >&2
fi

echo "Logs saved to ${LOG_FILE}"
exit ${STATUS}
