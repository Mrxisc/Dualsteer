#!/usr/bin/env bash
set -euo pipefail
TEXT2IMAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${TEXT2IMAGE_ROOT}"
WORKSPACE_ROOT="$(cd "${TEXT2IMAGE_ROOT}/.." && pwd)"
CACHE_SCRIPT="${REPO_ROOT}/Scripts/collect/collect_real_activations_diffusers.py"

CSV_PATH="${CSV_PATH:-${TEXT2IMAGE_ROOT}/Datasets/i2p_no_sexual.csv}"
MODEL_DIR="${MODEL_DIR:-${TEXT2IMAGE_ROOT}/Models/FLUX.1-dev}"
OUTPUT_ROOT="${TEXT2IMAGE_ROOT}/Activations/i2p_no_sexual_flux_real"
LOG_DIR="${TEXT2IMAGE_ROOT}/Logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"

RESUME="${RESUME:-0}"
CACHE_EVERY_N_TIMESTEPS="${CACHE_EVERY_N_TIMESTEPS:-6}"

TOKEN_SUBSAMPLE="${TOKEN_SUBSAMPLE:-256}"
TOKEN_SUBSAMPLE_SEED="${TOKEN_SUBSAMPLE_SEED:-0}"
CHANNEL_PROJ_DIM="${CHANNEL_PROJ_DIM:-1024}"
CHANNEL_PROJ_SEED="${CHANNEL_PROJ_SEED:-0}"
TOKEN_INDICES_PATH="${TOKEN_INDICES_PATH:-${OUTPUT_ROOT}/token_indices.pt}"
CHANNEL_PROJ_PATH="${CHANNEL_PROJ_PATH:-${OUTPUT_ROOT}/channel_proj.pt}"

# Last single-stream block in diffusers FLUX: transformer.single_transformer_blocks[-1] (len=38 -> idx 37)
SINGLE_HOOK="${SINGLE_HOOK:-transformer.single_transformer_blocks.37}"

RESUME_FLAG=()
if [[ "${RESUME}" != "0" ]]; then
  RESUME_FLAG=(--resume true)
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/collect_i2p_no_sexual_flux_real_single_${TIMESTAMP}.log"

python "${CACHE_SCRIPT}" \
  --backend flux \
  --model_name "${MODEL_DIR}" \
  --hook_names "${SINGLE_HOOK}" \
  --new_cached_activations_path "${OUTPUT_ROOT}" \
  --csv_path "${CSV_PATH}" \
  --csv_prompt_column prompt \
  --csv_category_column categories \
  --seed 42 \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --cache_every_n_timesteps "${CACHE_EVERY_N_TIMESTEPS}" \
  --token_subsample "${TOKEN_SUBSAMPLE}" \
  --token_subsample_seed "${TOKEN_SUBSAMPLE_SEED}" \
  --token_indices_path "${TOKEN_INDICES_PATH}" \
  --channel_proj_dim "${CHANNEL_PROJ_DIM}" \
  --channel_proj_seed "${CHANNEL_PROJ_SEED}" \
  --channel_proj_path "${CHANNEL_PROJ_PATH}" \
  --activation_part tuple1 \
  --cfg_keep cond \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "Logs saved to ${LOG_FILE}"
