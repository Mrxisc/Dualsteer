set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
FLUX_CKPT="${FLUX_CKPT:-${ROOT_DIR}/Models/FLUX.1-dev}"
PROMPTS_CSV="${PROMPTS_CSV:-${ROOT_DIR}/Datasets/i2p_benchmark.csv}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT_DIR}/Results/i2p_flux_text_encoder}"
STEPS=${STEPS:-30}
GUIDANCE=${GUIDANCE:-7.0}
NUM_IMAGES=${NUM_IMAGES:-10}
BATCH_SIZE=${BATCH_SIZE:-2}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-256}
NEG_PROMPT=${NEG_PROMPT:-""}
SEED_BASE=${SEED_BASE:-42}
DTYPE=${DTYPE:-fp16}

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

LOG_DIR="${ROOT_DIR}/Logs"
mkdir -p "${LOG_DIR}"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/generate_flux_image_i2p47030_${TS}.log"

{
  echo "timestamp=${TS}"
  echo "flux_ckpt=${FLUX_CKPT}"
  echo "prompts_csv=${PROMPTS_CSV}"
  echo "save_root=${SAVE_ROOT}"
  echo "steps=${STEPS} guidance=${GUIDANCE} num_images=${NUM_IMAGES} batch_size=${BATCH_SIZE}"
  echo "height=${HEIGHT} width=${WIDTH} seed_base=${SEED_BASE} dtype=${DTYPE}"
  echo "negative_prompt=${NEG_PROMPT}"

  python "${ROOT_DIR}/Scripts/generate/generate_flux_image_i2p_47030_textencoder.py" \
    --flux-ckpt "${FLUX_CKPT}" \
    --prompts-csv "${PROMPTS_CSV}" \
    --save-root "${SAVE_ROOT}" \
    --steps "${STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --num-images-per-prompt "${NUM_IMAGES}" \
    --batch-size "${BATCH_SIZE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --negative-prompt "${NEG_PROMPT}" \
    --seed-base "${SEED_BASE}" \
    --dtype "${DTYPE}" \
    2>&1
} | tee "${LOG_FILE}"

echo "Logs: ${LOG_FILE}"
