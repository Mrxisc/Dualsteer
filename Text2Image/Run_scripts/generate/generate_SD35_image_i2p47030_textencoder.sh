set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
SD35_CKPT="${SD35_CKPT:-${ROOT_DIR}/Models/SD/stable-diffusion-3.5-large}"
PROMPTS_CSV="${PROMPTS_CSV:-${ROOT_DIR}/Datasets/i2p_benchmark.csv}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT_DIR}/Results}"
SAE_ROOT="${SAE_ROOT:-${ROOT_DIR}/SAEs/sd35_text_encoder_i2p_no_sexual_activations}"
STEPS=${STEPS:-30}
GUIDANCE=${GUIDANCE:-7.0}
SAE_STRENGTH=${SAE_STRENGTH:-1.5}
VERIFY_N=${VERIFY_N:-1}
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
LOG_FILE="${LOG_DIR}/generate_sd35_image_i2p47030_${TS}.log"

{
  echo "timestamp=${TS}"
  echo "sd35_ckpt=${SD35_CKPT}"
  echo "prompts_csv=${PROMPTS_CSV}"
  echo "save_root=${SAVE_ROOT}"
  echo "sae_root=${SAE_ROOT}"
  echo "steps=${STEPS} guidance=${GUIDANCE} num_images=${NUM_IMAGES} batch_size=${BATCH_SIZE}"
  echo "sae_strength=${SAE_STRENGTH} verify_n=${VERIFY_N}"
  echo "height=${HEIGHT} width=${WIDTH} seed_base=${SEED_BASE} dtype=${DTYPE}"
  echo "negative_prompt=${NEG_PROMPT}"

  python "${ROOT_DIR}/Scripts/generate/generate_SD35_image_i2p_47030_textencoder.py" \
    --sd35-ckpt "${SD35_CKPT}" \
    --prompts-csv "${PROMPTS_CSV}" \
    --save-root "${SAVE_ROOT}" \
    --sae-root "${SAE_ROOT}" \
    --steps "${STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --sae-strength "${SAE_STRENGTH}" \
    --verify-n "${VERIFY_N}" \
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
