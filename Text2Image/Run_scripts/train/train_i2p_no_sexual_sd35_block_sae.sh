set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
TEXT2IMAGE_DIR=$(realpath "${SCRIPT_DIR}/../..")
REPO_ROOT="${TEXT2IMAGE_DIR}"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-${TEXT2IMAGE_DIR}/Activations/i2p_no_sexual_SD}
HOOK_NAME=${HOOK_NAME:-transformer.transformer_blocks.36}
RUN_NAME=${RUN_NAME:-sd35_block36_i2p_no_sexual}
WANDB_PROJECT=${WANDB_PROJECT:-dualsteer_sd35_i2p}
LOG_DIR=${LOG_DIR:-${TEXT2IMAGE_DIR}/Logs}

if [[ ! -d "${DATASET_ROOT}/${HOOK_NAME}" ]]; then
  echo "[error] Missing activations for hook '${HOOK_NAME}' under ${DATASET_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_sd35_${HOOK_NAME//./_}_${TIMESTAMP}.log"

ARGS=(
  "--dataset_path" "${DATASET_ROOT}"
  "--hookpoints" "${HOOK_NAME}"
  "--effective_batch_size" "${EFFECTIVE_BATCH_SIZE:-4096}"
  "--num_epochs" "${NUM_EPOCHS:-5}"
  "--lr" "${LR:-4e-4}"
  "--lr_scheduler" "${LR_SCHEDULER:-linear}"
  "--lr_warmup_steps" "${LR_WARMUP_STEPS:-0}"
  "--auxk_alpha" "${AUXK_ALPHA:-0.03125}"
  "--dead_feature_threshold" "${DEAD_FEATURE_THRESHOLD:-10000000}"
  "--grad_acc_steps" "${GRAD_ACC_STEPS:-1}"
  "--micro_acc_steps" "${MICRO_ACC_STEPS:-1}"
  "--num_workers" "${NUM_WORKERS:-4}"
  "--wandb_project" "${WANDB_PROJECT}"
  "--wandb_log_frequency" "${WANDB_LOG_FREQUENCY:-2000}"
  "--run_name" "${RUN_NAME}"
  "--save_every" "${SAVE_EVERY:-0}"
  "--mixed_precision" "${MIXED_PRECISION:-fp16}"
  "--device" "${DEVICE:-cuda}"
  "--seed" "${SEED:-42}"
  "--expansion_factor" "${EXPANSION_FACTOR:-16}"
  "--k" "${TOPK:-32}"
  "--batch_topk" "${BATCH_TOPK:-True}"
  "--multi_topk" "${MULTI_TOPK:-False}"
  "--log_to_wandb" "False"
  "--save_dir" "${TEXT2IMAGE_DIR}/SAEs"
)

if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  ARGS+=("--max_examples" "${MAX_EXAMPLES}")
fi

{
  echo "=== SAE Training (SD3.5 transformer block) ==="
  echo "timestamp=${TIMESTAMP}"
  echo "hook=${HOOK_NAME}"
  echo "dataset_path=${DATASET_ROOT}"
  echo "run_name=${RUN_NAME}"
  echo "save_dir=${TEXT2IMAGE_DIR}/SAEs"
  echo "effective_batch_size=${EFFECTIVE_BATCH_SIZE:-4096}"
  echo "num_epochs=${NUM_EPOCHS:-5}"
  echo "lr=${LR:-4e-4}"
  echo "lr_scheduler=${LR_SCHEDULER:-linear}"
  echo "lr_warmup_steps=${LR_WARMUP_STEPS:-0}"
  echo "auxk_alpha=${AUXK_ALPHA:-0.03125}"
  echo "dead_feature_threshold=${DEAD_FEATURE_THRESHOLD:-10000000}"
  echo "grad_acc_steps=${GRAD_ACC_STEPS:-1}"
  echo "micro_acc_steps=${MICRO_ACC_STEPS:-1}"
  echo "num_workers=${NUM_WORKERS:-4}"
  echo "mixed_precision=${MIXED_PRECISION:-fp16}"
  echo "expansion_factor=${EXPANSION_FACTOR:-16}"
  echo "k=${TOPK:-32}"
  echo "batch_topk=${BATCH_TOPK:-True}"
  echo "multi_topk=${MULTI_TOPK:-False}"
  echo "seed=${SEED:-42}"
  echo "device=${DEVICE:-cuda}"
  echo "wandb_project=${WANDB_PROJECT} (log_to_wandb=False)"
  echo "max_examples=${MAX_EXAMPLES:-<all>}"
  echo "python=${PYTHON_BIN}"
  echo "=== Training output ==="
  "${PYTHON_BIN}" "${REPO_ROOT}/Scripts/train/train.py" "${ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}
