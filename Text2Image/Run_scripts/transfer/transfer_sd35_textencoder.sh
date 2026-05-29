#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(realpath "$(dirname "$0")")
TEXT2IMAGE_DIR=$(realpath "${SCRIPT_DIR}/../..")
REPO_ROOT="${TEXT2IMAGE_DIR}"
WORKSPACE_ROOT=$(realpath "${TEXT2IMAGE_DIR}/..")
PYTHON_BIN=${PYTHON_BIN:-python}
SD35_CKPT=${SD35_CKPT:-${TEXT2IMAGE_DIR}/Models/SD/stable-diffusion-3.5-large}
DIT_CONFIG=${DIT_CONFIG:-${TEXT2IMAGE_DIR}/_Backup_code/configs/sd35_large.json}
SEXUAL_CSV=${SEXUAL_CSV:-${TEXT2IMAGE_DIR}/Datasets/i2p_sexual.csv}
HOOK_NAME=${HOOK_NAME:-text_encoder}
BASE_SAE_DIR=${BASE_SAE_DIR:-${TEXT2IMAGE_DIR}/SAEs/sd35_text_encoder_i2p_no_sexual_activations}


TARGET_ACTIVATIONS_DIR=${TARGET_ACTIVATIONS_DIR:-${TEXT2IMAGE_DIR}/Activations/i2p_sexual_SD}
REPLAY_ACTIVATIONS_DIR=${REPLAY_ACTIVATIONS_DIR:-${TEXT2IMAGE_DIR}/Activations/i2p_no_sexual_SD}
OUTPUT_SAE_DIR=${OUTPUT_SAE_DIR:-${TEXT2IMAGE_DIR}/SAEs/sd35_text_encoder_sexual_transfer}

LOG_DIR=${LOG_DIR:-${TEXT2IMAGE_DIR}/Logs}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/transfer_sd35_textencoder_${TIMESTAMP}.log"
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-1}
CACHE_STEPS=${CACHE_STEPS:-30}
CACHE_GUIDANCE_SCALE=${CACHE_GUIDANCE_SCALE:-7.0}
CACHE_EVERY_N=${CACHE_EVERY_N:-1}
CACHE_DTYPE=${CACHE_DTYPE:-fp16}
FORCE_CACHE=${FORCE_CACHE:-0}

TRAIN_DECODER_ONLY=${TRAIN_DECODER_ONLY:-1}
REPLAY_RATIO=${REPLAY_RATIO:-0.2}
MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-256}
MAX_REPLAY_EXAMPLES=${MAX_REPLAY_EXAMPLES:-4096}

NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-5e-5}
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE:-1024}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
MICRO_ACC_STEPS=${MICRO_ACC_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
TRAIN_DTYPE=${TRAIN_DTYPE:-fp16}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}

die() {
	echo "[error] $*" >&2
	exit 1
}

[[ -d "${SD35_CKPT}" ]] || die "SD35_CKPT not found: ${SD35_CKPT}"
[[ -f "${DIT_CONFIG}" ]] || die "DIT_CONFIG not found: ${DIT_CONFIG}"
[[ -f "${SEXUAL_CSV}" ]] || die "SEXUAL_CSV not found: ${SEXUAL_CSV}"
[[ -d "${BASE_SAE_DIR}/${HOOK_NAME}" ]] || die "Missing base SAE checkpoint: ${BASE_SAE_DIR}/${HOOK_NAME}"
[[ -d "${REPLAY_ACTIVATIONS_DIR}/${HOOK_NAME}" ]] || die "Missing replay activations: ${REPLAY_ACTIVATIONS_DIR}/${HOOK_NAME}"

ARGS=(
	"--base-sae-dir" "${BASE_SAE_DIR}"
	"--hookpoints" "${HOOK_NAME}"
	"--model-name" "${SD35_CKPT}"
	"--sexual-csv" "${SEXUAL_CSV}"
	"--csv-filter" "sexual"
	"--activations-dir" "${TARGET_ACTIVATIONS_DIR}"
	"--output-sae-dir" "${OUTPUT_SAE_DIR}"
	"--use-dit"
	"--dit-config" "${DIT_CONFIG}"
	"--cache-batch-size" "${CACHE_BATCH_SIZE}"
	"--cache-steps" "${CACHE_STEPS}"
	"--cache-guidance-scale" "${CACHE_GUIDANCE_SCALE}"
	"--cache-every-n" "${CACHE_EVERY_N}"
	"--cache-dtype" "${CACHE_DTYPE}"
	"--effective-batch-size" "${EFFECTIVE_BATCH_SIZE}"
	"--num-epochs" "${NUM_EPOCHS}"
	"--lr" "${LR}"
	"--grad-acc-steps" "${GRAD_ACC_STEPS}"
	"--micro-acc-steps" "${MICRO_ACC_STEPS}"
	"--num-workers" "${NUM_WORKERS}"
	"--train-dtype" "${TRAIN_DTYPE}"
	"--device" "${DEVICE}"
	"--seed" "${SEED}"
	"--max-train-examples" "${MAX_TRAIN_EXAMPLES}"
)

if [[ "${FORCE_CACHE}" == "1" ]]; then
	ARGS+=("--force-cache")
fi

if [[ "${TRAIN_DECODER_ONLY}" == "1" ]]; then
	ARGS+=("--train-decoder-only")
fi

if [[ "${REPLAY_RATIO}" != "0" ]]; then
	ARGS+=(
		"--replay-activations-dir" "${REPLAY_ACTIVATIONS_DIR}"
		"--replay-ratio" "${REPLAY_RATIO}"
		"--max-replay-examples" "${MAX_REPLAY_EXAMPLES}"
	)
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

{
	echo "=== SD3.5 Text-Encoder SAE Transfer ==="
	echo "timestamp=${TIMESTAMP}"
	echo "hook=${HOOK_NAME}"
	echo "sd35_ckpt=${SD35_CKPT}"
	echo "dit_config=${DIT_CONFIG}"
	echo "source_base_sae_dir=${BASE_SAE_DIR}"
	echo "target_csv=${SEXUAL_CSV}"
	echo "target_activations_dir=${TARGET_ACTIVATIONS_DIR}"
	echo "replay_activations_dir=${REPLAY_ACTIVATIONS_DIR}"
	echo "output_sae_dir=${OUTPUT_SAE_DIR}"
	echo "train_decoder_only=${TRAIN_DECODER_ONLY}"
	echo "replay_ratio=${REPLAY_RATIO}"
	echo "max_train_examples=${MAX_TRAIN_EXAMPLES}"
	echo "max_replay_examples=${MAX_REPLAY_EXAMPLES}"
	echo "cache_batch_size=${CACHE_BATCH_SIZE} cache_steps=${CACHE_STEPS} cache_guidance_scale=${CACHE_GUIDANCE_SCALE} cache_every_n=${CACHE_EVERY_N} cache_dtype=${CACHE_DTYPE}"
	echo "num_epochs=${NUM_EPOCHS} lr=${LR} effective_batch_size=${EFFECTIVE_BATCH_SIZE}"
	echo "device=${DEVICE} train_dtype=${TRAIN_DTYPE} seed=${SEED}"
	echo "python=${PYTHON_BIN}"
	echo "=== Running finetune ==="
	"${PYTHON_BIN}" "${TEXT2IMAGE_DIR}/Scripts/transfer/fewshot_textencoder_sexual_finetune.py" "${ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}
