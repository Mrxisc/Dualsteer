#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(realpath "$(dirname "$0")")
TEXT2IMAGE_DIR=$(realpath "${SCRIPT_DIR}/../..")
REPO_ROOT="${TEXT2IMAGE_DIR}"
WORKSPACE_ROOT=$(realpath "${TEXT2IMAGE_DIR}/..")
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_DIR=${MODEL_DIR:-${TEXT2IMAGE_DIR}/Models/FLUX.1-dev}
SEXUAL_CSV=${SEXUAL_CSV:-${TEXT2IMAGE_DIR}/Datasets/i2p_sexual.csv}
HOOK_NAME=${HOOK_NAME:-transformer.single_transformer_blocks.37}
BASE_SAE_DIR=${BASE_SAE_DIR:-${TEXT2IMAGE_DIR}/SAEs/flux_real_singleblock37_i2p_no_sexual_i2p_no_sexual_flux_real}
TARGET_ACTIVATIONS_DIR=${TARGET_ACTIVATIONS_DIR:-${TEXT2IMAGE_DIR}/Activations/i2p_sexual_flux_real}
REPLAY_ACTIVATIONS_DIR=${REPLAY_ACTIVATIONS_DIR:-${TEXT2IMAGE_DIR}/Activations/i2p_no_sexual_flux_real}
OUTPUT_SAE_DIR=${OUTPUT_SAE_DIR:-${TEXT2IMAGE_DIR}/SAEs/flux_real_singleblock37_sexual_transfer}

LOG_DIR=${LOG_DIR:-${TEXT2IMAGE_DIR}/Logs}
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/transfer_flux_singleblock_real_${TIMESTAMP}.log"

CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-1}
CACHE_STEPS=${CACHE_STEPS:-30}
CACHE_GUIDANCE_SCALE=${CACHE_GUIDANCE_SCALE:-4.0}
CACHE_EVERY_N=${CACHE_EVERY_N:-6}
CACHE_DTYPE=${CACHE_DTYPE:-float16}
CACHE_RESUME=${CACHE_RESUME:-1}
FORCE_CACHE=${FORCE_CACHE:-0}

CHANNEL_PROJ_PATH=${CHANNEL_PROJ_PATH:-${REPLAY_ACTIVATIONS_DIR}/channel_proj.pt}
CHANNEL_PROJ_DIM=${CHANNEL_PROJ_DIM:-1024}
TOKEN_SUBSAMPLE=${TOKEN_SUBSAMPLE:-256}

TRAIN_DECODER_ONLY=${TRAIN_DECODER_ONLY:-1}
REPLAY_RATIO=${REPLAY_RATIO:-0.2}
MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-64}
MAX_REPLAY_EXAMPLES=${MAX_REPLAY_EXAMPLES:-256}

NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-5e-5}
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE:-256}
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

[[ -d "${MODEL_DIR}" ]] || die "MODEL_DIR not found: ${MODEL_DIR}"
[[ -f "${SEXUAL_CSV}" ]] || die "SEXUAL_CSV not found: ${SEXUAL_CSV}"
[[ -d "${BASE_SAE_DIR}/${HOOK_NAME}" ]] || die "Missing base SAE checkpoint: ${BASE_SAE_DIR}/${HOOK_NAME}"
[[ -d "${REPLAY_ACTIVATIONS_DIR}/${HOOK_NAME}" ]] || die "Missing replay activations: ${REPLAY_ACTIVATIONS_DIR}/${HOOK_NAME}"
[[ -f "${CHANNEL_PROJ_PATH}" ]] || die "Missing channel projection file: ${CHANNEL_PROJ_PATH}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

hook_cache_ready() {
	local base="$1"
	local hook="$2"
	[[ -f "${base}/${hook}/dataset_info.json" ]] && [[ -f "${base}/${hook}/state.json" ]]
}

{
	echo "=== Flux REAL SingleBlock SAE Transfer ==="
	echo "timestamp=${TIMESTAMP}"
	echo "hook=${HOOK_NAME}"
	echo "model_dir=${MODEL_DIR}"
	echo "source_base_sae_dir=${BASE_SAE_DIR}"
	echo "target_csv=${SEXUAL_CSV}"
	echo "target_activations_dir=${TARGET_ACTIVATIONS_DIR}"
	echo "replay_activations_dir=${REPLAY_ACTIVATIONS_DIR}"
	echo "output_sae_dir=${OUTPUT_SAE_DIR}"
	echo "cache_batch_size=${CACHE_BATCH_SIZE} cache_steps=${CACHE_STEPS} cache_guidance_scale=${CACHE_GUIDANCE_SCALE} cache_every_n=${CACHE_EVERY_N} cache_dtype=${CACHE_DTYPE} resume=${CACHE_RESUME} force_cache=${FORCE_CACHE}"
	echo "token_subsample=${TOKEN_SUBSAMPLE} channel_proj_dim=${CHANNEL_PROJ_DIM} channel_proj_path=${CHANNEL_PROJ_PATH}"
	echo "train_decoder_only=${TRAIN_DECODER_ONLY} replay_ratio=${REPLAY_RATIO} max_train_examples=${MAX_TRAIN_EXAMPLES} max_replay_examples=${MAX_REPLAY_EXAMPLES}"
	echo "num_epochs=${NUM_EPOCHS} lr=${LR} effective_batch_size=${EFFECTIVE_BATCH_SIZE}"
	echo "device=${DEVICE} train_dtype=${TRAIN_DTYPE} seed=${SEED}"
	echo "python=${PYTHON_BIN}"

	if [[ "${FORCE_CACHE}" == "1" ]] || ! hook_cache_ready "${TARGET_ACTIVATIONS_DIR}" "${HOOK_NAME}"; then
		echo "[cache] Target activations missing/incomplete; collecting with real diffusers backend..."
		CACHE_ARGS=(
			"--backend" "flux"
			"--model_name" "${MODEL_DIR}"
			"--hook_names" "${HOOK_NAME}"
			"--new_cached_activations_path" "${TARGET_ACTIVATIONS_DIR}"
			"--csv_path" "${SEXUAL_CSV}"
			"--csv_prompt_column" "prompt"
			"--csv_category_column" "categories"
			"--csv_filter_categories" "sexual"
			"--seed" "${SEED}"
			"--batch_size" "${CACHE_BATCH_SIZE}"
			"--num_inference_steps" "${CACHE_STEPS}"
			"--guidance_scale" "${CACHE_GUIDANCE_SCALE}"
			"--height" "256"
			"--width" "256"
			"--cache_every_n_timesteps" "${CACHE_EVERY_N}"
			"--token_subsample" "${TOKEN_SUBSAMPLE}"
			"--channel_proj_dim" "${CHANNEL_PROJ_DIM}"
			"--channel_proj_path" "${CHANNEL_PROJ_PATH}"
			"--activation_part" "tuple1"
			"--cfg_keep" "cond"
			"--dtype" "${CACHE_DTYPE}"
		)
		if [[ "${CACHE_RESUME}" == "1" ]]; then
			CACHE_ARGS+=("--resume" "true")
		fi
		"${PYTHON_BIN}" "${REPO_ROOT}/Scripts/collect/collect_real_activations_diffusers.py" "${CACHE_ARGS[@]}"
	else
		echo "[cache] Using cached activations in ${TARGET_ACTIVATIONS_DIR}"
	fi

	echo "=== Running finetune ==="
	FINETUNE_ARGS=(
		"--base-sae-dir" "${BASE_SAE_DIR}"
		"--hookpoints" "${HOOK_NAME}"
		"--model-name" "${MODEL_DIR}"
		"--sexual-csv" "${SEXUAL_CSV}"
		"--csv-filter" "sexual"
		"--activations-dir" "${TARGET_ACTIVATIONS_DIR}"
		"--output-sae-dir" "${OUTPUT_SAE_DIR}"
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
	if [[ "${TRAIN_DECODER_ONLY}" == "1" ]]; then
		FINETUNE_ARGS+=("--train-decoder-only")
	fi
	if [[ "${REPLAY_RATIO}" != "0" ]]; then
		FINETUNE_ARGS+=(
			"--replay-activations-dir" "${REPLAY_ACTIVATIONS_DIR}"
			"--replay-ratio" "${REPLAY_RATIO}"
			"--max-replay-examples" "${MAX_REPLAY_EXAMPLES}"
		)
	fi

	"${PYTHON_BIN}" "${TEXT2IMAGE_DIR}/Scripts/transfer/fewshot_textencoder_sexual_finetune.py" "${FINETUNE_ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"

echo "Logs: ${LOG_FILE}"

