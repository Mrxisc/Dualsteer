#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_qwen3_feature_suppression.py"
MODEL_PATH="${QWEN3_MODEL_PATH}"
SAE_ROOT="$SCRIPT_DIR/SAEs"
FEATURE_FILE="$SAE_ROOT/qwen3_harmful_features.json"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
export CUDA_VISIBLE_DEVICES=1
echo "[run_eval_qwen3_feature_suppression.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$EVAL_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --sae-root "$SAE_ROOT" \
    --feature-file "$FEATURE_FILE" \
    --layers 30 \
    --features-per-layer 64 \
    --suppression-strength 1.0 \
    --alpha 1.0 \
    --datasets JBB-harmful HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200
echo "[run_eval_qwen3_feature_suppression.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
