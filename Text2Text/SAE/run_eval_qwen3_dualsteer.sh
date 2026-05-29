#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_qwen3_dualsteer.py"
MODEL_PATH="${QWEN3_MODEL_PATH}"
SAE_ROOT="$SCRIPT_DIR/SAEs"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_eval_qwen3_dualsteer.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$EVAL_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --sae-root "$SAE_ROOT" \
    --layers 30 \
    --alpha 0.2 \
    --datasets JBB-harmful HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200 \
    --cuda 1
echo "[run_eval_qwen3_dualsteer.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
