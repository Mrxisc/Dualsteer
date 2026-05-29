#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECT_SCRIPT="$SCRIPT_DIR/select_harmful_features.py"
ACT_DIR="$SCRIPT_DIR/activation/qwen3_sae_train"
SAE_ROOT="$SCRIPT_DIR/SAEs"
OUTPUT="$SAE_ROOT/qwen3_harmful_features.json"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_select_qwen3_harmful_features.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$SELECT_SCRIPT" \
    --activation-dir "$ACT_DIR" \
    --sae-root "$SAE_ROOT" \
    --output "$OUTPUT" \
    --layers 30,32,33,34,35 \
    --top-k 128 \
    --min-score 0.0 \
    --cuda 0
echo "[run_select_qwen3_harmful_features.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
