#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_sae_activations.py"
MODEL_PATH="${QWEN3_MODEL_PATH}"
OUTPUT_DIR="$SCRIPT_DIR/activation/qwen3_sae_train"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_collect_qwen3_sae_activations.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$COLLECT_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --datasets JBB-harmful HarmBench StrongREJECT JBB-benign \
    --layers 30,32,33,34,35 \
    --token-position last \
    --max-length 2048 \
    --shard-size 512 \
    --cuda 1 \
    --seed 42
echo "[run_collect_qwen3_sae_activations.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
