#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAN_SCRIPT="$SCRIPT_DIR/collect_layer_scan.py"
MODEL_PATH="${QWEN3_MODEL_PATH}"
OUTPUT_DIR="$SCRIPT_DIR/activation/qwen3_layer_scan"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_qwen3_layer_scan.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$SCAN_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --harmful-datasets JBB-harmful HarmBench \
    --benign-dataset JBB-benign \
    --max-harmful 200 \
    --max-benign 100 \
    --layers all \
    --token-position last \
    --max-length 2048 \
    --cuda 1 \
    --top-k 5
echo "[run_qwen3_layer_scan.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
