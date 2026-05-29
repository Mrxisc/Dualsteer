#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
PY_SCRIPT="$SCRIPT_DIR/eval_text_quality_metrics.py"
RESULT_ROOT="$DAJA_ROOT/result"
OUTPUT_ROOT="$DAJA_ROOT/result_text_metrics"
MODEL_PATH="${QWEN3_MODEL_PATH}"
CUDA_DEVICE="1"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate backdoorllm || conda activate PRISM
echo "[run_text_quality_metrics.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Result root: $RESULT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "PPL model  : $MODEL_PATH"
echo "CUDA       : $CUDA_DEVICE"
python "$PY_SCRIPT" \
    --result-root "$RESULT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --model-path "$MODEL_PATH" \
    --cuda "$CUDA_DEVICE" \
    --max-length 4096
echo "[run_text_quality_metrics.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary: $OUTPUT_ROOT/summary.csv"
