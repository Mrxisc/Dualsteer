#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$DAJA_ROOT/test/eval_no_defense.py"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_parallel.sh] Starting parallel evaluation — $(date '+%Y-%m-%d %H:%M:%S')"
CUDA_VISIBLE_DEVICES=0 python "$TEST_SCRIPT" \
    --models Qwen3-8B \
    --datasets JBB-harmful JBB-benign HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200 &
PID_QWEN=$!
echo "[run_parallel.sh] Qwen3-8B      → GPU 0  (PID=$PID_QWEN)"
CUDA_VISIBLE_DEVICES=1 python "$TEST_SCRIPT" \
    --models DeepSeek-R1-Distill-Llama-8B \
    --datasets JBB-harmful JBB-benign HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200 &
PID_DS=$!
echo "[run_parallel.sh] DeepSeek-R1   → GPU 1  (PID=$PID_DS)"
EXIT_QWEN=0
EXIT_DS=0
wait $PID_QWEN  || EXIT_QWEN=$?
echo "[run_parallel.sh] Qwen3-8B done (exit=$EXIT_QWEN) — $(date '+%Y-%m-%d %H:%M:%S')"
wait $PID_DS    || EXIT_DS=$?
echo "[run_parallel.sh] DeepSeek done (exit=$EXIT_DS)   — $(date '+%Y-%m-%d %H:%M:%S')"
echo "[run_parallel.sh] All done — $(date '+%Y-%m-%d %H:%M:%S')"
if [ $EXIT_QWEN -ne 0 ] || [ $EXIT_DS -ne 0 ]; then
    echo "[run_parallel.sh] WARNING: one or more processes exited with error."
    exit 1
fi
