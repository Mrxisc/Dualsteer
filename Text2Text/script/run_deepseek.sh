#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$DAJA_ROOT/test/eval_no_defense.py"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_deepseek.sh] DeepSeek-R1-Distill-Llama-8B × all datasets — $(date '+%Y-%m-%d %H:%M:%S')"
python "$TEST_SCRIPT" \
    --models DeepSeek-R1-Distill-Llama-8B \
    --datasets JBB-harmful JBB-benign HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200
echo "[run_deepseek.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
