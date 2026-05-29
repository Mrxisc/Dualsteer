#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$DAJA_ROOT/test/eval_no_defense.py"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_qwen3.sh] Qwen3-8B × all datasets — $(date '+%Y-%m-%d %H:%M:%S')"
python "$TEST_SCRIPT" \
    --models Qwen3-8B \
    --datasets JBB-harmful JBB-benign HarmBench StrongREJECT WildJailbreak \
    --wj-n-per-label 200
echo "[run_qwen3.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
