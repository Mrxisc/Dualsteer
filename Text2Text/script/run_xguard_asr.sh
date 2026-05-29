#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$DAJA_ROOT/test/eval_xguard_asr.py"
XGUARD_PATH="${XGUARD_MODEL_PATH:-${PROJECT_ROOT}/Models/YuFeng-XGuard-Reason-8B}"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_xguard_asr.sh] Starting XGuard ASR evaluation — $(date '+%Y-%m-%d %H:%M:%S')"
python "$TEST_SCRIPT" \
    --xguard-path "$XGUARD_PATH" \
    --threshold 0.5 \
    --cuda 1
echo "[run_xguard_asr.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
