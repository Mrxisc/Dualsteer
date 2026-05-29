#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAJA_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$DAJA_ROOT/test/eval_xguard_asr.py"
XGUARD_PATH="${XGUARD_MODEL_PATH:-${PROJECT_ROOT}/Models/YuFeng-XGuard-Reason-8B}"
RESULT_DIR="$DAJA_ROOT/result/Qwen3-8B_dualsteer"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_xguard_asr_dualsteer.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$TEST_SCRIPT" \
    --xguard-path "$XGUARD_PATH" \
    --result-files \
        "$RESULT_DIR/JBB-harmful.jsonl" \
        "$RESULT_DIR/HarmBench.jsonl" \
        "$RESULT_DIR/StrongREJECT.jsonl" \
        "$RESULT_DIR/WildJailbreak.jsonl" \
    --threshold 0.5 \
    --cuda 1
echo "[run_xguard_asr_dualsteer.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
