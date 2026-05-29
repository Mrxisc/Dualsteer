#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_topk_sae.py"
ACT_DIR="$SCRIPT_DIR/activation/qwen3_sae_train"
SAVE_ROOT="$SCRIPT_DIR/SAEs"
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate PRISM
echo "[run_train_qwen3_topk_sae.sh] Starting — $(date '+%Y-%m-%d %H:%M:%S')"
python "$TRAIN_SCRIPT" \
    --activation-dir "$ACT_DIR" \
    --save-root "$SAVE_ROOT" \
    --layers 30,32,33,34,35 \
    --expansion-factor 8 \
    --k 32 \
    --batch-size 256 \
    --epochs 80 \
    --lr 3e-4 \
    --cuda 0 \
    --seed 42
echo "[run_train_qwen3_topk_sae.sh] Done — $(date '+%Y-%m-%d %H:%M:%S')"
