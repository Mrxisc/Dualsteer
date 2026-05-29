#!/usr/bin/env bash
set -euo pipefail
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate backdoorllm
python ${DUALSTEER_ROOT}/Text2Text/KG/build_jailbreak_safety_kg.py \
  --datasets JBB-harmful JBB-benign HarmBench StrongREJECT WildJailbreak \
  --output-dir ${DUALSTEER_ROOT}/Text2Text/KG/outputs \
  --wildjailbreak-n-per-label 200 \
  --seed 42 \
  --top-k-phrases 600 \
  --min-df 2 \
  --semantic-threshold 0.78 \
  --lexical-threshold 0.34 \
  --top-semantic-edges 1500
