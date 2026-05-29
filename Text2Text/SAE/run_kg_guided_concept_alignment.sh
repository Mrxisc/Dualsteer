#!/usr/bin/env bash
set -euo pipefail
SAE_DIR="${DUALSTEER_ROOT}/Text2Text/SAE"
ACT_DIR="$SAE_DIR/activation/qwen3_sae_train"
SAE_ROOT="$SAE_DIR/SAEs"
KG_DIR="${DUALSTEER_ROOT}/Text2Text/KG/outputs"
OUTPUT="$SAE_ROOT/qwen3_kg_sae_concept_alignment.json"
if [[ ! -f "$KG_DIR/nodes.csv" || ! -f "$KG_DIR/source_prompts.csv" ]]; then
  echo "[run_kg_guided_concept_alignment.sh] KG outputs not found. Build KG first."
  exit 1
fi
if [[ ! -d "$ACT_DIR" ]]; then
  echo "[run_kg_guided_concept_alignment.sh] SAE activations not found. Collect activations first."
  exit 1
fi
echo "[run_kg_guided_concept_alignment.sh] KG-guided SAE concept alignment — $(date '+%Y-%m-%d %H:%M:%S')"
conda run -n backdoorllm python "$SAE_DIR/kg_guided_concept_alignment.py" \
  --activation-dir "$ACT_DIR" \
  --sae-root "$SAE_ROOT" \
  --kg-dir "$KG_DIR" \
  --output "$OUTPUT" \
  --layers "30,32,33,34,35" \
  --min-risk 0.5 \
  --min-frequency 2 \
  --min-concept-samples 3 \
  --top-k-features 32 \
  --cuda 0
echo "[run_kg_guided_concept_alignment.sh] Saved: $OUTPUT"
