# DualSteer

DualSteer is a safety steering framework for reliable generative Transformer models. It combines explicit safety knowledge modeling with implicit sparse feature steering to support both text-to-text jailbreak defense and text-to-image safety control.

The framework contains three main components:

- **Safety Knowledge Graph**: organizes and expands safety concepts from jailbreak prompts, harmful categories, and rule-based risk signals.
- **Graph-Guided Sparse Concept Alignment**: aligns explicit KG concepts with sparse autoencoder features learned from model activations.
- **Dual-Distribution Safety Control**: combines input-level screening with activation-level steering to reduce unsafe generation while preserving utility.

## 1) Repository Structure

```text
Dualsteer_Code/
├── Text2Text/        # Text-to-text jailbreak defense workflow
├── Text2Image/      # Text-to-image safety steering workflow
├── Datasets/         # Place datasets or symbolic links here
├── Models/           # Place model checkpoints or symbolic links here
├── Activations/      # Generated activation caches
├── SAEs/             # Trained SAE checkpoints and selected features
├── Results/          # Generated outputs and evaluation results
└── Logs/             # Runtime logs
```

## 2) Environment Preparation

Install the dependencies required by your selected workflow. The text-to-image workflow provides a reference dependency file:

```bash
pip install -r Text2Image/requirements.txt
```

For text-to-text experiments, install common LLM evaluation dependencies such as:

```bash
pip install torch transformers accelerate pandas tqdm scikit-learn sentence-transformers matplotlib seaborn
```

Additional packages may be needed depending on the selected benchmark, detector, or model backend.

## 3) Model and Data Preparation

Prepare local checkpoints and datasets before running experiments. The following resources are commonly used:

- Qwen3-8B
- DeepSeek-R1-Distill-Llama-8B
- YuFeng-XGuard-Reason-8B
- JailbreakBench / JBB-Harmful
- HarmBench
- StrongREJECT
- WildJailbreak
- I2P benchmark
- Q16 safety classifier
- NudeNet detector
- FLUX.1-dev
- Stable Diffusion 3.5 Large

You can place these resources under `Models/` and `Datasets/`, or configure paths through environment variables:

```bash
export DUALSTEER_ROOT=/path/to/Dualsteer_Code
export PROJECT_ROOT=/path/to/Dualsteer_Code
export CONDA_ROOT=/path/to/conda
export QWEN3_MODEL_PATH=/path/to/Qwen3-8B
export DEEPSEEK_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Llama-8B
export XGUARD_MODEL_PATH=/path/to/YuFeng-XGuard-Reason-8B
```

Generated activations, SAE checkpoints, logs, and evaluation outputs are written to the corresponding workflow directories and can be redirected through script arguments or environment variables.

## 4) Text-to-Text Workflow

The text-to-text workflow is organized under `Text2Text/`.

### Build and apply the safety KG

```bash
bash Text2Text/KG/run_build_jailbreak_safety_kg.sh
bash Text2Text/KG/run_qwen3_kg_filter.sh
bash Text2Text/KG/run_deepseek_kg_filter.sh
```

### Collect activations and train SAE

```bash
bash Text2Text/SAE/run_collect_qwen3_sae_activations.sh
bash Text2Text/SAE/run_train_qwen3_topk_sae.sh
```

### Align KG concepts to SAE features

```bash
bash Text2Text/SAE/run_kg_guided_concept_alignment.sh
bash Text2Text/SAE/run_select_qwen3_harmful_features.sh
```

### Evaluate feature-level steering

```bash
bash Text2Text/SAE/run_eval_qwen3_feature_suppression.sh
bash Text2Text/SAE/run_eval_qwen3_dualsteer.sh
```

### Baseline and metric scripts

```bash
bash Text2Text/script/run_qwen3.sh
bash Text2Text/script/run_deepseek.sh
bash Text2Text/script/run_xguard_asr.sh
bash Text2Text/script/run_text_quality_metrics.sh
```

The benchmark adapters under `Text2Text/dataset/` provide auxiliary loaders and evaluation utilities for HarmBench, StrongREJECT, and WildJailbreak-style experiments. Text-to-text evaluation uses ASR, refusal behavior, and XGuard-style safety classification.

## 5) Text-to-Image Workflow

The text-to-image workflow is organized under `Text2Image/`.

### Activation collection

```bash
bash Text2Image/Run_scripts/collect/collect_i2p_no_sexual_flux_textencoder.sh
bash Text2Image/Run_scripts/collect/collect_i2p_no_sexual_sd35_textencoder.sh
```

### SAE training

```bash
bash Text2Image/Run_scripts/train/train_i2p_no_sexual_flux_text_encoder_sae.sh
bash Text2Image/Run_scripts/train/train_i2p_no_sexual_sd35_text_encoder_sae.sh
```

### Knowledge transfer and steering

```bash
bash Text2Image/Run_scripts/transfer/transfer_flux_textencoder.sh
bash Text2Image/Run_scripts/transfer/transfer_sd35_textencoder.sh
```

### Generation

```bash
bash Text2Image/Run_scripts/generate/generate_flux_image_i2p47030_textencoder.sh
bash Text2Image/Run_scripts/generate/generate_SD35_image_i2p47030_textencoder.sh
```

The corresponding Python implementations are under `Text2Image/Scripts/collect`, `Text2Image/Scripts/train`, `Text2Image/Scripts/transfer`, and `Text2Image/Scripts/generate`. Text-to-image evaluation scripts and model wrappers are provided under `Text2Image/Eval/`.
