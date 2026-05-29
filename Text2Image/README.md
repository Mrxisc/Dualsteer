# Toward Safety Generalization for Text-to-Image Generation via Robust Steering

This repository contains the DualSteer text-to-image workflow used in the accompanying paper.

## 1) 📦 Recommended model/data preparation

Recommended resources:

- 🔗 I2P benchmark
- 🔗 Q16 safety classifier
- 🔗 NudeNet detector
- 🔗 MMA dataset
- 🔗 MM-SafetyBench dataset
- 🤗 Stable Diffusion 3.5 Large
- 🤗 FLUX.1-dev

These are **recommended** defaults. You can replace datasets, checkpoints, and safety tools according to your own needs.

## 2) 🛡️ Optional pre-interpretion module (`Dualsteer_Code/Text2Image/Utils`)

`Dualsteer_Code/Text2Image/Utils` is an optional defense module before running the main DualSteer pipeline.

You can use or extend it with strategies such as:

- harmful keyword filtering,
- LLM-based prompt rewriting / dataset reconstruction,
- distance-metric-based sample screening,
- or your own custom pre-interception logic.

Current utility scripts include examples of these directions and are intended as plug-in style preprocessing.

## 3) 🚀 DualSteer pipeline entry points (`Dualsteer_Code/Text2Image/Run_scripts`)

The main runnable entry points are under `Dualsteer_Code/Text2Image/Run_scripts`.
By default, text-encoder series scripts are ready to run directly (with repository default parameters), for example:

- `Run_scripts/collect/collect_i2p_no_sexual_flux_textencoder.sh`
- `Run_scripts/train/train_i2p_no_sexual_flux_text_encoder_sae.sh`
- `Run_scripts/transfer/transfer_flux_textencoder.sh`
- `Run_scripts/generate/generate_flux_image_i2p47030_textencoder.sh`

The corresponding Python implementations are organized under `Dualsteer_Code/Text2Image/Scripts` (collect/train/transfer/generate).

Key Python entries:

- `Scripts/collect/cache_activations_runner.py`
- `Scripts/collect/collect_activations.py`
- `Scripts/transfer/fewshot_textencoder_sexual_finetune.py`

### 🧭 Path convention (default)

- 📥 Inputs: `Dualsteer_Code/Text2Image/Datasets`, `Dualsteer_Code/Text2Image/Models`
- 📤 Outputs: `Dualsteer_Code/Text2Image/Activations`, `Dualsteer_Code/Text2Image/SAEs`, `Dualsteer_Code/Text2Image/Results`
- 📝 Logs: `Dualsteer_Code/Text2Image/Logs`

If needed, these defaults can still be overridden by environment variables / CLI arguments.

## 4) 📊 Evaluation

DualSteer evaluation is based on the I2P benchmark:

- I2P link: ...

Batch image safety evaluation is provided by:

- `Dualsteer_Code/Text2Image/Eval/eval_I2P_ONLY_eval.py`

For convenient benchmarking, model wrappers are prepared under:

- `Dualsteer_Code/Text2Image/Eval/model_in_i2p`

## 5) ⚠️ Important notes

- The model backbone loading implementations under `Dualsteer_Code/Text2Image/_Backup_code` are **fallback constructed adapters**.
  If you can directly hook into the real model modules in your runtime environment, prefer real hooks and do not rely on these backup adapters.
- 🧩 Config files under `Dualsteer_Code/Text2Image/_Backup_code/configs` are editable templates; you can modify existing configs or add new ones as needed.
- 💾 This project enables disk-overhead mitigation strategies by default. Please check and reserve enough storage space before running large-scale cache/train/generate pipelines.
- 🛠️ The codebase includes built-in resume, logging, `--help`, and error-handling mechanisms. Please use these interfaces correctly in scripts and command-line execution.

## 6) 🙏 Acknowledgements

Special thanks to the SAeUron project for SAE code contributions and theoretical foundations.
