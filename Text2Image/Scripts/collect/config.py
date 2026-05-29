import json
from dataclasses import dataclass
from pathlib import Path
import torch
from simple_parsing import Serializable, list_field


@dataclass
class SaeConfig(Serializable):
    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32
    batch_topk: bool = False
    sample_topk: bool = False
    input_unit_norm: bool = False
    multi_topk: bool = False


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig
    dataset_path: list[str] = list_field()
    effective_batch_size: int = 4096
    num_workers: int = 1
    persistent_workers: bool = True
    prefetch_factor: int = 2
    grad_acc_steps: int = 1
    micro_acc_steps: int = 1
    lr: float | None = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 1000
    auxk_alpha: float = 0.0
    dead_feature_threshold: int = 10_000_000
    feature_sampling_window: int = 100
    hookpoints: list[str] = list_field()
    distribute_modules: bool = False
    save_every: int = 5000
    log_to_wandb: bool = True
    run_name: str | None = None
    #wandb_log_frequency: int = 1
    #wandb_project: str = "sae_stable-diffusion-v1-4"
    save_dir: str | None = None
    train_decoder_only: bool = False

    def __post_init__(self):
        if self.run_name is None:
            variant = "patch_topk"
            if self.sae.batch_topk:
                variant = "batch_topk"
            elif self.sae.sample_topk:
                variant = "sample_topk"
            self.run_name = f"{variant}_expansion_factor{self.sae.expansion_factor}_k{self.sae.k}_multi_topk{self.sae.multi_topk}_auxk_alpha{self.auxk_alpha}"


@dataclass
class CacheActivationsRunnerConfig:
    hook_names: list[str] | None = None
    new_cached_activations_path: str | None = None
    dataset_name: str = "guangyil/laion-coco-aesthetic"
    split: str = "train"
    column: str = "caption"
    device: torch.device | str = "cuda"
    model_name: str = "CompVis/stable-diffusion-v1-4"
    dtype: torch.dtype = torch.float16
    num_inference_steps: int = 50
    seed: int = 42
    batch_size: int = 100
    num_workers: int = 8
    output_or_diff: str = "output"
    max_num_examples: int | None = None
    cache_every_n_timesteps: int = 1
    guidance_scale: float = 9.0
    class_start: int = 0
    class_end: int = 20
    hf_repo_id: str | None = None
    hf_num_shards: int | None = None
    hf_revision: str = "main"
    hf_is_private_repo: bool = False
    csv_path: str | None = None
    csv_prompt_column: str = "prompt"
    csv_category_column: str | None = "categories"
    csv_filter_categories: list[str] = list_field()
    csv_category_match_all: bool = False
    csv_hard_column: str | None = "hard"
    csv_require_hard: bool = False
    csv_max_rows: int | None = None
    csv_strip_period: bool = True
    csv_deduplicate: bool = True
    use_dit: bool = False
    dit_config_path: str | None = None
    dit_config: dict | None = None
    use_flux: bool = False
    flux_config_path: str | None = None
    flux_config: dict | None = None
    use_sd35_backbone: bool = False
    sd35_config_path: str | None = None
    sd35_config: dict | None = None
    denoiser_attr: str | None = None
    resume: bool = False
    token_subsample: int | None = None
    token_subsample_seed: int = 0
    token_indices_path: str | None = None
    channel_proj_dim: int | None = None
    channel_proj_seed: int = 0
    channel_proj_path: str | None = None


    def __post_init__(self):
        if sum(bool(flag) for flag in (self.use_dit, self.use_flux, self.use_sd35_backbone)) > 1:
            raise ValueError("DiT, Flux, and SD3.5 backbones are mutually exclusive.")


        if self.dit_config is None:
            self.dit_config = _load_optional_json(self.dit_config_path)
        if self.flux_config is None:
            self.flux_config = _load_optional_json(self.flux_config_path)
        if self.sd35_config is None:
            self.sd35_config = _load_optional_json(self.sd35_config_path)
        if self.token_indices_path is not None:
            self.token_indices_path = _expand_optional_path(self.token_indices_path)
        if self.channel_proj_path is not None:
            self.channel_proj_path = _expand_optional_path(self.channel_proj_path)


        if self.new_cached_activations_path is None:
            dataset_token = (
                self.dataset_name.split("/")[-1]
                if self.dataset_name is not None
                else "custom"
            )
            self.new_cached_activations_path = (
                f"Activations/{dataset_token}/{self.model_name.split('/')[-1]}/{self.output_or_diff}/"
            )
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]


def _load_optional_json(path: str | None) -> dict | None:
    if path is None:
        return None
    json_path = Path(path).expanduser()
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expand_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())
