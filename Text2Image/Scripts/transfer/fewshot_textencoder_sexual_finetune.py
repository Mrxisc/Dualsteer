from __future__ import annotations
import argparse
import json
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence
import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from Dualsteer_Code/Text2Image.Scripts.collect.cache_activations_runner import CacheActivationsRunner
from Dualsteer_Code/Text2Image.Scripts.train.config import CacheActivationsRunnerConfig, SaeConfig, TrainConfig
from Dualsteer_Code/Text2Image.Scripts.train.sae import Sae
from Dualsteer_Code/Text2Image.Scripts.train.trainer import SaeTrainer


@dataclass
class FewShotTrainConfig(TrainConfig):
    num_epochs: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    seed: int = 42


class TorchActivationDataset(torch.utils.data.Dataset):
    def __init__(self, activations: torch.Tensor, timestep: torch.Tensor):
        if activations.ndim != 3:
            raise ValueError(f"Expected activations shape [N, sample_size, d_in], got {tuple(activations.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"Expected timestep shape [N], got {tuple(timestep.shape)}")
        if activations.shape[0] != timestep.shape[0]:
            raise ValueError("activations and timestep must have the same length")
        self._activations = activations
        self._timestep = timestep
        # SaeTrainer reads dataset.features["activations"].shape
        self.features = {"activations": SimpleNamespace(shape=tuple(activations.shape[1:]))}

    def __len__(self) -> int:
        return int(self._activations.shape[0])

    def __getitem__(self, idx: int):
        return {"activations": self._activations[idx], "timestep": self._timestep[idx]}

    def select(self, indices: Sequence[int]) -> "TorchActivationDataset":
        if isinstance(indices, range):
            indices = list(indices)
        if isinstance(indices, torch.Tensor):
            idx_t = indices.to(dtype=torch.long)
        else:
            idx_t = torch.as_tensor(list(indices), dtype=torch.long)
        return TorchActivationDataset(self._activations[idx_t], self._timestep[idx_t])


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - handled by argparse choices
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(mapping)}") from exc


def dataset_ready(base_dir: Path, hookpoints: List[str]) -> bool:
    if not base_dir.exists():
        return False

    for hook in hookpoints:
        hook_dir = base_dir / hook
        if not hook_dir.exists():
            return False
        if not (hook_dir / "dataset_info.json").exists():
            return False
        if not (hook_dir / "state.json").exists():
            return False
    return True


def collect_activations(args, hookpoints: List[str], cache_dtype: torch.dtype) -> Path:
    output_dir = Path(args.activations_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    need_cache = args.force_cache or not dataset_ready(output_dir, hookpoints)

    if not need_cache:
        print(f"[cache] Using cached activations in {output_dir}")
        return output_dir

    for hook in hookpoints:
        hook_dir = output_dir / hook
        if not hook_dir.exists():
            continue
        finalized = (hook_dir / "dataset_info.json").exists() and (hook_dir / "state.json").exists()
        if finalized and not args.force_cache:
            continue
        try:
            if any(hook_dir.iterdir()):
                print(f"[cache] Removing partial cache folder: {hook_dir}")
                shutil.rmtree(hook_dir)
        except Exception:
            pass

    cfg = CacheActivationsRunnerConfig(
        hook_names=hookpoints,
        new_cached_activations_path=str(output_dir),
        model_name=args.model_name,
        dtype=cache_dtype,
        num_inference_steps=args.cache_steps,
        seed=args.seed,
        batch_size=args.cache_batch_size,
        output_or_diff="output",
        cache_every_n_timesteps=args.cache_every_n,
        guidance_scale=args.cache_guidance_scale,
        csv_path=args.sexual_csv,
        csv_prompt_column=args.csv_prompt_column,
        csv_category_column=args.csv_category_column,
        csv_filter_categories=args.csv_filter,
        csv_category_match_all=args.csv_match_all,
        csv_max_rows=args.max_cache_rows,
        max_num_examples=args.max_cache_examples,
        use_dit=args.use_dit,
        dit_config_path=args.dit_config,
        use_flux=args.use_flux,
        flux_config_path=args.flux_config,
        denoiser_attr=args.denoiser_attr,
        column="caption",
    )
    print("[cache] Launching CacheActivationsRunner ...")
    CacheActivationsRunner(cfg).run()
    return output_dir


def _iter_arrow_record_batches(hook_dir: Path):
    state_path = hook_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json in {hook_dir} (cache not finalized?)")
    state = json.loads(state_path.read_text())
    data_files = [hook_dir / f["filename"] for f in state.get("_data_files", [])]
    if not data_files:
        data_files = sorted(hook_dir.glob("data-*.arrow"))
    for fp in data_files:
        if not fp.exists():
            continue
        with pa.memory_map(str(fp), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                yield batch


def _load_arrow_activations(hook_dir: Path, *, dtype: torch.dtype, max_examples: int | None, seed: int) -> TorchActivationDataset:
    if not hook_dir.exists():
        raise FileNotFoundError(f"Missing activations for hook in {hook_dir}")

    want = None if max_examples is None else int(max_examples)
    acts_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    total = 0

    for batch in _iter_arrow_record_batches(hook_dir):
        acts_py = batch.column(0).to_pylist()
        ts_py = batch.column(1).to_pylist()
        acts_np = np.asarray(acts_py, dtype=np.float16)
        ts_np = np.asarray(ts_py, dtype=np.uint16)

        if want is not None and total + acts_np.shape[0] > want:
            keep = want - total
            if keep <= 0:
                break
            acts_np = acts_np[:keep]
            ts_np = ts_np[:keep]

        acts_chunks.append(acts_np)
        ts_chunks.append(ts_np)
        total += int(acts_np.shape[0])
        if want is not None and total >= want:
            break

    if total == 0:
        raise RuntimeError(f"No activation rows found under {hook_dir}")

    activations = np.concatenate(acts_chunks, axis=0)
    timesteps = np.concatenate(ts_chunks, axis=0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(activations.shape[0])
    activations = activations[perm]
    timesteps = timesteps[perm]
    activations_t = torch.from_numpy(activations)
    if dtype != activations_t.dtype:
        activations_t = activations_t.to(dtype)
    timestep_t = torch.from_numpy(timesteps.astype(np.int64, copy=False))
    return TorchActivationDataset(activations_t, timestep_t)


def _load_single_dataset(
    base_dir: Path,
    hook: str,
    *,
    seed: int,
    max_examples: int | None,
) -> TorchActivationDataset:
    hook_dir = base_dir / hook
    return _load_arrow_activations(hook_dir, dtype=torch.float16, max_examples=max_examples, seed=seed)


def load_dataset_dict_with_replay(
    primary_dir: Path,
    replay_dirs: Sequence[Path],
    hookpoints: List[str],
    dtype: torch.dtype,
    *,
    primary_max_examples: int | None,
    replay_max_examples: int | None,
    replay_ratio: float,
    seed: int,
) -> Dict[str, TorchActivationDataset]:
    r = float(replay_ratio)
    if r < 0.0 or r >= 1.0:
        raise ValueError("--replay-ratio must be in [0, 1). Use 0 to disable replay.")

    dataset_dict: Dict[str, TorchActivationDataset] = {}
    for hook in hookpoints:
        primary = _load_arrow_activations(primary_dir / hook, dtype=dtype, max_examples=primary_max_examples, seed=seed)

        combined = primary
        if replay_dirs and r > 0.0:
            replay_sets = [
                _load_arrow_activations(d / hook, dtype=dtype, max_examples=replay_max_examples, seed=seed + 13)
                for d in replay_dirs
            ]
            replay_acts = torch.cat([ds._activations for ds in replay_sets], dim=0)
            replay_ts = torch.cat([ds._timestep for ds in replay_sets], dim=0)
            replay_count = int((r / (1.0 - r)) * len(primary))
            replay_count = min(replay_count, int(replay_acts.shape[0]))
            if replay_count > 0:
                gen = torch.Generator().manual_seed(seed + 17)
                perm = torch.randperm(replay_acts.shape[0], generator=gen)
                replay_sample_acts = replay_acts[perm[:replay_count]]
                replay_sample_ts = replay_ts[perm[:replay_count]]

                combined_acts = torch.cat([primary._activations, replay_sample_acts], dim=0)
                combined_ts = torch.cat([primary._timestep, replay_sample_ts], dim=0)
                gen2 = torch.Generator().manual_seed(seed + 23)
                perm2 = torch.randperm(combined_acts.shape[0], generator=gen2)
                combined = TorchActivationDataset(combined_acts[perm2], combined_ts[perm2])
                print(
                    f"[data] {hook}: primary={len(primary)} + replay={replay_count} (ratio~{replay_count/len(combined):.3f})"
                )
            else:
                print(f"[data] {hook}: replay requested but replay_count=0 (replay pool too small)")

        dataset_dict[hook] = combined
        print(f"[data] Loaded {len(combined)} total samples for {hook}")

    min_len = min(len(ds) for ds in dataset_dict.values()) if dataset_dict else 0
    if min_len > 0:
        for hook in hookpoints:
            ds = dataset_dict[hook]
            if len(ds) != min_len:
                dataset_dict[hook] = ds.select(range(min_len))
        print(f"[data] Aligned hookpoint dataset lengths to min_len={min_len}")
    return dataset_dict


def load_base_saes(base_dir: Path, hookpoints: List[str], device: torch.device) -> Dict[str, Sae]:
    saes: Dict[str, Sae] = {}
    for hook in hookpoints:
        ckpt_dir = base_dir / hook
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Missing SAE checkpoint for hook '{hook}' in {ckpt_dir}")
        saes[hook] = Sae.load_from_disk(ckpt_dir, device=device)
        print(f"[init] Loaded base SAE weights from {ckpt_dir}")
    return saes


def export_saes(saes: Dict[str, Sae], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for hook, sae in saes.items():
        target = output_dir / hook
        sae.save_to_disk(target)
        print(f"[export] Saved finetuned SAE for {hook} to {target}")


def build_train_config(args, sae_cfg: SaeConfig) -> FewShotTrainConfig:
    cfg = FewShotTrainConfig(
        sae=sae_cfg,
        dataset_path=[str(Path(args.activations_dir).expanduser())],
        effective_batch_size=args.effective_batch_size,
        num_workers=args.num_workers,
        grad_acc_steps=args.grad_acc_steps,
        micro_acc_steps=args.micro_acc_steps,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        auxk_alpha=args.auxk_alpha,
        dead_feature_threshold=args.dead_feature_threshold,
        hookpoints=args.hookpoints,
        distribute_modules=False,
        save_every=args.save_every,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_log_frequency=args.wandb_log_frequency,
        run_name=args.run_name,
    )
    cfg.device = args.device
    cfg.dtype = parse_dtype(args.train_dtype)
    cfg.num_epochs = args.num_epochs
    cfg.seed = args.seed
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot SAE finetuning on i2p sexual subset")
    parser.add_argument("--base-sae-dir", required=True, help="Directory that contains the pretrained SAE checkpoints (one subfolder per hookpoint)")
    parser.add_argument("--hookpoints", nargs="+", required=True, help="Hookpoint names to finetune (must match both cached activations and SAE checkpoints)")
    parser.add_argument("--model-name", required=True, help="Diffusion backbone path or huggingface id")
    parser.add_argument("--sexual-csv", required=True, help="CSV file with i2p prompts")
    parser.add_argument("--activations-dir", default="Activations/i2p_sexual_flux", help="Where to cache the few-shot activations")
    parser.add_argument("--output-sae-dir", required=True, help="Destination directory for finetuned SAEs")
    parser.add_argument("--flux-config", default=None, help="Optional JSON cfg for Flux backbone")
    parser.add_argument("--dit-config", default=None, help="Optional JSON cfg for DiT backbone")
    parser.add_argument("--use-flux", action="store_true", help="Use Flux backbone hooks instead of UNet")
    parser.add_argument("--use-dit", action="store_true", help="Use DiT backbone hooks instead of UNet")
    parser.add_argument("--denoiser-attr", default=None, help="Custom denoiser attribute exposed by HookedStableDiffusionPipeline")
    parser.add_argument("--csv-prompt-column", default="prompt")
    parser.add_argument("--csv-category-column", default="categories")
    parser.add_argument("--csv-filter", nargs="+", default=["sexual"], help="Category keywords to keep from the CSV")
    parser.add_argument("--csv-match-all", action="store_true", help="Require all category keywords to be present")
    parser.add_argument("--max-cache-rows", type=int, default=None, help="Cut CSV after N rows before filtering")
    parser.add_argument("--max-cache-examples", type=int, default=None, help="Stop caching after N prompt rows")
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-steps", type=int, default=30)
    parser.add_argument("--cache-guidance-scale", type=float, default=4.0)
    parser.add_argument("--cache-every-n", type=int, default=1)
    parser.add_argument("--cache-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--force-cache", action="store_true", help="Recompute activations even if they already exist")
    parser.add_argument(
        "--train-decoder-only",
        action="store_true",
        help="Freeze SAE encoder and only finetune decoder (W_dec and b_dec).",
    )

    parser.add_argument(
        "--replay-activations-dir",
        nargs="+",
        default=None,
        help=(
            "Optional cached activations root(s) from a source dataset for replay. "
            "Each directory must contain per-hookpoint Dataset folders."
        ),
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.0,
        help="Fraction of replay samples in the mixed dataset (0 disables replay).",
    )
    parser.add_argument(
        "--max-replay-examples",
        type=int,
        default=None,
        help="Optional clamp on replay rows per hook (applied before sampling replay-ratio).",
    )
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--auxk-alpha", type=float, default=0.0)
    parser.add_argument("--dead-feature-threshold", type=int, default=10_000_000)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--micro-acc-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--wandb-project", default="dualsteer_finetune")
    parser.add_argument("--wandb-log-frequency", type=int, default=4000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--train-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None, help="Optional clamp on training rows per hook")
    parser.add_argument("--train-hard-limit", type=int, default=None, help="Deprecated: use --max-train-examples")
    return parser.parse_args()


def main():
    args = parse_args()
    hookpoints = args.hookpoints
    device = torch.device(args.device)
    cache_dtype = parse_dtype(args.cache_dtype)
    activations_dir = collect_activations(args, hookpoints, cache_dtype)
    replay_dirs: List[Path] = []
    if args.replay_activations_dir:
        replay_dirs = [Path(p).expanduser() for p in args.replay_activations_dir]
        for p in replay_dirs:
            if not p.exists():
                raise FileNotFoundError(f"Replay activations dir does not exist: {p}")

    dataset_dict = load_dataset_dict_with_replay(
        Path(activations_dir).expanduser(),
        replay_dirs,
        hookpoints,
        parse_dtype(args.train_dtype),
        primary_max_examples=args.max_train_examples or args.train_hard_limit,
        replay_max_examples=args.max_replay_examples,
        replay_ratio=args.replay_ratio,
        seed=args.seed,
    )

    base_saes = load_base_saes(Path(args.base_sae_dir).expanduser(), hookpoints, device)
    sae_cfg = next(iter(base_saes.values())).cfg
    train_cfg = build_train_config(args, sae_cfg)
    train_cfg.train_decoder_only = bool(args.train_decoder_only)
    trainer = SaeTrainer(train_cfg, dataset_dict)
    for hook, sae in trainer.saes.items():
        state_dict = base_saes[hook].state_dict()
        sae.load_state_dict(state_dict)
        print(f"[init] Warm-started trainer SAE for {hook}")

    trainer.fit()
    export_saes(trainer.saes, Path(args.output_sae_dir).expanduser())


if __name__ == "__main__":
    main()
