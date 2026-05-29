from __future__ import annotations
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Array2D, Dataset, Features, Value
from simple_parsing import Serializable, list_field, parse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Dualsteer_Code/Text2Image.Scripts.collect.cache_activations_runner import (  # noqa: E402
    CacheActivationsRunner as _LegacyRunner,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32", torch.bfloat16: "bfloat16"}


@dataclass
class CacheRealDiffusersActivationsConfig(Serializable):
    backend: str = "flux" 
    model_name: str = "Models/FLUX.1-dev"
    hook_names: List[str] = list_field()
    new_cached_activations_path: str = "Activations/i2p_no_sexual_flux_real"
    csv_path: Optional[str] = None
    csv_prompt_column: str = "prompt"
    csv_category_column: Optional[str] = "categories"
    csv_filter_categories: List[str] = list_field()
    csv_category_match_all: bool = False
    csv_hard_column: Optional[str] = "hard"
    csv_require_hard: bool = False
    csv_max_rows: Optional[int] = None
    csv_strip_period: bool = True
    csv_deduplicate: bool = True
    seed: int = 42
    batch_size: int = 1
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    height: int = 256
    width: int = 256
    cache_every_n_timesteps: int = 6
    resume: bool = False
    token_subsample: Optional[int] = 256
    token_subsample_seed: int = 0
    token_indices_path: Optional[str] = None
    channel_proj_dim: Optional[int] = 1024
    channel_proj_seed: int = 0
    channel_proj_path: Optional[str] = None
    activation_part: str = "tuple1"
    cfg_keep: str = "cond"
    dtype: str = "float16"


def _torch_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_module(root: Any, dotted_path: str) -> torch.nn.Module:
    obj: Any = root
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError as e:
                if part == "double_transformer_blocks" and hasattr(obj, "transformer_blocks"):
                    obj = getattr(obj, "transformer_blocks")
                elif part == "transformer_blocks" and hasattr(obj, "double_transformer_blocks"):
                    obj = getattr(obj, "double_transformer_blocks")
                else:
                    raise e
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"Resolved hookpoint is not a module: {dotted_path} -> {type(obj)}")
    return obj


def _select_activation(output: Any, activation_part: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        if activation_part not in ("tensor", "tuple0", "tuple1", "concat01"):
            return output
        return output

    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if isinstance(x, torch.Tensor)]
        if not tensors:
            raise TypeError(f"Hook output had no tensors: {type(output)}")

        if activation_part == "tuple0":
            return tensors[0]
        if activation_part == "tuple1":
            return tensors[1] if len(tensors) > 1 else tensors[0]
        if activation_part == "concat01":
            if len(tensors) == 1:
                return tensors[0]
            a, b = tensors[0], tensors[1]
            if a.ndim != 3 or b.ndim != 3:
                raise ValueError(f"concat01 expects 3D tensors; got {a.shape} and {b.shape}")
            if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
                raise ValueError(f"concat01 expects same batch and channel dims; got {a.shape} and {b.shape}")
            return torch.cat([a, b], dim=1)

        return tensors[0]

    raise TypeError(f"Unsupported hook output type: {type(output)}")


def _maybe_keep_cond_branch(x: torch.Tensor, guidance_scale: float, cfg_keep: str) -> torch.Tensor:
    if cfg_keep != "cond":
        return x
    if guidance_scale is None or float(guidance_scale) <= 1.0:
        return x
    if x.ndim < 1:
        return x
    if x.shape[0] % 2 != 0:
        return x
    half = x.shape[0] // 2
    return x[half:]


def _load_prompts_from_csv(cfg: CacheRealDiffusersActivationsConfig, accelerator: Accelerator) -> List[str]:
    if pd is None:
        raise ImportError("pandas is required for --csv_path mode")
    if cfg.csv_path is None:
        raise ValueError("--csv_path is required")

    df = pd.read_csv(cfg.csv_path)

    if cfg.csv_category_column and cfg.csv_filter_categories:
        col = cfg.csv_category_column
        if cfg.csv_category_match_all:
            mask = pd.Series(True, index=df.index)
            for cat in cfg.csv_filter_categories:
                mask &= df[col].fillna("").astype(str).str.contains(cat, case=False, na=False)
        else:
            mask = pd.Series(False, index=df.index)
            for cat in cfg.csv_filter_categories:
                mask |= df[col].fillna("").astype(str).str.contains(cat, case=False, na=False)
        df = df[mask]

    if cfg.csv_require_hard and cfg.csv_hard_column and cfg.csv_hard_column in df.columns:
        df = df[df[cfg.csv_hard_column].fillna(0).astype(int) > 0]

    if cfg.csv_max_rows is not None:
        df = df.head(int(cfg.csv_max_rows))

    if cfg.csv_prompt_column not in df.columns:
        raise ValueError(f"Prompt column '{cfg.csv_prompt_column}' not found in {cfg.csv_path}")

    prompts = df[cfg.csv_prompt_column].fillna("").astype(str).tolist()
    processed: List[str] = []
    seen = set()
    for p in prompts:
        p = p.strip()
        if cfg.csv_strip_period and p.endswith("."):
            p = p[:-1]
        if cfg.csv_deduplicate:
            if not p or p in seen:
                continue
            seen.add(p)
        if p:
            processed.append(p)

    if not processed:
        raise ValueError(f"No prompts loaded from {cfg.csv_path} after filtering")

    if accelerator.is_main_process:
        print(f"Loaded {len(processed)} prompts from CSV: {cfg.csv_path}")

    return processed


def _get_batches(items: Dataset, batch_size: int) -> List[Dataset]:
    batches: List[Dataset] = []
    num_batches = (len(items) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(items))
        batches.append(items.select(range(start, end)))
    return batches


class _Transforms:
    def __init__(self, cfg: CacheRealDiffusersActivationsConfig, accelerator: Accelerator, root_dir: Path):
        self.cfg = cfg
        self.accelerator = accelerator
        self.root_dir = root_dir

        self.token_indices: Optional[torch.Tensor] = None
        self.channel_proj: Optional[torch.Tensor] = None

        if self.cfg.token_indices_path is None:
            self.cfg.token_indices_path = str(self.root_dir / "token_indices.pt")
        if self.cfg.channel_proj_path is None:
            self.cfg.channel_proj_path = str(self.root_dir / "channel_proj.pt")

    def _prepare_token_indices(self, num_tokens: int) -> Optional[torch.Tensor]:
        cfg = self.cfg
        if cfg.token_subsample is None or int(cfg.token_subsample) <= 0 or int(cfg.token_subsample) >= int(num_tokens):
            return None

        if self.token_indices is not None:
            return self.token_indices

        path = Path(cfg.token_indices_path).expanduser()
        if path.exists():
            self.token_indices = torch.load(path).to(torch.long)
            return self.token_indices

        gen = torch.Generator().manual_seed(int(cfg.token_subsample_seed))
        indices = torch.randperm(int(num_tokens), generator=gen)[: int(cfg.token_subsample)]
        indices, _ = torch.sort(indices)
        self.token_indices = indices.to(torch.long)

        if self.accelerator.is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.token_indices.cpu(), path)

        return self.token_indices

    def _prepare_channel_proj(self, in_dim: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        cfg = self.cfg
        if cfg.channel_proj_dim is None or int(cfg.channel_proj_dim) <= 0 or int(cfg.channel_proj_dim) >= int(in_dim):
            return None

        if self.channel_proj is not None:
            return self.channel_proj.to(dtype=dtype)

        path = Path(cfg.channel_proj_path).expanduser()
        if path.exists():
            self.channel_proj = torch.load(path).to(torch.float32)
            return self.channel_proj.to(dtype=dtype)

        gen = torch.Generator().manual_seed(int(cfg.channel_proj_seed))
        mat = torch.randn((int(in_dim), int(cfg.channel_proj_dim)), generator=gen, dtype=torch.float32)
        mat = torch.nn.functional.normalize(mat, dim=0)
        self.channel_proj = mat

        if self.accelerator.is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.channel_proj.to(dtype=torch.float16).cpu(), path)

        return self.channel_proj.to(dtype=dtype)

    def apply(self, acts: torch.Tensor) -> torch.Tensor:
        if acts.ndim != 3:
            return acts

        idx = self._prepare_token_indices(acts.shape[1])
        if idx is not None:
            idx = idx.to(device=acts.device)
            acts = torch.index_select(acts, 1, idx)

        proj = self._prepare_channel_proj(acts.shape[2], acts.dtype)
        if proj is not None:
            proj = proj.to(device=acts.device, dtype=acts.dtype)
            with torch.autocast(device_type=acts.device.type, dtype=acts.dtype):
                acts = acts @ proj

        return acts

    def write_metadata(self, hook_dir: Path) -> None:
        metadata = {
            "cache_every_n_timesteps": int(self.cfg.cache_every_n_timesteps),
            "token_subsample": self.cfg.token_subsample,
            "token_indices_path": self.cfg.token_indices_path,
            "channel_proj_dim": self.cfg.channel_proj_dim,
            "channel_proj_path": self.cfg.channel_proj_path,
            "activation_part": self.cfg.activation_part,
            "cfg_keep": self.cfg.cfg_keep,
        }
        with open(hook_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def _features(dtype: torch.dtype, num_tokens: int, d_out: int) -> Features:
    return Features(
        {
            "activations": Array2D(shape=(int(num_tokens), int(d_out)), dtype=TORCH_STRING_DTYPE_MAP[dtype]),
            "timestep": Value(dtype="uint16"),
        }
    )


def _create_shard(
    buffer: torch.Tensor,
    timesteps: torch.Tensor,
    dtype: torch.dtype,
    cache_every_n: int,
    features: Features,
) -> Dataset:
    if cache_every_n > 1:
        buffer = buffer[:, ::cache_every_n, :, :]
        timesteps = timesteps[::cache_every_n]

    bsz, n_steps, tokens, dim = buffer.shape
    activations = buffer.reshape(-1, tokens, dim)

    ts = timesteps[:n_steps].clone()
    ts = ts.repeat(bsz).cpu().numpy().astype("uint16")

    return Dataset.from_dict({"activations": activations, "timestep": ts}, features=features)


def _supports_callback_on_step_end(pipe) -> bool:
    import inspect

    try:
        sig = inspect.signature(pipe.__call__)
    except Exception:
        return False
    return "callback_on_step_end" in sig.parameters


def main() -> None:
    cfg = parse(CacheRealDiffusersActivationsConfig)
    dtype = _torch_dtype(cfg.dtype)

    accelerator = Accelerator()
    out_root = Path(cfg.new_cached_activations_path).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Output root: {out_root}")

    prompts = _load_prompts_from_csv(cfg, accelerator)
    ds = Dataset.from_dict({"caption": prompts})
    ds = ds.shuffle(cfg.seed)
    batches = _get_batches(ds, int(cfg.batch_size))

    if cfg.backend.lower() == "flux":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(cfg.model_name, torch_dtype=dtype)
    elif cfg.backend.lower() in ("sd35", "sd3", "sd3.5"):
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(cfg.model_name, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    pipe.set_progress_bar_config(disable=True)
    pipe.to(accelerator.device)
    if hasattr(pipe, "vae"):
        try:
            pipe.vae.to("cpu")
        except Exception:
            pass


    hook_modules: Dict[str, torch.nn.Module] = {}
    for hook_name in cfg.hook_names:
        hook_modules[hook_name] = _resolve_module(pipe, hook_name)

    transforms = _Transforms(cfg, accelerator, out_root)
    final_paths: Dict[str, Path] = {h: out_root / h for h in cfg.hook_names}

    hooks_to_process: List[str] = []
    resume_start_index = 0

    if accelerator.is_main_process:
        for hook_name, path in final_paths.items():
            path.mkdir(exist_ok=True, parents=True)
            tmp_dir = path / ".tmp_shards"
            finalized = _LegacyRunner._dataset_is_finalized(path)

            if not cfg.resume:
                if any(path.iterdir()):
                    raise RuntimeError(
                        f"Activations directory ({path}) is not empty. Delete it or run with --resume."
                    )
                tmp_dir.mkdir(exist_ok=False, parents=False)
                hooks_to_process.append(hook_name)
                continue

            if finalized and not tmp_dir.exists():
                print(f"Hook {hook_name} already cached; skipping due to resume flag.")
                continue

            if not tmp_dir.exists():
                tmp_dir.mkdir(exist_ok=True, parents=False)

            existing = _LegacyRunner._existing_shard_indices(tmp_dir)
            resume_start_index = (existing[-1] + 1) if existing else 0
            hooks_to_process.append(hook_name)

    shared = broadcast_object_list(
        [hooks_to_process, resume_start_index] if accelerator.is_main_process else [None, None]
    )
    hooks_to_process, resume_start_index = shared

    if not hooks_to_process:
        if accelerator.is_main_process:
            print("Nothing to do; all requested hooks already cached.")
        return

    cfg.hook_names = hooks_to_process
    tmp_paths: Dict[str, Path] = {h: final_paths[h] / ".tmp_shards" for h in hooks_to_process}
    if accelerator.is_main_process:
        probe_state: Dict[str, Tuple[int, int]] = {}

        handles = []

        def _probe_hook(hook_name: str):
            def fn(_m, _inp, out):
                t = _select_activation(out, cfg.activation_part)
                t = _maybe_keep_cond_branch(t, cfg.guidance_scale, cfg.cfg_keep)
                if t.ndim != 3:
                    raise ValueError(f"Expected 3D activation (batch,tokens,channels); got {t.shape}")
                probe_state[hook_name] = (int(t.shape[1]), int(t.shape[2]))

            return fn

        for h in hooks_to_process:
            handles.append(hook_modules[h].register_forward_hook(_probe_hook(h)))

        gen = torch.Generator(device=accelerator.device).manual_seed(int(cfg.seed))
        _ = pipe(
            prompt=["shape probe"],
            num_inference_steps=1,
            guidance_scale=float(cfg.guidance_scale),
            height=int(cfg.height),
            width=int(cfg.width),
            output_type="latent",
            generator=gen,
        )

        for h in handles:
            h.remove()

        if not probe_state:
            raise RuntimeError("Probe forward did not trigger any hooks; check hook_names.")

        ref_tokens, ref_dim = next(iter(probe_state.values()))
        for hn, (nt, dd) in probe_state.items():
            if nt != ref_tokens or dd != ref_dim:
                raise ValueError(
                    f"Hook shapes differ within one run; please run hooks separately. {probe_state}"
                )

        _ = transforms._prepare_token_indices(ref_tokens)
        _ = transforms._prepare_channel_proj(ref_dim, dtype)

    accelerator.wait_for_everyone()

    if transforms.cfg.token_indices_path and Path(transforms.cfg.token_indices_path).exists():
        transforms.token_indices = torch.load(transforms.cfg.token_indices_path).to(torch.long)
    if transforms.cfg.channel_proj_path and Path(transforms.cfg.channel_proj_path).exists():
        transforms.channel_proj = torch.load(transforms.cfg.channel_proj_path).to(torch.float32)

    accelerator.wait_for_everyone()

    for i, batch in enumerate(batches):
        if i < int(resume_start_index):
            continue

        with accelerator.split_between_processes(batch) as prompt_batch:
            prompts_local = prompt_batch["caption"]
            if isinstance(prompts_local, str):
                prompts_local = [prompts_local]

            latest: Dict[str, Optional[torch.Tensor]] = {h: None for h in hooks_to_process}
            collected: Dict[str, List[torch.Tensor]] = {h: [] for h in hooks_to_process}
            timesteps_seen: List[int] = []

            def _make_hook(hook_name: str):
                def fn(_m, _inp, out):
                    t = _select_activation(out, cfg.activation_part)
                    t = _maybe_keep_cond_branch(t, cfg.guidance_scale, cfg.cfg_keep)
                    t = transforms.apply(t)
                    latest[hook_name] = t.detach().to("cpu", dtype=torch.float16)

                return fn

            handles = [hook_modules[h].register_forward_hook(_make_hook(h)) for h in hooks_to_process]

            def _on_step_end(pipe_obj, step: int, timestep: int, callback_kwargs: Dict[str, Any]):
                try:
                    ts_val = int(timestep) if not isinstance(timestep, torch.Tensor) else int(timestep.item())
                except Exception:
                    ts_val = int(step)
                timesteps_seen.append(ts_val)
                for h in hooks_to_process:
                    if latest[h] is None:
                        raise RuntimeError(f"Hook {h} did not run at step {step}")
                    collected[h].append(latest[h])
                return callback_kwargs

            gen = torch.Generator(device=accelerator.device).manual_seed(int(cfg.seed) + int(i))

            if _supports_callback_on_step_end(pipe):
                _ = pipe(
                    prompt=list(prompts_local),
                    num_inference_steps=int(cfg.num_inference_steps),
                    guidance_scale=float(cfg.guidance_scale),
                    height=int(cfg.height),
                    width=int(cfg.width),
                    output_type="latent",
                    generator=gen,
                    callback_on_step_end=_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            else:
                def _callback(step: int, timestep: int, _latents: torch.Tensor):
                    try:
                        ts_val = int(timestep) if not isinstance(timestep, torch.Tensor) else int(timestep.item())
                    except Exception:
                        ts_val = int(step)
                    timesteps_seen.append(ts_val)
                    for h in hooks_to_process:
                        if latest[h] is None:
                            raise RuntimeError(f"Hook {h} did not run at step {step}")
                        collected[h].append(latest[h])

                _ = pipe(
                    prompt=list(prompts_local),
                    num_inference_steps=int(cfg.num_inference_steps),
                    guidance_scale=float(cfg.guidance_scale),
                    height=int(cfg.height),
                    width=int(cfg.width),
                    output_type="latent",
                    generator=gen,
                    callback=_callback,
                    callback_steps=1,
                )

            for h in handles:
                h.remove()

        accelerator.wait_for_everyone()

        gathered = gather_object([
            {
                "acts": {h: collected[h] for h in hooks_to_process},
                "timesteps": timesteps_seen,
            }
        ])

        if accelerator.is_main_process:
            if not gathered or "timesteps" not in gathered[0]:
                raise RuntimeError("Failed to gather per-step timesteps from worker processes")

            gathered_timesteps = gathered[0]["timesteps"]
            if not isinstance(gathered_timesteps, list) or len(gathered_timesteps) == 0:
                raise RuntimeError("No timesteps recorded; callback may not have been invoked")
            timesteps_tensor = torch.tensor(gathered_timesteps, dtype=torch.long)

            for h in hooks_to_process:
                per_rank_steps = [g["acts"][h] for g in gathered]

                steps_cat: List[torch.Tensor] = []
                for step_idx in range(len(per_rank_steps[0])):
                    step_tensors = [rank_steps[step_idx] for rank_steps in per_rank_steps]
                    steps_cat.append(torch.cat(step_tensors, dim=0))

                buffer = torch.stack(steps_cat, dim=1) 
                feats = _features(dtype=torch.float16, num_tokens=int(buffer.shape[2]), d_out=int(buffer.shape[3]))

                shard = _create_shard(
                    buffer=buffer,
                    timesteps=timesteps_tensor,
                    dtype=torch.float16,
                    cache_every_n=int(cfg.cache_every_n_timesteps),
                    features=feats,
                )

                shard.save_to_disk(str(tmp_paths[h] / f"shard_{i:05d}"), num_shards=1)
                del buffer, shard

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        for h in hooks_to_process:
            _LegacyRunner._consolidate_shards(tmp_paths[h], final_paths[h], copy_files=False)
            transforms.write_metadata(final_paths[h])
            print(f"Consolidated dataset for hook {h}")


if __name__ == "__main__":
    main()
