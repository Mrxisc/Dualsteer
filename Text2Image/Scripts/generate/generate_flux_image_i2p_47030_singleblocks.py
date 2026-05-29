import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
from diffusers import FluxPipeline
from tqdm import tqdm
from Dualsteer_Code/Text2Image.Scripts.train.sae import Sae

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = ROOT_DIR.parent

MANIFEST_COLUMNS = [
    "sample_index",
    "case_number",
    "prompt",
    "categories",
    "sd_seed",
    "sd_guidance_scale",
    "idx_generation",
    "image_path",
    "sae_tag",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate I2P images with Flux + singleblock SAE protection")
    parser.add_argument("--flux-ckpt", required=True, help="Diffusers checkpoint for Flux model")
    parser.add_argument(
        "--prompts-csv",
        default=str(ROOT_DIR / "Datasets" / "i2p_benchmark.csv"),
    )
    parser.add_argument(
        "--save-root",
        default=str(ROOT_DIR / "Results" / "i2p_flux_singleblock"),
    )
    parser.add_argument(
        "--sae-single-checkpoint",
        default=str(ROOT_DIR / "SAEs" / "flux_real_singleblock37_i2p_no_sexual_i2p_no_sexual_flux_real_i2p_no_sexual_flux_real"),
        help="Checkpoint root for the singleblock SAE",
    )
    parser.add_argument(
        "--single-hookpoint",
        default="transformer.single_transformer_blocks.37",
        help="Hookpoint path used during SAE training/collection (will be mapped to runtime path if needed)",
    )
    parser.add_argument(
        "--activations-root",
        default=str(ROOT_DIR / "Activations" / "i2p_no_sexual_flux_real" / "transformer.single_transformer_blocks.37"),
        help="Activation cache root (used to load channel_proj.pt / token_indices.pt; if absent, parent dir is tried)",
    )
    parser.add_argument(
        "--sae-tag",
        default="singleblock37",
        help="Tag name for output subdir and manifest (e.g., singleblock37_s0.2)",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-images-per-prompt", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--sae-strength",
        type=float,
        default=0.2,
        help="Blend strength for SAE intervention (0=no-op, 1=full replacement). Default=0.2 to reduce garbling.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="For debugging: limit to first N prompts (0 means no limit).",
    )
    return parser


def _parse_dtype(dtype_str: str) -> torch.dtype:
    lut = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return lut[dtype_str]


def _resolve_checkpoint_dir(checkpoint_root: str, hookpoint: str) -> Optional[Path]:
    root = Path(checkpoint_root)
    if (root / "sae.safetensors").exists():
        return root

    def _expand_subdir_candidates(p: str) -> list[str]:
        candidates: list[str] = []
        if p:
            candidates.append(p)
        parts = p.split(".") if p else []
        if parts and parts[0] == "transformer":
            rest = ".".join(parts[1:])
            if rest:
                candidates.append(rest)

        if p.startswith("transformer."):
            candidates.append(p[len("transformer."):])

        seen = set()
        out: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    for sub in _expand_subdir_candidates(hookpoint):
        cand = root / sub
        if (cand / "sae.safetensors").exists():
            return cand
    return None


def _maybe_load_token_indices(activations_root: str, device: torch.device) -> Optional[torch.Tensor]:
    token_path = Path(activations_root) / "token_indices.pt"
    if not token_path.exists():
        parent = Path(activations_root).parent
        token_path = parent / "token_indices.pt"
        if not token_path.exists():
            return None
    try:
        indices = torch.load(token_path, map_location="cpu").to(torch.long)
    except Exception:
        indices = torch.load(token_path).to(torch.long)
    return indices.to(device=device)


def _maybe_load_channel_proj(activations_root: str, device: torch.device) -> Optional[torch.Tensor]:
    proj_path = Path(activations_root) / "channel_proj.pt"
    if not proj_path.exists():
        parent = Path(activations_root).parent
        proj_path = parent / "channel_proj.pt"
        if not proj_path.exists():
            return None
    proj = torch.load(proj_path, map_location="cpu")
    if not isinstance(proj, torch.Tensor):
        proj = torch.tensor(proj)
    return proj.to(device=device)


def _apply_sae_to_hidden(
    hidden: torch.Tensor,
    sae: Sae,
    token_indices: Optional[torch.Tensor] = None,
    channel_proj: Optional[torch.Tensor] = None,
    strength: float = 1.0,
) -> torch.Tensor:
    if hidden.ndim < 2:
        return hidden

    try:
        strength_f = float(strength)
    except Exception:
        strength_f = 1.0
    if strength_f <= 0.0:
        return hidden
    if strength_f > 1.0:
        strength_f = 1.0

    orig_shape = hidden.shape
    tokens = hidden.shape[-2]
    d_in = hidden.shape[-1]
    flat = hidden.reshape(-1, tokens, d_in)

    sae_d_in = None
    if hasattr(sae, "b_dec") and isinstance(getattr(sae, "b_dec"), torch.Tensor):
        sae_d_in = int(sae.b_dec.numel())

    proj = None
    if sae_d_in is not None and int(d_in) != int(sae_d_in):
        if isinstance(channel_proj, torch.Tensor) and channel_proj.ndim == 2:
            if int(channel_proj.shape[0]) == int(d_in) and int(channel_proj.shape[1]) == int(sae_d_in):
                proj = channel_proj.to(device=flat.device, dtype=flat.dtype)
            else:
                raise RuntimeError(
                    f"channel_proj shape mismatch: got {tuple(channel_proj.shape)} but expected ({d_in}, {sae_d_in})"
                )
        else:
            raise RuntimeError(
                f"SAE expects d_in={sae_d_in} but hooked tensor has d_in={d_in}; missing channel_proj.pt"
            )

    def _run_sae(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        bs, sample_size, _ = x.shape
        sae_in, x_mean, x_std = sae.preprocess_input(x)
        pre_acts = sae.pre_acts(sae_in)
        top_acts, top_indices = sae.select_topk(pre_acts, batch_size=bs, k=sae.cfg.k)
        recon = sae.decode(top_acts, top_indices)
        recon = sae.postprocess_output(recon, x_mean, x_std)
        return recon.view(bs, sample_size, out_dim)

    if token_indices is not None and token_indices.numel() > 0:
        idx = token_indices
        if idx.device != flat.device:
            idx = idx.to(device=flat.device)
        if int(idx.max()) < tokens and idx.numel() < tokens:
            picked = torch.index_select(flat, dim=1, index=idx)
            picked_in = picked
            if proj is not None:
                picked_in = picked_in @ proj
            recon = _run_sae(picked_in, sae_d_in if sae_d_in is not None else int(d_in))
            if proj is not None:
                recon = recon @ proj.transpose(0, 1)
            out = flat.clone()
            out[:, idx, :] = out[:, idx, :] * (1.0 - strength_f) + recon * strength_f
            return out.reshape(orig_shape)

    x_in = flat
    if proj is not None:
        x_in = x_in @ proj
    recon = _run_sae(x_in, sae_d_in if sae_d_in is not None else int(d_in))
    if proj is not None:
        recon = recon @ proj.transpose(0, 1)
    blended = flat * (1.0 - strength_f) + recon * strength_f
    return blended.reshape(orig_shape)


def _build_tensor_sae_hook(
    sae: Sae,
    *,
    token_indices: Optional[torch.Tensor] = None,
    channel_proj: Optional[torch.Tensor] = None,
    strength: float = 1.0,
):
    @torch.no_grad()
    def hook(module, inp, out):
        sae_d_in = None
        if hasattr(sae, "b_dec") and isinstance(getattr(sae, "b_dec"), torch.Tensor):
            sae_d_in = int(sae.b_dec.numel())

        if isinstance(out, tuple):
            out_list = list(out)
            target_idx = None
            for idx, elem in enumerate(out_list):
                if isinstance(elem, torch.Tensor) and elem.ndim >= 2:
                    if sae_d_in is not None and int(elem.shape[-1]) == int(sae_d_in):
                        target_idx = idx
                        break
            if target_idx is None:
                if len(out_list) >= 2 and isinstance(out_list[1], torch.Tensor):
                    target_idx = 1
                elif len(out_list) >= 1 and isinstance(out_list[0], torch.Tensor):
                    target_idx = 0

            if target_idx is not None and isinstance(out_list[target_idx], torch.Tensor):
                out_list[target_idx] = _apply_sae_to_hidden(
                    out_list[target_idx],
                    sae,
                    token_indices=token_indices,
                    channel_proj=channel_proj,
                    strength=strength,
                )
                return tuple(out_list)
            return out

        if not isinstance(out, torch.Tensor):
            return out

        return _apply_sae_to_hidden(
            out,
            sae,
            token_indices=token_indices,
            channel_proj=channel_proj,
            strength=strength,
        )

    return hook


def _resolve_module_for_hook(pipe, hookpoint: str):
    def _expand_candidates(p: str) -> list[str]:
        cands: list[str] = []
        if p:
            cands.append(p)
        if "double_transformer_blocks" in p:
            cands.append(p.replace("double_transformer_blocks", "transformer_blocks"))
        if "single_transformer_blocks" in p:
            cands.append(p.replace("single_transformer_blocks", "transformer_blocks"))
        seen = set()
        out: list[str] = []
        for x in cands:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    last_err: Exception | None = None
    for candidate in _expand_candidates(hookpoint):
        try:
            target = pipe
            for part in candidate.split("."):
                if part.isdigit():
                    target = target[int(part)]
                else:
                    target = getattr(target, part)
            return target
        except Exception as e:
            last_err = e
            continue

    raise AttributeError(
        f"Cannot resolve hookpoint '{hookpoint}'. Tried candidates: {_expand_candidates(hookpoint)}. Last error: {last_err}"
    )


def _load_sae(sae_dir: Path) -> Optional[Sae]:
    sae_file = sae_dir / "sae.safetensors"
    if not sae_file.exists():
        return None
    return Sae.load_from_disk(sae_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()


def _load_manifest(manifest_path: Path) -> Dict[str, set]:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return defaultdict(set)
    df = pd.read_csv(manifest_path)
    existing = defaultdict(set)
    for _, row in df.iterrows():
        try:
            case_key = str(row["case_number"])
            idx = int(row["idx_generation"])
            existing[case_key].add(idx)
        except Exception:
            continue
    return existing


def _iter_rows(df: pd.DataFrame) -> List[dict]:
    rows = []
    for idx, row in df.iterrows():
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append({
            "prompt": prompt.strip(),
            "case_number": idx,
            "categories": row.get("categories", ""),
        })
    return rows


def generate_for_sae(
    pipe: FluxPipeline,
    sae_tag: str,
    hook_fn,
    hookpoint: str,
    rows: List[dict],
    out_root: Path,
    steps: int,
    guidance: float,
    n_per_prompt: int,
    batch_size: int,
    height: int,
    width: int,
    neg_prompt: Optional[str],
    seed_base: int,
):
    out_root.mkdir(parents=True, exist_ok=True)
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "flux_manifest.csv"
    existing = _load_manifest(manifest_path)
    manifest_has_header = manifest_path.exists() and manifest_path.stat().st_size > 0

    torch.set_grad_enabled(False)
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device

    # Register hook once for the whole run (faster and avoids handle churn).
    target_module = _resolve_module_for_hook(pipe, hookpoint)
    handle = target_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))

    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_COLUMNS)
        if not manifest_has_header:
            writer.writeheader()

        for row in tqdm(rows, desc=f"SAE={sae_tag}"):
            prompt = row["prompt"]
            case_number = row["case_number"]
            categories = row.get("categories", "")
            case_key = str(case_number)
            case_dir = images_root / f"case_{case_number}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # base seed per prompt
            base_seed = seed_base + case_number

            # identify already present
            existing_case = existing.get(case_key, set())

            # generate list of missing indices
            missing = [i for i in range(n_per_prompt) if i not in existing_case]
            if not missing:
                continue

            try:
                for start in range(0, len(missing), batch_size):
                    batch_idx = missing[start:start + batch_size]
                    generators = []
                    for idx in batch_idx:
                        g = torch.Generator(device=device)
                        g.manual_seed(base_seed + idx)
                        generators.append(g)

                    outputs = pipe(
                        prompt=[prompt] * len(batch_idx),
                        negative_prompt=[neg_prompt] * len(batch_idx) if neg_prompt else None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=height,
                        width=width,
                        generator=generators,
                        output_type="pil",
                    )

                    for img, idx in zip(outputs.images, batch_idx):
                        fname = f"{case_number}_img{idx}.png"
                        img_path = case_dir / fname
                        img.save(img_path)
                        rel_path = Path("images") / f"case_{case_number}" / fname
                        writer.writerow({
                            "sample_index": case_number,
                            "case_number": case_number,
                            "prompt": prompt,
                            "categories": categories,
                            "sd_seed": base_seed + idx,
                            "sd_guidance_scale": guidance,
                            "idx_generation": idx,
                            "image_path": str(rel_path),
                            "sae_tag": sae_tag,
                        })
                    mf.flush()
            finally:
                pass

    handle.remove()


def main():
    args = _build_parser().parse_args()
    dtype = _parse_dtype(args.dtype)

    print(f"Loading prompts from {args.prompts_csv}")
    df = pd.read_csv(args.prompts_csv)
    rows = _iter_rows(df)
    if args.max_prompts and args.max_prompts > 0:
        rows = rows[: int(args.max_prompts)]
    print(f"Total prompts: {len(rows)} (expect ~4703; each generates {args.num_images_per_prompt} images)")

    print(f"Loading Flux pipeline from {args.flux_ckpt}")
    pipe = FluxPipeline.from_pretrained(args.flux_ckpt, torch_dtype=dtype).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device

    sae_path = _resolve_checkpoint_dir(args.sae_single_checkpoint, args.single_hookpoint)
    if sae_path is None:
        raise FileNotFoundError(f"Cannot find sae.safetensors under {args.sae_single_checkpoint} (hookpoint={args.single_hookpoint})")

    single_sae = Sae.load_from_disk(sae_path, device=device).eval()
    single_sae.cfg.batch_topk = False
    single_sae.cfg.sample_topk = False

    token_indices = _maybe_load_token_indices(args.activations_root, device=device)
    channel_proj = _maybe_load_channel_proj(args.activations_root, device=device)
    hook_fn = _build_tensor_sae_hook(
        single_sae,
        token_indices=token_indices,
        channel_proj=channel_proj,
        strength=args.sae_strength,
    )

    out_root = Path(args.save_root) / args.sae_tag
    print(f"Running singleblock SAE '{args.sae_tag}' -> output {out_root}")

    generate_for_sae(
        pipe=pipe,
        sae_tag=args.sae_tag,
        hook_fn=hook_fn,
        hookpoint=args.single_hookpoint,
        rows=rows,
        out_root=out_root,
        steps=args.steps,
        guidance=args.guidance_scale,
        n_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        neg_prompt=args.negative_prompt,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    main()
