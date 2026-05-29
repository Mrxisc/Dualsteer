import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import ImageChops
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
    p = argparse.ArgumentParser(description="Generate I2P images with SD3.5 + MMDiT SAE steering")
    p.add_argument("--sd35-ckpt", required=True, help="Diffusers checkpoint for SD3.5 model")
    p.add_argument(
        "--prompts-csv",
        default=str(ROOT_DIR / "Datasets" / "i2p_benchmark.csv"),
        help="CSV containing at least a 'prompt' column",
    )
    p.add_argument(
        "--save-root",
        default=str(ROOT_DIR / "Results"),
        help="Base output dir; results go into i2p_sd35_mmdit/mmdit_block36/",
    )
    p.add_argument(
        "--sae-root",
        default=str(ROOT_DIR / "SAEs" / "sd35_block36_i2p_no_sexual_activations"),
        help=(
            "Path to SD3.5 MMDiT SAE root. Can be either the leaf directory (containing sae.safetensors) "
            "or a run root containing a per-hookpoint subdir."
        ),
    )
    p.add_argument(
        "--mmdit-hookpoint",
        default="transformer.transformer_blocks.36",
        help="Hookpoint module path for the SD3.5 transformer block to hook",
    )
    p.add_argument(
        "--activations-root",
        default=str(ROOT_DIR / "Activations" / "i2p_no_sexual_SD_real"),
        help="Activation cache root used to load token_indices.pt / channel_proj.pt (optional but recommended)",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-images-per-prompt", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument(
        "--sae-strength",
        type=float,
        default=1.0,
        help="Interpolation strength: x' = x + strength*(sae(x)-x). Use >1 to amplify.",
    )
    p.add_argument(
        "--verify-n",
        type=int,
        default=1,
        help=(
            "Verify SAE effect by generating 1 baseline + 1 SAE image (same seed) for the first N prompts. "
            "If identical, raises an error. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="For debugging: only process first N prompts (0 means no limit).",
    )
    return p


def _parse_dtype(dtype_str: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def _images_identical(img_a, img_b) -> bool:
    try:
        diff = ImageChops.difference(img_a.convert("RGB"), img_b.convert("RGB"))
        return diff.getbbox() is None
    except Exception:
        return False


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
    rows: List[dict] = []
    for idx, row in df.iterrows():
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append(
            {
                "prompt": prompt.strip(),
                "case_number": int(idx),
                "categories": row.get("categories", ""),
            }
        )
    return rows


def _maybe_load_token_indices(activations_root: str, device: torch.device) -> Optional[torch.Tensor]:
    token_path = Path(activations_root) / "token_indices.pt"
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
        return None
    proj = torch.load(proj_path, map_location="cpu")
    if not isinstance(proj, torch.Tensor):
        proj = torch.tensor(proj)
    return proj.to(device=device)


def _resolve_checkpoint_dir(checkpoint_root: str, hookpoint: str) -> Optional[Path]:
    root = Path(checkpoint_root)
    if (root / "sae.safetensors").exists():
        return root


    candidates: list[str] = []
    if hookpoint:
        candidates.append(hookpoint)
    if hookpoint.startswith("transformer."):
        candidates.append(hookpoint[len("transformer.") :])


    seen = set()
    for sub in candidates:
        if sub in seen:
            continue
        seen.add(sub)
        cand = root / sub
        if (cand / "sae.safetensors").exists():
            return cand
    return None


def _resolve_module(root, dotted_path: str):
    cur = root
    for part in dotted_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


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
            out[:, idx, :] = out[:, idx, :] + strength_f * (recon - out[:, idx, :])
            return out.reshape(orig_shape)


    x_in = flat
    if proj is not None:
        x_in = x_in @ proj
    recon = _run_sae(x_in, sae_d_in if sae_d_in is not None else int(d_in))
    if proj is not None:
        recon = recon @ proj.transpose(0, 1)
    blended = flat + strength_f * (recon - flat)
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
        if isinstance(out, tuple):
            out_list = list(out)
            target_idx = None
            for idx, elem in enumerate(out_list):
                if isinstance(elem, torch.Tensor) and elem.ndim >= 2:
                    target_idx = idx
                    break
            if target_idx is None:
                return out
            out_list[target_idx] = _apply_sae_to_hidden(
                out_list[target_idx],
                sae,
                token_indices=token_indices,
                channel_proj=channel_proj,
                strength=strength,
            )
            return tuple(out_list)


        if isinstance(out, torch.Tensor):
            return _apply_sae_to_hidden(
                out,
                sae,
                token_indices=token_indices,
                channel_proj=channel_proj,
                strength=strength,
            )


        return out


    return hook


def generate_for_sae(
    *,
    pipe: StableDiffusion3Pipeline,
    hook_module,
    hook_fn,
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
    verify_n: int,
):
    out_root.mkdir(parents=True, exist_ok=True)
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "flux_manifest.csv"
    existing = _load_manifest(manifest_path)
    manifest_has_header = manifest_path.exists() and manifest_path.stat().st_size > 0


    torch.set_grad_enabled(False)
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device


    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_COLUMNS)
        if not manifest_has_header:
            writer.writeheader()


        for row_idx, row in enumerate(tqdm(rows, desc="SAE=mmdit_block36")):
            prompt = row["prompt"]
            case_number = int(row["case_number"])
            categories = row.get("categories", "")
            case_key = str(case_number)


            case_dir = images_root / f"case_{case_number}"
            case_dir.mkdir(parents=True, exist_ok=True)


            base_seed = int(seed_base) + int(case_number)
            existing_case = existing.get(case_key, set())
            missing = [i for i in range(int(n_per_prompt)) if i not in existing_case]
            if not missing:
                continue


            if int(verify_n) > 0 and row_idx < int(verify_n):
                verify_seed = base_seed + int(missing[0])
                g0 = torch.Generator(device=device).manual_seed(int(verify_seed))
                base = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt if neg_prompt else None,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    height=int(height),
                    width=int(width),
                    generator=g0,
                    output_type="pil",
                ).images[0]


                g1 = torch.Generator(device=device).manual_seed(int(verify_seed))
                handle_verify = hook_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
                try:
                    steered = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt if neg_prompt else None,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        height=int(height),
                        width=int(width),
                        generator=g1,
                        output_type="pil",
                    ).images[0]
                finally:
                    handle_verify.remove()


                if _images_identical(base, steered):
                    raise RuntimeError(
                        f"Verification failed: SAE image identical to baseline for case_number={case_number}. "
                        f"Try increasing --sae-strength or verify the SAE matches SD3.5 MMDiT block."
                    )


            handle = hook_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
            try:
                for start in range(0, len(missing), int(batch_size)):
                    batch_idx = missing[start : start + int(batch_size)]
                    generators = []
                    for idx in batch_idx:
                        g = torch.Generator(device=device)
                        g.manual_seed(int(base_seed) + int(idx))
                        generators.append(g)


                    outputs = pipe(
                        prompt=[prompt] * len(batch_idx),
                        negative_prompt=[neg_prompt] * len(batch_idx) if neg_prompt else None,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        height=int(height),
                        width=int(width),
                        generator=generators,
                        output_type="pil",
                    )


                    for img, idx in zip(outputs.images, batch_idx):
                        fname = f"{case_number}_img{idx}.png"
                        img_path = case_dir / fname
                        img.save(img_path)
                        rel_path = Path("images") / f"case_{case_number}" / fname
                        writer.writerow(
                            {
                                "sample_index": case_number,
                                "case_number": case_number,
                                "prompt": prompt,
                                "categories": categories,
                                "sd_seed": int(base_seed) + int(idx),
                                "sd_guidance_scale": float(guidance),
                                "idx_generation": int(idx),
                                "image_path": str(rel_path),
                                "sae_tag": "mmdit_block36",
                            }
                        )
                    mf.flush()
            finally:
                handle.remove()


def main() -> None:
    args = _build_parser().parse_args()
    dtype = _parse_dtype(args.dtype)


    print(f"Loading prompts from {args.prompts_csv}")
    df = pd.read_csv(args.prompts_csv)
    if int(args.max_prompts) > 0:
        df = df.head(int(args.max_prompts)).copy()
    rows = _iter_rows(df)
    print(f"Total prompts: {len(rows)} (expect ~4703; each generates {args.num_images_per_prompt} images)")


    print(f"Loading SD3.5 pipeline from {args.sd35_ckpt}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.sd35_ckpt, torch_dtype=dtype).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe.set_progress_bar_config(disable=True)


    sae_dir = _resolve_checkpoint_dir(args.sae_root, args.mmdit_hookpoint)
    if sae_dir is None:
        raise FileNotFoundError(f"SD3.5 MMDiT SAE not found under: {args.sae_root} (hook={args.mmdit_hookpoint})")


    sae = Sae.load_from_disk(sae_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False


    device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device
    token_indices = _maybe_load_token_indices(args.activations_root, device=device)
    channel_proj = _maybe_load_channel_proj(args.activations_root, device=device)


    hook_fn = _build_tensor_sae_hook(
        sae,
        token_indices=token_indices,
        channel_proj=channel_proj,
        strength=float(args.sae_strength),
    )


    hook_module = _resolve_module(pipe, args.mmdit_hookpoint)


    out_root = Path(args.save_root) / "i2p_sd35_mmdit" / "mmdit_block36"
    print(f"Running SD3.5 MMDiT SAE -> output {out_root}")
    print(f"mmdit_hookpoint={args.mmdit_hookpoint} sae_dir={sae_dir}")
    if token_indices is not None:
        print(f"token_indices={tuple(token_indices.shape)}")
    if channel_proj is not None:
        print(f"channel_proj={tuple(channel_proj.shape)}")


    generate_for_sae(
        pipe=pipe,
        hook_module=hook_module,
        hook_fn=hook_fn,
        rows=rows,
        out_root=out_root,
        steps=int(args.steps),
        guidance=float(args.guidance_scale),
        n_per_prompt=int(args.num_images_per_prompt),
        batch_size=int(args.batch_size),
        height=int(args.height),
        width=int(args.width),
        neg_prompt=args.negative_prompt,
        seed_base=int(args.seed_base),
        verify_n=int(args.verify_n),
    )


if __name__ == "__main__":
    main()
