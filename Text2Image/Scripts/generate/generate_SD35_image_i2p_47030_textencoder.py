import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(description="Generate I2P images with SD3.5 + text-encoder SAE steering")
    parser.add_argument("--sd35-ckpt", required=True, help="Diffusers checkpoint for SD3.5 model")
    parser.add_argument("--prompts-csv", default=str(ROOT_DIR / "Datasets" / "i2p_benchmark.csv"))
    parser.add_argument("--save-root", default=str(ROOT_DIR / "Results"))
    parser.add_argument(
        "--sae-root",
        default=str(ROOT_DIR / "SAEs" / "sd35_text_encoder_i2p_no_sexual_activations"),
        help=(
            "Path to SD3.5 text-encoder SAE root. Can be either the run root (containing text_encoder/) "
            "or the leaf directory (containing sae.safetensors)."
        ),
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
        default=1.0,
        help="Interpolation strength: x' = x + strength*(sae(x)-x). Use >1 to amplify.",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=1,
        help=(
            "Verify SAE effect by generating 1 baseline + 1 SAE image (same seed) for the first N prompts. "
            "If identical, raises an error. Set 0 to disable."
        ),
    )
    return parser


def _resolve_text_encoder_sae_dir(sae_root: Path) -> Path:
    if (sae_root / "sae.safetensors").exists():
        return sae_root
    return sae_root / "text_encoder"


def _parse_dtype(dtype_str: str) -> torch.dtype:
    lut = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return lut[dtype_str]


def _build_text_encoder_hook(sae: Sae, strength: float):
    stats = {"calls": 0, "last_mean_abs_delta": None}


    @torch.no_grad()
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]


        if isinstance(out, torch.Tensor):
            hidden_states = None
            pooled = None
            attentions = None
            out_cls = None
            target = out
        else:
            hidden_states = getattr(out, "hidden_states", None)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = getattr(out, "pooled_output", None)
            attentions = getattr(out, "attentions", None)
            out_cls = out.__class__


            target = None
            if isinstance(hidden_states, (tuple, list)) and len(hidden_states) >= 2:
                target = hidden_states[-2]
            if target is None:
                target = getattr(out, "last_hidden_state", None)


        if target is None:
            raise RuntimeError("text_encoder output has no usable hidden states; cannot apply 3D SAE hook")


        bs, seq_len, d_in = target.shape
        sae_in, x_mean, x_std = sae.preprocess_input(target)
        pre_acts = sae.pre_acts(sae_in)
        top_acts, top_indices = sae.select_topk(pre_acts)
        hidden_recon = sae.decode(top_acts, top_indices)
        hidden_recon = sae.postprocess_output(hidden_recon, x_mean, x_std)
        hidden_recon = hidden_recon.view(bs, seq_len, d_in)


        if float(strength) != 1.0:
            hidden_recon = target + float(strength) * (hidden_recon - target)


        stats["calls"] += 1
        try:
            stats["last_mean_abs_delta"] = (hidden_recon - target).abs().mean().item()
        except Exception:
            stats["last_mean_abs_delta"] = None


        if out_cls is None:
            return hidden_recon


        pooled_recon = pooled
        if pooled is not None:
            input_ids = inp[0] if isinstance(inp, (tuple, list)) and len(inp) > 0 else None
            eos_token_id = getattr(getattr(module, "config", None), "eos_token_id", None)
            if isinstance(input_ids, torch.Tensor) and eos_token_id is not None and input_ids.ndim == 2:
                eos_mask = input_ids.eq(int(eos_token_id))
                if eos_mask.any():
                    pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bs, -1)
                    eos_pos = torch.where(eos_mask, pos, pos.new_full(pos.shape, -1)).max(dim=1).values
                    eos_pos = torch.where(eos_pos >= 0, eos_pos, eos_pos.new_full(eos_pos.shape, seq_len - 1))
                else:
                    eos_pos = input_ids.new_full((bs,), seq_len - 1)
                pooled_recon = hidden_recon[torch.arange(bs, device=hidden_recon.device), eos_pos]
            else:
                pooled_recon = hidden_recon.mean(dim=1)


            if float(strength) != 1.0:
                pooled_recon = pooled + float(strength) * (pooled_recon - pooled)


        data = None
        if hasattr(out, "to_dict"):
            try:
                data = out.to_dict()
            except Exception:
                data = None
        if data is None:
            try:
                data = dict(out)
            except Exception:
                data = {}


        data["last_hidden_state"] = hidden_recon
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) >= 2:
            hs = list(hidden_states)
            hs[-2] = hidden_recon
            data["hidden_states"] = tuple(hs)
        if pooled_recon is not None:
            if "pooler_output" in data:
                data["pooler_output"] = pooled_recon
            if "pooled_output" in data:
                data["pooled_output"] = pooled_recon


        try:
            return out_cls(**data)
        except TypeError:
            try:
                setattr(out, "last_hidden_state", hidden_recon)
                if isinstance(hidden_states, (tuple, list)) and len(hidden_states) >= 2 and hasattr(out, "hidden_states"):
                    hs = list(hidden_states)
                    hs[-2] = hidden_recon
                    setattr(out, "hidden_states", tuple(hs))
                if pooled_recon is not None:
                    if hasattr(out, "pooler_output"):
                        setattr(out, "pooler_output", pooled_recon)
                    if hasattr(out, "pooled_output"):
                        setattr(out, "pooled_output", pooled_recon)
                return out
            except Exception:
                return hidden_recon


    return hook


def _images_identical(img_a, img_b) -> bool:
    try:
        diff = ImageChops.difference(img_a.convert("RGB"), img_b.convert("RGB"))
        return diff.getbbox() is None
    except Exception:
        return False


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
    pipe: StableDiffusion3Pipeline,
    sae_tag: str,
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


        for row_idx, row in enumerate(tqdm(rows, desc=f"SAE={sae_tag}")):
            prompt = row["prompt"]
            case_number = row["case_number"]
            categories = row.get("categories", "")
            case_key = str(case_number)
            case_dir = images_root / f"case_{case_number}"
            case_dir.mkdir(parents=True, exist_ok=True)


            base_seed = seed_base + case_number


            existing_case = existing.get(case_key, set())


            missing = [i for i in range(n_per_prompt) if i not in existing_case]
            if not missing:
                continue


            if verify_n > 0 and row_idx < verify_n:
                verify_seed = base_seed + missing[0]
                g0 = torch.Generator(device=device).manual_seed(verify_seed)
                base = pipe(
                    prompt=prompt,
                    negative_prompt=neg_prompt if neg_prompt else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=g0,
                    output_type="pil",
                ).images[0]


                g1 = torch.Generator(device=device).manual_seed(verify_seed)
                handle_verify = pipe.text_encoder.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
                try:
                    steered = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt if neg_prompt else None,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=height,
                        width=width,
                        generator=g1,
                        output_type="pil",
                    ).images[0]
                finally:
                    handle_verify.remove()


                if _images_identical(base, steered):
                    raise RuntimeError(
                        f"Verification failed: SAE image identical to baseline for case_number={case_number}. "
                        f"Try increasing --sae-strength or verify the SAE matches SD3.5 text_encoder."
                    )


            handle = pipe.text_encoder.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
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
                handle.remove()


def main():
    args = _build_parser().parse_args()
    dtype = _parse_dtype(args.dtype)


    sae_dir = _resolve_text_encoder_sae_dir(Path(args.sae_root))


    print(f"Loading prompts from {args.prompts_csv}")
    df = pd.read_csv(args.prompts_csv)
    rows = _iter_rows(df)
    print(f"Total prompts: {len(rows)} (expect ~4703; each generates {args.num_images_per_prompt} images)")


    print(f"Loading SD3.5 pipeline from {args.sd35_ckpt}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.sd35_ckpt, torch_dtype=dtype).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe.set_progress_bar_config(disable=True)


    sae = _load_sae(sae_dir)
    if sae is None:
        raise FileNotFoundError(f"SD3.5 text-encoder SAE not found: {sae_dir}")
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    hook_fn = _build_text_encoder_hook(sae, strength=args.sae_strength)


    out_root = Path(args.save_root) / "i2p_sd35_text_encoder" / "text_encoder"
    print(f"Running SD3.5 text-encoder SAE -> output {out_root}")
    generate_for_sae(
        pipe=pipe,
        sae_tag="text_encoder",
        hook_fn=hook_fn,
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
        verify_n=args.verify_n,
    )


    stats = getattr(hook_fn, "stats", None)
    if isinstance(stats, dict):
        print(f"text_encoder_hook_calls={stats.get('calls')} last_mean_abs_delta={stats.get('last_mean_abs_delta')}")


if __name__ == "__main__":
    main()
