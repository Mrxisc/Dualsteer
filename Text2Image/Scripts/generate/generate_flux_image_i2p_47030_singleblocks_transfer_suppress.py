import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
from diffusers import FluxPipeline
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
    parser = argparse.ArgumentParser(description="Generate I2P images with Flux + singleblock SAE protection")
    parser.add_argument("--flux-ckpt", required=True, help="Diffusers checkpoint for Flux model")
    parser.add_argument(
        "--prompts-csv",
        default=str(ROOT_DIR / "Datasets" / "i2p_benchmark.csv"),
    )
    parser.add_argument(
        "--save-root",
        default=str(ROOT_DIR / "Results" / "i2p_flux_singleblock_transfer_suppress"),
    )
    parser.add_argument(
        "--source-sae-dir",
        default=str(ROOT_DIR / "SAEs" / "flux_real_singleblock37_i2p_no_sexual_i2p_no_sexual_flux_real"),
        help="SOURCE-domain singleblock SAE root (contains per-hookpoint subdir; set empty to disable)",
    )
    parser.add_argument(
        "--target-sae-dir",
        default=str(ROOT_DIR / "SAEs" / "flux_singleblock37_sexual_transfer"),
        help="TARGET-domain (transfer) singleblock SAE root (contains per-hookpoint subdir; set empty to disable)",
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
        default="singleblock37_repel",
        help="Tag name for output subdir and manifest (e.g., singleblock37_repel)",
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
        "--mode",
        default="repel",
        choices=["blend", "repel"],
        help=(
            "Intervention mode: 'blend' moves toward TARGET SAE reconstruction manifold; "
            "'repel' pushes away from SOURCE and/or TARGET reconstruction(s)."
        ),
    )
    parser.add_argument(
        "--sae-strength",
        type=float,
        default=0.2,
        help="Blend strength for --mode=blend (0=no-op, 1=full replacement). Default=0.2 to reduce garbling.",
    )
    parser.add_argument(
        "--suppress-source",
        type=float,
        default=0.1,
        help="Repel strength for SOURCE SAE (0 disables). Used only when --mode=repel.",
    )
    parser.add_argument(
        "--suppress-target",
        type=float,
        default=0.2,
        help="Repel strength for TARGET/transfer SAE (0 disables). Used only when --mode=repel.",
    )
    parser.add_argument(
        "--repel-clamp-mult",
        type=float,
        default=9.0,
        help="Stabilizer for repel: clamp delta per token to +/- (mult * std(hidden)).",
    )
    parser.add_argument(
        "--repel-nan-to-num",
        default=True,
        type=lambda x: str(x).lower() not in {"0", "false", "no", "off"},
        help="Replace NaN/Inf with 0 during repel (default: True).",
    )
    parser.add_argument(
        "--verify",
        default=True,
        type=lambda x: str(x).lower() not in {"0", "false", "no", "off"},
        help="Verify that the singleblock hook changes activations (mean abs delta > 0).",
    )


    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log path. If not set, writes run.log under <save-root>/<sae-tag>/.",
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


def _apply_repel_to_hidden(
    hidden: torch.Tensor,
    sae: Sae,
    *,
    token_indices: Optional[torch.Tensor] = None,
    channel_proj: Optional[torch.Tensor] = None,
    suppress: float,
    clamp_mult: float,
    nan_to_num: bool,
) -> torch.Tensor:
    try:
        s = float(suppress)
    except Exception:
        s = 0.0
    if s <= 0.0:
        return hidden


    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        if not nan_to_num:
            return x
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


    base = _sanitize(hidden)
    recon = _sanitize(
        _apply_sae_to_hidden(
            base,
            sae,
            token_indices=token_indices,
            channel_proj=channel_proj,
            strength=1.0,
        )
    )
    delta = _sanitize(recon - base)


    try:
        cm = float(clamp_mult)
    except Exception:
        cm = 0.0
    if cm > 0.0:
        std = base.float().std(dim=-1, keepdim=True).to(dtype=base.dtype)
        clip = (std * cm).clamp_min(1e-6)
        delta = torch.minimum(torch.maximum(delta, -clip), clip)


    out = base - s * delta
    return _sanitize(out)


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


def _build_tensor_dual_repel_hook(
    source_sae: Optional[Sae],
    target_sae: Optional[Sae],
    *,
    token_indices: Optional[torch.Tensor] = None,
    channel_proj: Optional[torch.Tensor] = None,
    suppress_source: float,
    suppress_target: float,
    repel_clamp_mult: float,
    repel_nan_to_num: bool,
    stats: Optional[dict] = None,
):
    @torch.no_grad()
    def hook(module, inp, out):
        ss = float(suppress_source)
        st = float(suppress_target)
        if ss <= 0.0 and st <= 0.0:
            return out


        sae_d_in = None
        for s in (source_sae, target_sae):
            if s is not None and hasattr(s, "b_dec") and isinstance(getattr(s, "b_dec"), torch.Tensor):
                sae_d_in = int(s.b_dec.numel())
                break


        def _apply(x: torch.Tensor) -> torch.Tensor:
            x0 = x
            x_out = x
            if source_sae is not None and ss > 0.0:
                x_out = _apply_repel_to_hidden(
                    x_out,
                    source_sae,
                    token_indices=token_indices,
                    channel_proj=channel_proj,
                    suppress=ss,
                    clamp_mult=repel_clamp_mult,
                    nan_to_num=repel_nan_to_num,
                )
            if target_sae is not None and st > 0.0:
                x_out = _apply_repel_to_hidden(
                    x_out,
                    target_sae,
                    token_indices=token_indices,
                    channel_proj=channel_proj,
                    suppress=st,
                    clamp_mult=repel_clamp_mult,
                    nan_to_num=repel_nan_to_num,
                )
            if stats is not None and "mean_abs_delta" not in stats:
                try:
                    stats["mean_abs_delta"] = float((x_out - x0).abs().mean().detach().item())
                except Exception:
                    pass
            return x_out


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
                out_list[target_idx] = _apply(out_list[target_idx])
                return tuple(out_list)
            return out


        if isinstance(out, torch.Tensor):
            return _apply(out)


        return out


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


def _append_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


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


            base_seed = seed_base + case_number


            existing_case = existing.get(case_key, set())


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


    out_root = Path(args.save_root) / args.sae_tag
    out_root.mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        log_path = out_root / "run.log"
    else:
        log_path = Path(args.log_file)
    _append_log(log_path, "argv: " + " ".join(sys.argv))


    print(f"Loading prompts from {args.prompts_csv}")
    _append_log(log_path, f"Loading prompts from {args.prompts_csv}")
    df = pd.read_csv(args.prompts_csv)
    rows = _iter_rows(df)
    if args.max_prompts and args.max_prompts > 0:
        rows = rows[: int(args.max_prompts)]
    print(f"Total prompts: {len(rows)} (expect ~4703; each generates {args.num_images_per_prompt} images)")
    _append_log(log_path, f"Total prompts: {len(rows)}")


    print(f"Loading Flux pipeline from {args.flux_ckpt}")
    _append_log(log_path, f"Loading Flux pipeline from {args.flux_ckpt}")
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
    mode = str(args.mode).strip().lower()


    token_indices = _maybe_load_token_indices(args.activations_root, device=device)
    channel_proj = _maybe_load_channel_proj(args.activations_root, device=device)


    source_sae = None
    target_sae = None
    if mode == "repel":
        if args.source_sae_dir and float(args.suppress_source) > 0.0:
            src_path = _resolve_checkpoint_dir(args.source_sae_dir, args.single_hookpoint)
            if src_path is None:
                raise FileNotFoundError(
                    f"Cannot find sae.safetensors under {args.source_sae_dir} (hookpoint={args.single_hookpoint})"
                )
            source_sae = Sae.load_from_disk(src_path, device=device).eval()
            source_sae.cfg.batch_topk = False
            source_sae.cfg.sample_topk = False


        if args.target_sae_dir and float(args.suppress_target) > 0.0:
            tgt_path = _resolve_checkpoint_dir(args.target_sae_dir, args.single_hookpoint)
            if tgt_path is None:
                raise FileNotFoundError(
                    f"Cannot find sae.safetensors under {args.target_sae_dir} (hookpoint={args.single_hookpoint})"
                )
            target_sae = Sae.load_from_disk(tgt_path, device=device).eval()
            target_sae.cfg.batch_topk = False
            target_sae.cfg.sample_topk = False


        if source_sae is None and target_sae is None:
            raise RuntimeError("--mode=repel requires at least one of --suppress-source/--suppress-target > 0")


        hook_stats: dict = {}
        hook_fn = _build_tensor_dual_repel_hook(
            source_sae,
            target_sae,
            token_indices=token_indices,
            channel_proj=channel_proj,
            suppress_source=float(args.suppress_source),
            suppress_target=float(args.suppress_target),
            repel_clamp_mult=float(args.repel_clamp_mult),
            repel_nan_to_num=bool(args.repel_nan_to_num),
            stats=hook_stats,
        )
        args.sae_tag = args.sae_tag or "singleblock37_repel"


        if args.verify and rows:
            prompt0 = rows[0]["prompt"]
            g0 = torch.Generator(device=device)
            g0.manual_seed(int(args.seed_base) + int(rows[0]["case_number"]))
            target_module = _resolve_module_for_hook(pipe, args.single_hookpoint)
            h = target_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
            try:
                _ = pipe(
                    prompt=[prompt0],
                    negative_prompt=[args.negative_prompt] if args.negative_prompt else None,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=[g0],
                    output_type="pil",
                )
            finally:
                h.remove()
            delta = hook_stats.get("mean_abs_delta")
            msg = f"[verify] singleblock mean abs delta (first observed): {delta}"
            print(msg)
            _append_log(log_path, msg)
            if delta is None or float(delta) == 0.0:
                raise RuntimeError("Verification failed: singleblock repel hook did not change activations.")


    else:
        if not args.target_sae_dir:
            raise RuntimeError("--mode=blend requires --target-sae-dir")
        tgt_path = _resolve_checkpoint_dir(args.target_sae_dir, args.single_hookpoint)
        if tgt_path is None:
            raise FileNotFoundError(
                f"Cannot find sae.safetensors under {args.target_sae_dir} (hookpoint={args.single_hookpoint})"
            )
        target_sae = Sae.load_from_disk(tgt_path, device=device).eval()
        target_sae.cfg.batch_topk = False
        target_sae.cfg.sample_topk = False
        hook_fn = _build_tensor_sae_hook(
            target_sae,
            token_indices=token_indices,
            channel_proj=channel_proj,
            strength=args.sae_strength,
        )


    print(f"Running singleblock run tag='{args.sae_tag}' mode={mode} -> output {out_root}")
    _append_log(log_path, f"Running tag={args.sae_tag} mode={mode}")


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


    _append_log(log_path, "done")


if __name__ == "__main__":
    main()
