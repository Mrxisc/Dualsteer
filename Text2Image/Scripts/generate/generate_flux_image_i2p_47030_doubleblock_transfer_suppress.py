import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    parser = argparse.ArgumentParser(description="Generate I2P images with Flux + SAE steering")
    parser.add_argument("--flux-ckpt", required=True, help="Diffusers checkpoint for Flux model")
    parser.add_argument("--prompts-csv", default=str(ROOT_DIR / "Datasets" / "i2p_benchmark.csv"))
    parser.add_argument(
        "--save-root",
        default=str(ROOT_DIR / "Results" / "i2p_flux_doubleblock_transfer_suppress"),
    )
    parser.add_argument("--sae-base", default=str(ROOT_DIR / "SAEs"))
    parser.add_argument(
        "--activations-root",
        default=str(ROOT_DIR / "Activations" / "i2p_no_sexual_flux_real"),
        help="Used to locate token_indices.pt / channel_proj.pt when needed.",
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
        default=0.8,
        help="Blend strength for SAE intervention (0=no-op, 1=full replacement). Default=0.8.",
    )
    parser.add_argument(
        "--mode",
        default="repel",
        choices=["blend", "repel"],
        help=(
            "Intervention mode: 'blend' moves toward the (transfer) SAE reconstruction manifold; "
            "'repel' pushes away from SAE reconstruction(s) to suppress learned concepts."
        ),
    )


    parser.add_argument(
        "--double-hookpoint",
        default="transformer.double_transformer_blocks.18",
        help="Doubleblock module hookpoint (collection-time name).",
    )
    parser.add_argument(
        "--source-double-sae-root",
        default=str(ROOT_DIR / "SAEs" / "flux_real_doubleblock18_i2p_no_sexual_i2p_no_sexual_flux_real"),
        help="Source-domain doubleblock SAE root (contains per-hookpoint subdir).",
    )
    parser.add_argument(
        "--target-double-sae-root",
        default=str(ROOT_DIR / "SAEs" / "flux_doubleblock18_sexual_transfer"),
        help="Target-domain (transfer) doubleblock SAE root (contains per-hookpoint subdir).",
    )
    parser.add_argument(
        "--source-sae-dir",
        default=str(ROOT_DIR / "SAEs" / "flux_text_encoder_i2p_no_sexual_activations" / "flux_model.text_encoder"),
        help="Source-domain text-encoder SAE directory (contains sae.safetensors)",
    )
    parser.add_argument(
        "--target-sae-dir",
        default=str(ROOT_DIR / "SAEs" / "flux_text_encoder_sexual_transfer" / "flux_model.text_encoder"),
        help="Target-domain (transfer) text-encoder SAE directory (contains sae.safetensors)",
    )
    parser.add_argument(
        "--suppress-source",
        type=float,
        default=0.1,
        help="Repel strength for the SOURCE SAE (0 disables). Used only when --mode=repel.",
    )
    parser.add_argument(
        "--suppress-target",
        type=float,
        default=0.2,
        help="Repel strength for the TARGET/transfer SAE (0 disables). Used only when --mode=repel.",
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
        help="Verify that the doubleblock hook changes activations (mean abs delta > 0).",
    )
    parser.add_argument(
        "--hooks",
        default="doubleblock",
        help="Comma-separated hook modes to run; supported: doubleblock",
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
        if parts and parts[0] in {"transformer", "flux_model"}:
            rest = ".".join(parts[1:])
            if rest:
                candidates.append(rest)


        if parts and parts[0] == "flux_model":
            rest = ".".join(parts[1:])
            if rest:
                candidates.append(f"transformer.{rest}")


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


def _maybe_load_token_indices(activations_root: str, hookpoint: str, device: torch.device) -> Optional[torch.Tensor]:
    candidates = [
        Path(activations_root) / hookpoint / "token_indices.pt",
        Path(activations_root) / "token_indices.pt",
        Path(activations_root).parent / "token_indices.pt",
        Path(activations_root).parent / "i2p_no_sexual_flux" / "token_indices.pt",
        Path(activations_root).parent / "i2p_sexual_flux" / "token_indices.pt",
    ]
    for c in candidates:
        if c.exists():
            idx = torch.load(c, map_location="cpu").to(torch.long)
            return idx.to(device=device)
    return None


def _maybe_load_channel_proj(activations_root: str, device: torch.device) -> Optional[torch.Tensor]:
    candidates = [
        Path(activations_root) / "channel_proj.pt",
        Path(activations_root).parent / "channel_proj.pt",
        Path(activations_root).parent / "i2p_no_sexual_flux" / "channel_proj.pt",
        Path(activations_root).parent / "i2p_sexual_flux" / "channel_proj.pt",
        Path(activations_root).parent / "i2p_no_sexual_flux_real" / "channel_proj.pt",
    ]
    for c in candidates:
        if c.exists():
            proj = torch.load(c, map_location="cpu")
            if not isinstance(proj, torch.Tensor):
                proj = torch.tensor(proj)
            return proj.to(device=device)
    return None


def _apply_sae_to_hidden(
    hidden: torch.Tensor,
    sae: Sae,
    *,
    token_indices: Optional[torch.Tensor],
    channel_proj: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden.ndim < 2:
        return hidden


    orig_shape = hidden.shape
    tokens = hidden.shape[-2]
    d_in = hidden.shape[-1]
    flat = hidden.reshape(-1, tokens, d_in)


    sae_d_in = None
    if getattr(sae, "cfg", None) is not None and hasattr(sae.cfg, "d_in"):
        try:
            sae_d_in = int(getattr(sae.cfg, "d_in"))
        except Exception:
            sae_d_in = None
    if sae_d_in is None and hasattr(sae, "b_dec") and isinstance(getattr(sae, "b_dec"), torch.Tensor):
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
            raise RuntimeError(f"SAE expects d_in={sae_d_in} but hooked tensor has d_in={d_in}; missing channel_proj.pt")


    def _run_sae(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        bs, sample_size, _ = x.shape
        sae_in, x_mean, x_std = sae.preprocess_input(x)
        pre_acts = sae.pre_acts(sae_in)
        top_acts, top_indices = sae.select_topk(pre_acts, batch_size=bs, k=sae.cfg.k)
        recon = sae.decode(top_acts, top_indices)
        recon = sae.postprocess_output(recon, x_mean, x_std)
        return recon.view(bs, sample_size, out_dim)


    if token_indices is not None and token_indices.numel() > 0:
        idx = token_indices.to(device=flat.device)
        if int(idx.max()) < tokens and idx.numel() < tokens:
            picked = torch.index_select(flat, dim=1, index=idx)
            x_in = picked @ proj if proj is not None else picked
            recon = _run_sae(x_in, sae_d_in if sae_d_in is not None else int(d_in))
            if proj is not None:
                recon = recon @ proj.transpose(0, 1)
            out = flat.clone()
            out[:, idx, :] = recon
            return out.reshape(orig_shape)


    x_in = flat @ proj if proj is not None else flat
    recon = _run_sae(x_in, sae_d_in if sae_d_in is not None else int(d_in))
    if proj is not None:
        recon = recon @ proj.transpose(0, 1)
    return recon.reshape(orig_shape)


def _apply_repel_to_hidden(
    hidden: torch.Tensor,
    sae: Sae,
    *,
    token_indices: Optional[torch.Tensor],
    channel_proj: Optional[torch.Tensor],
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


    recon = _apply_sae_to_hidden(hidden, sae, token_indices=token_indices, channel_proj=channel_proj)
    recon = _sanitize(recon)
    base = _sanitize(hidden)
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


def _resolve_module_for_hook(pipe: FluxPipeline, hookpoint: str) -> Tuple[torch.nn.Module, str]:
    def _expand_candidates(p: str) -> list[str]:
        cands: list[str] = []
        if p:
            cands.append(p)
        if "double_transformer_blocks" in p:
            cands.append(p.replace("double_transformer_blocks", "transformer_blocks"))
        if p.startswith("flux_model."):
            cands.append(p.replace("flux_model.", ""))
            cands.append(p.replace("flux_model.", "transformer."))


        seen = set()
        out: list[str] = []
        for x in cands:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out


    last_err: Exception | None = None
    for cand in _expand_candidates(hookpoint):
        try:
            target = pipe
            for part in cand.split("."):
                if part.isdigit():
                    target = target[int(part)]
                else:
                    target = getattr(target, part)
            if not isinstance(target, torch.nn.Module):
                raise TypeError(f"Resolved object is not a module: {cand} -> {type(target)}")
            return target, cand
        except Exception as e:
            last_err = e
    raise AttributeError(f"Cannot resolve hookpoint '{hookpoint}'. Tried {_expand_candidates(hookpoint)}. Last error: {last_err}")


def _build_doubleblock_dual_repel_hook(
    source_sae: Optional[Sae],
    target_sae: Optional[Sae],
    *,
    token_indices: Optional[torch.Tensor],
    channel_proj: Optional[torch.Tensor],
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
                    stats["mean_abs_delta"] = float((x_out - x).abs().mean().detach().item())
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


def _build_text_encoder_hook(sae: Sae, strength: float):
    @torch.no_grad()
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]


        if isinstance(out, torch.Tensor):
            hidden = out
            pooled = None
            hidden_states = None
            attentions = None
            out_cls = None
        else:
            hidden = getattr(out, "last_hidden_state", None)
            pooled = getattr(out, "pooler_output", None)
            hidden_states = getattr(out, "hidden_states", None)
            attentions = getattr(out, "attentions", None)
            out_cls = out.__class__


        if hidden is None:
            raise RuntimeError("text_encoder output has no last_hidden_state; cannot apply 3D SAE hook")


        try:
            strength_f = float(strength)
        except Exception:
            strength_f = 1.0
        if strength_f <= 0.0:
            return out
        if strength_f > 1.0:
            strength_f = 1.0


        bs, seq_len, d_in = hidden.shape
        sae_in, x_mean, x_std = sae.preprocess_input(hidden)
        pre_acts = sae.pre_acts(sae_in)
        top_acts, top_indices = sae.select_topk(pre_acts, batch_size=bs, k=sae.cfg.k)
        hidden_recon = sae.decode(top_acts, top_indices)
        hidden_recon = sae.postprocess_output(hidden_recon, x_mean, x_std)
        hidden_recon = hidden_recon.view(bs, seq_len, d_in)


        hidden_out = hidden * (1.0 - strength_f) + hidden_recon * strength_f


        if out_cls is None:
            return hidden_out


        pooled_out = pooled
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


            try:
                final_ln = getattr(getattr(module, "text_model", None), "final_layer_norm", None)
            except Exception:
                final_ln = None
            if final_ln is not None:
                try:
                    pooled_recon = final_ln(pooled_recon)
                except Exception:
                    pass


            pooled_out = pooled * (1.0 - strength_f) + pooled_recon * strength_f


        try:
            out["last_hidden_state"] = hidden_out
        except Exception:
            try:
                setattr(out, "last_hidden_state", hidden_out)
            except Exception:
                pass


        if pooled_out is not None:
            try:
                out["pooler_output"] = pooled_out
            except Exception:
                try:
                    setattr(out, "pooler_output", pooled_out)
                except Exception:
                    pass


        return out


    return hook


def _build_text_encoder_repel_hook(
    source_sae: Optional[Sae],
    target_sae: Optional[Sae],
    *,
    suppress_source: float,
    suppress_target: float,
    stats: Optional[dict] = None,
):
    @torch.no_grad()
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]


        if isinstance(out, torch.Tensor):
            hidden = out
            pooled = None
            out_obj = None
        else:
            hidden = getattr(out, "last_hidden_state", None)
            pooled = getattr(out, "pooler_output", None)
            out_obj = out


        if hidden is None:
            raise RuntimeError("text_encoder output has no last_hidden_state; cannot apply repulsion hook")


        ss = float(suppress_source)
        st = float(suppress_target)
        if ss <= 0.0 and st <= 0.0:
            return out


        bs, seq_len, d_in = hidden.shape
        hidden_out = hidden


        if source_sae is not None and ss > 0.0:
            sae_in, x_mean, x_std = source_sae.preprocess_input(hidden_out)
            pre_acts = source_sae.pre_acts(sae_in)
            top_acts, top_indices = source_sae.select_topk(pre_acts, batch_size=bs, k=source_sae.cfg.k)
            recon = source_sae.decode(top_acts, top_indices)
            recon = source_sae.postprocess_output(recon, x_mean, x_std)
            recon = recon.view(bs, seq_len, d_in)
            hidden_out = hidden_out - ss * (recon - hidden_out)


        if target_sae is not None and st > 0.0:
            sae_in, x_mean, x_std = target_sae.preprocess_input(hidden_out)
            pre_acts = target_sae.pre_acts(sae_in)
            top_acts, top_indices = target_sae.select_topk(pre_acts, batch_size=bs, k=target_sae.cfg.k)
            recon = target_sae.decode(top_acts, top_indices)
            recon = target_sae.postprocess_output(recon, x_mean, x_std)
            recon = recon.view(bs, seq_len, d_in)
            hidden_out = hidden_out - st * (recon - hidden_out)


        if out_obj is None:
            return hidden_out


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
            pooled_new = hidden_out[torch.arange(bs, device=hidden_out.device), eos_pos]
        else:
            pooled_new = hidden_out.mean(dim=1)


        try:
            final_ln = getattr(getattr(module, "text_model", None), "final_layer_norm", None)
        except Exception:
            final_ln = None
        if final_ln is not None:
            try:
                pooled_new = final_ln(pooled_new)
            except Exception:
                pass


        if stats is not None and "pooler_mean_abs_delta" not in stats:
            try:
                if isinstance(pooled, torch.Tensor):
                    stats["pooler_mean_abs_delta"] = float((pooled_new - pooled).abs().mean().detach().item())
                else:
                    stats["pooler_mean_abs_delta"] = float(pooled_new.abs().mean().detach().item())
            except Exception:
                pass


        try:
            out_obj["last_hidden_state"] = hidden_out
        except Exception:
            try:
                setattr(out_obj, "last_hidden_state", hidden_out)
            except Exception:
                pass


        try:
            out_obj["pooler_output"] = pooled_new
        except Exception:
            try:
                setattr(out_obj, "pooler_output", pooled_new)
            except Exception:
                pass


        return out_obj


    return hook


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
    hook_module: torch.nn.Module,
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


            handle = hook_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
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


    requested_hooks = [h.strip() for h in args.hooks.split(",") if h.strip()]


    sae_specs = [
        {
            "tag": "doubleblock",
            "hook": "doubleblock",
            "hook_builder": "doubleblock",
        },
    ]


    sae_specs = [s for s in sae_specs if s["hook"] in requested_hooks]


    print(f"Loading prompts from {args.prompts_csv}")
    df = pd.read_csv(args.prompts_csv)
    rows = _iter_rows(df)
    if args.max_prompts and args.max_prompts > 0:
        rows = rows[: args.max_prompts]
    print(f"Total prompts: {len(rows)} (expect ~4703; each generates {args.num_images_per_prompt} images)")


    print(f"Loading Flux pipeline from {args.flux_ckpt}")
    pipe = FluxPipeline.from_pretrained(args.flux_ckpt, torch_dtype=dtype).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe.set_progress_bar_config(disable=True)


    for spec in sae_specs:
        if spec["hook"] not in requested_hooks:
            continue
        if spec["hook_builder"] != "doubleblock":
            continue


        mode = str(args.mode).strip().lower()
        hook_stats: dict = {}


        source_sae = None
        target_sae = None
        if float(args.suppress_source) > 0:
            src_dir = _resolve_checkpoint_dir(args.source_double_sae_root, args.double_hookpoint)
            if src_dir is None:
                raise FileNotFoundError(f"Missing SOURCE doubleblock SAE under {args.source_double_sae_root} for {args.double_hookpoint}")
            source_sae = Sae.load_from_disk(src_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        if float(args.suppress_target) > 0:
            tgt_dir = _resolve_checkpoint_dir(args.target_double_sae_root, args.double_hookpoint)
            if tgt_dir is None:
                raise FileNotFoundError(f"Missing TARGET doubleblock SAE under {args.target_double_sae_root} for {args.double_hookpoint}")
            target_sae = Sae.load_from_disk(tgt_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        for s in (source_sae, target_sae):
            if s is not None:
                s.cfg.batch_topk = False
                s.cfg.sample_topk = False


        if mode != "repel":
            raise RuntimeError("This batch script currently supports only --mode=repel for doubleblock")
        if source_sae is None and target_sae is None:
            raise RuntimeError("--mode=repel requires at least one of --suppress-source/--suppress-target > 0")


        device = pipe._execution_device if hasattr(pipe, "_execution_device") else pipe.device
        token_indices = _maybe_load_token_indices(args.activations_root, args.double_hookpoint, device=device)
        channel_proj = _maybe_load_channel_proj(args.activations_root, device=device)


        hook_fn = _build_doubleblock_dual_repel_hook(
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


        hook_module, resolved = _resolve_module_for_hook(pipe, args.double_hookpoint)
        sae_tag = "doubleblock_repel"
        out_root = Path(args.save_root) / sae_tag
        print(f"Running mode={mode} tag='{sae_tag}' hook='{resolved}' -> output {out_root}")


        if args.verify and rows:
            prompt0 = rows[0]["prompt"]
            g0 = torch.Generator(device=device)
            g0.manual_seed(int(args.seed_base) + int(rows[0]["case_number"]))
            h = hook_module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
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
            print(f"[verify] doubleblock mean abs delta (first observed): {delta}")
            if delta is None or float(delta) == 0.0:
                raise RuntimeError(
                    "Verification failed: doubleblock repel hook did not change the activation (mean_abs_delta == 0)."
                )


        generate_for_sae(
            pipe=pipe,
            sae_tag=sae_tag,
            hook_fn=hook_fn,
            hook_module=hook_module,
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
