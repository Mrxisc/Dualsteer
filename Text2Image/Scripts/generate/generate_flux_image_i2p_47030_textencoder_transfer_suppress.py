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
        default=str(ROOT_DIR / "Results" / "i2p_flux_text_encoder_transfer"),
    )
    parser.add_argument("--sae-base", default=str(ROOT_DIR / "SAEs"))
    parser.add_argument("--activations-root", default=str(ROOT_DIR / "Activations" / "i2p_no_sexual_flux"))
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
        default=0.6,
        help="Repel strength for the SOURCE SAE (0 disables). Used only when --mode=repel.",
    )
    parser.add_argument(
        "--suppress-target",
        type=float,
        default=0.6,
        help="Repel strength for the TARGET/transfer SAE (0 disables). Used only when --mode=repel.",
    )
    parser.add_argument(
        "--verify-hook",
        default=True,
        type=lambda x: str(x).lower() not in {"0", "false", "no", "off"},
        help=(
            "For --mode=repel: verify that CLIP pooler_output actually changes (Flux depends on it). "
            "If the measured mean abs delta is 0, abort early. Default: True."
        ),
    )
    parser.add_argument(
        "--hooks",
        default="text_encoder",
        help="Comma-separated hook modes to run; currently supported: text_encoder",
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

            # Apply CLIP final layer norm to match CLIPTextModel pooler_output computation.
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

        # Mutate ModelOutput in-place when possible (robust across transformers versions).
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

        # Repel from each SAE's reconstruction manifold (sequentially on hidden_out):
        #   x' = x - s*(sae(x)-x)
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

        # Recompute pooler_output from modified hidden_out (Flux uses pooler_output heavily).
        # Always recompute pooler_output from modified hidden_out (Flux depends on it).
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

            # base seed per prompt
            base_seed = seed_base + case_number

            # identify already present
            existing_case = existing.get(case_key, set())

            # generate list of missing indices
            missing = [i for i in range(n_per_prompt) if i not in existing_case]
            if not missing:
                continue

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

    requested_hooks = [h.strip() for h in args.hooks.split(",") if h.strip()]

    # Prepare SAEs (transfer text encoder SAE)
    sae_specs = [
        {
            "tag": "text_encoder",
            "hook": "text_encoder",
            "sae_dir": Path(args.sae_base) / "flux_text_encoder_sexual_transfer" / "flux_model.text_encoder",
            "hook_builder": "text_encoder",
        },
    ]

    # Filter by requested hooks
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
        if spec["hook_builder"] != "text_encoder":
            print(f"[skip] Unsupported hook_builder={spec['hook_builder']} for tag={spec['tag']}")
            continue

        mode = str(args.mode).strip().lower()
        if mode == "repel":
            source_sae = _load_sae(Path(args.source_sae_dir)) if float(args.suppress_source) > 0 else None
            target_sae = _load_sae(Path(args.target_sae_dir)) if float(args.suppress_target) > 0 else None
            if source_sae is None and target_sae is None:
                raise RuntimeError("--mode=repel requires at least one of --suppress-source/--suppress-target > 0")
            for s in (source_sae, target_sae):
                if s is not None:
                    s.cfg.batch_topk = False
                    s.cfg.sample_topk = False
            hook_stats: dict = {}
            hook_fn = _build_text_encoder_repel_hook(
                source_sae,
                target_sae,
                suppress_source=float(args.suppress_source),
                suppress_target=float(args.suppress_target),
                stats=hook_stats,
            )
            sae_tag = "text_encoder_repel"
        else:
            sae = _load_sae(spec["sae_dir"])
            if sae is None:
                print(f"[skip] SAE '{spec['tag']}' not available; dir={spec['sae_dir']}")
                continue
            sae.cfg.batch_topk = False
            sae.cfg.sample_topk = False
            hook_fn = _build_text_encoder_hook(sae, strength=args.sae_strength)
            sae_tag = spec["tag"]

        out_root = Path(args.save_root) / sae_tag
        print(f"Running mode={mode} tag='{sae_tag}' -> output {out_root}")
        generate_for_sae(
            pipe=pipe,
            sae_tag=sae_tag,
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
        )

        if mode == "repel" and args.verify_hook:
            delta = None
            try:
                delta = hook_stats.get("pooler_mean_abs_delta")
            except Exception:
                delta = None
            print(f"[verify] text-encoder pooler mean abs delta (first observed): {delta}")
            if delta is None or float(delta) == 0.0:
                raise RuntimeError(
                    "Verification failed: repulsion hook did not change CLIP pooler_output (mean abs delta == 0). "
                    "Flux generation would likely be unaffected; check hook registration and SAE paths."
                )


if __name__ == "__main__":
    main()
