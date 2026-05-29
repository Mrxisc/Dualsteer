from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Dualsteer_Code/Text2Image.Scripts.transfer.fewshot_textencoder_sexual_finetune import (  # noqa: E402
    build_train_config,
    load_base_saes,
    load_dataset_dict_with_replay,
    export_saes,
    parse_dtype,
)
from Dualsteer_Code/Text2Image.Scripts.train.trainer import SaeTrainer  # noqa: E402


DEFAULT_MODEL_NAME = str(REPO_ROOT / "Models" / "FLUX.1-dev")
DEFAULT_SEXUAL_CSV = str(REPO_ROOT / "Datasets" / "i2p_sexual.csv")
DEFAULT_HOOKPOINT = "transformer.double_transformer_blocks.18"
DEFAULT_BASE_SAE_DIR = (
    "SAEs/flux_real_doubleblock18_i2p_no_sexual_i2p_no_sexual_flux_real"
)
DEFAULT_ACTIVATIONS_DIR = "Activations/i2p_sexual_flux"
DEFAULT_CANONICAL_ACTIVATION_DIRNAME = "flux_model.doubleblock18"
DEFAULT_OUTPUT_SAE_DIR = "SAEs/flux_doubleblock18_sexual_transfer"
DEFAULT_REPLAY_ACTIVATIONS_DIR = "Activations/i2p_no_sexual_flux_real"
DEFAULT_FLUX_CONFIG = "_Backup_code/configs/flux1_dev_real.json" 
DEFAULT_TOKEN_SUBSAMPLE = 256
DEFAULT_CHANNEL_PROJ_DIM = 1024
DEFAULT_TARGET_ACTIVATION_ROWS = 900

DEFAULT_LOG_DIR = "Logs"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _run_and_tee(cmd: list[str], *, log_path: Path, cwd: Path | None = None) -> None:
    """Run a subprocess while tee-ing stdout/stderr into log file + console."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n[subprocess] " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _abs_path(p: str) -> str:
    pth = Path(p).expanduser()
    if pth.is_absolute():
        return str(pth)
    return str((REPO_ROOT / pth).resolve())


def _hook_cache_ready(
    base_dir: str,
    hook: str,
    *,
    token_subsample: int | None,
    channel_proj_dim: int | None,
) -> bool:
    base = Path(base_dir)
    canonical_dir = base / DEFAULT_CANONICAL_ACTIVATION_DIRNAME
    hook_dir = canonical_dir if canonical_dir.exists() else (base / hook)
    info_path = hook_dir / "dataset_info.json"
    state_path = hook_dir / "state.json"
    meta_path = hook_dir / "wrapper_cache_meta.json"

    if not info_path.exists() or not state_path.exists() or not meta_path.exists():
        return False
    try:
        info = json.loads(info_path.read_text())
        feats = info.get("features", {})
        act = feats.get("activations", {})
        shape = act.get("shape", None)
        if isinstance(shape, list) and len(shape) == 2:
            n_tokens, n_channels = shape
            if token_subsample is not None and int(n_tokens) != int(token_subsample):
                return False
            if channel_proj_dim is not None and int(n_channels) != int(channel_proj_dim):
                return False
    except Exception:
        return False

    try:
        meta = json.loads(meta_path.read_text())
        if bool(meta.get("csv_deduplicate", True)) is not False:
            return False
        if bool(meta.get("csv_strip_period", True)) is not False:
            return False
        if token_subsample is not None and int(meta.get("token_subsample", -1)) != int(token_subsample):
            return False
        if channel_proj_dim is not None and int(meta.get("channel_proj_dim", -1)) != int(channel_proj_dim):
            return False
    except Exception:
        return False

    return True


def _collect_real_activations_if_needed(args: argparse.Namespace) -> None:
    cache_ok = _hook_cache_ready(
        args.activations_dir,
        args.hookpoint,
        token_subsample=args.token_subsample,
        channel_proj_dim=args.channel_proj_dim,
    )
    if args.force_cache or not cache_ok:
        base = Path(args.activations_dir)
        canonical_dir = base / DEFAULT_CANONICAL_ACTIVATION_DIRNAME
        legacy_dir = base / args.hookpoint
        if args.force_cache:
            if canonical_dir.exists() or canonical_dir.is_symlink():
                print(f"[cache-real] --force-cache set; removing existing cache at {canonical_dir}")
                if canonical_dir.is_symlink():
                    canonical_dir.unlink()
                else:
                    shutil.rmtree(canonical_dir)
            if legacy_dir.exists() or legacy_dir.is_symlink():
                print(f"[cache-real] --force-cache set; removing existing cache at {legacy_dir}")
                if legacy_dir.is_symlink():
                    legacy_dir.unlink()
                else:
                    shutil.rmtree(legacy_dir)
        else:
            if legacy_dir.exists() and not cache_ok:
                print(f"[cache-real] Removing incompatible/partial cache at {legacy_dir}")
                shutil.rmtree(legacy_dir)

        script_path = (REPO_ROOT / "Scripts" / "collect" / "collect_real_activations_diffusers.py").resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Missing collector script: {script_path}")

        channel_proj_path = Path(args.replay_activations_dir[0]) / "channel_proj.pt"
        if not channel_proj_path.exists():
            raise FileNotFoundError(f"Missing channel_proj.pt at {channel_proj_path} (needed to match source d_in)")

        csv_max_rows = None
        if args.max_cache_rows is not None:
            csv_max_rows = int(args.max_cache_rows)
        elif args.max_cache_examples is not None:
            csv_max_rows = int(args.max_cache_examples)
        cmd = [
            sys.executable,
            str(script_path),
            "--backend",
            "flux",
            "--model_name",
            args.model_name,
            "--hook_names",
            args.hookpoint,
            "--new_cached_activations_path",
            args.activations_dir,
            "--csv_path",
            args.sexual_csv,
            "--csv_prompt_column",
            args.csv_prompt_column,
            "--csv_category_column",
            args.csv_category_column,
            "--csv_filter_categories",
            "sexual",
            "--csv_deduplicate",
            "false",
            "--csv_strip_period",
            "false",
            "--seed",
            str(args.seed),
            "--batch_size",
            str(args.cache_batch_size),
            "--num_inference_steps",
            str(args.cache_steps),
            "--guidance_scale",
            str(args.cache_guidance_scale),
            "--height",
            "256",
            "--width",
            "256",
            "--cache_every_n_timesteps",
            str(args.cache_every_n),
            "--token_subsample",
            str(args.token_subsample),
            "--channel_proj_dim",
            str(args.channel_proj_dim),
            "--channel_proj_path",
            str(channel_proj_path),
            "--activation_part",
            "tuple1",
            "--cfg_keep",
            "cond",
            "--dtype",
            ("float16" if args.cache_dtype == "fp16" else args.cache_dtype),
        ]
        if csv_max_rows is not None:
            cmd += ["--csv_max_rows", str(csv_max_rows)]
        if args.cache_resume:
            cmd += ["--resume", "true"]

        print("[cache-real] Collecting target activations with real diffusers backend...")
        _run_and_tee(cmd, log_path=Path(args.log_file))
        if legacy_dir.exists() and not canonical_dir.exists():
            print(f"[cache-real] Moving cache {legacy_dir.name} -> {canonical_dir.name}")
            legacy_dir.replace(canonical_dir)
        elif legacy_dir.exists() and canonical_dir.exists():
            print(f"[cache-real] Replacing existing canonical cache at {canonical_dir}")
            shutil.rmtree(canonical_dir)
            legacy_dir.replace(canonical_dir)

        if legacy_dir.exists() or legacy_dir.is_symlink():
            if legacy_dir.is_symlink():
                legacy_dir.unlink()
            else:
                shutil.rmtree(legacy_dir)
        try:
            legacy_dir.symlink_to(canonical_dir, target_is_directory=True)
        except OSError as e:
            raise RuntimeError(
                "Failed to create symlink at legacy hookpoint path. "
                "This would otherwise duplicate activation data on disk. "
                f"Details: {e}"
            )

        meta_path = canonical_dir / "wrapper_cache_meta.json"
        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "csv_deduplicate": False,
            "csv_strip_period": False,
            "csv_max_rows": csv_max_rows,
            "cache_every_n_timesteps": int(args.cache_every_n),
            "token_subsample": int(args.token_subsample),
            "channel_proj_dim": int(args.channel_proj_dim),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(canonical_dir))
            n = len(ds)
            if n > int(args.target_activation_rows):
                print(f"[cache-real] Truncating activations: {n} -> {args.target_activation_rows}")
                ds = ds.select(range(int(args.target_activation_rows)))
                shutil.rmtree(canonical_dir)
                ds.save_to_disk(str(canonical_dir))
                meta_path = canonical_dir / "wrapper_cache_meta.json"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            elif n < int(args.target_activation_rows):
                raise RuntimeError(
                    f"Activation cache too small: got {n}, expected {args.target_activation_rows}. "
                    "Check your CSV size/filtering and caching settings."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to validate/enforce activation cache size at {canonical_dir}: {e}")
    else:
        print(f"[cache-real] Using cached activations in {args.activations_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot finetune Flux doubleblock SAE on i2p sexual subset")
    p.add_argument("--base-sae-dir", default=DEFAULT_BASE_SAE_DIR)
    p.add_argument("--hookpoint", default=DEFAULT_HOOKPOINT, help="Doubleblock hookpoint to finetune")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Flux checkpoint path or HF id")
    p.add_argument("--sexual-csv", default=DEFAULT_SEXUAL_CSV, help="CSV file with i2p prompts")
    p.add_argument("--activations-dir", default=DEFAULT_ACTIVATIONS_DIR)
    p.add_argument("--output-sae-dir", default=DEFAULT_OUTPUT_SAE_DIR)
    p.add_argument("--flux-config", default=DEFAULT_FLUX_CONFIG)
    p.add_argument(
        "--target-activation-rows",
        type=int,
        default=DEFAULT_TARGET_ACTIVATION_ROWS,
        help="Require exactly N activation rows in activations/<flux_model.doubleblock18>.",
    )

    p.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log file path. If not set, a timestamped log is created under log/.",
    )

    p.add_argument("--csv-prompt-column", default="prompt")
    p.add_argument("--csv-category-column", default="categories")
    p.add_argument("--csv-filter", nargs="+", default=["sexual"], help="Category keywords to keep")
    p.add_argument("--csv-match-all", action="store_true")

    p.add_argument("--max-cache-rows", type=int, default=None)
    p.add_argument("--max-cache-examples", type=int, default=30)
    p.add_argument("--cache-batch-size", type=int, default=1)
    p.add_argument("--cache-steps", type=int, default=30)
    p.add_argument("--cache-guidance-scale", type=float, default=4.0)
    p.add_argument("--cache-every-n", type=int, default=6)
    p.add_argument("--cache-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--force-cache", action="store_true")
    p.add_argument(
        "--cache-resume",
        default=True,
        type=lambda x: str(x).lower() not in {"0", "false", "no", "off"},
        help="Resume real-diffusers activation caching if partial shards exist (default: True)",
    )

    p.add_argument("--token-subsample", type=int, default=DEFAULT_TOKEN_SUBSAMPLE)
    p.add_argument("--channel-proj-dim", type=int, default=DEFAULT_CHANNEL_PROJ_DIM)
    p.add_argument(
        "--train-decoder-only",
        dest="train_decoder_only",
        action="store_true",
        default=True,
        help="Freeze SAE encoder and only finetune decoder (default: on; match text-encoder transfer).",
    )
    p.add_argument(
        "--no-train-decoder-only",
        dest="train_decoder_only",
        action="store_false",
        help="Finetune both encoder and decoder.",
    )

    p.add_argument(
        "--replay-activations-dir",
        nargs="+",
        default=[DEFAULT_REPLAY_ACTIVATIONS_DIR],
        help="Source-domain cached activations for replay (default: match text-encoder transfer).",
    )
    p.add_argument("--replay-ratio", type=float, default=0.2)
    p.add_argument("--max-replay-examples", type=int, default=4096)
    p.add_argument("--effective-batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-scheduler", default="constant")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--auxk-alpha", type=float, default=0.0)
    p.add_argument("--dead-feature-threshold", type=int, default=10_000_000)
    p.add_argument("--grad-acc-steps", type=int, default=1)
    p.add_argument("--micro-acc-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", default="dualsteer_finetune")
    p.add_argument("--wandb-log-frequency", type=int, default=4000)
    p.add_argument("--run-name", default=None)
    p.add_argument("--train-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-to-wandb", action="store_true")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-train-examples", type=int, default=256)
    p.add_argument("--train-hard-limit", type=int, default=None, help="Deprecated: use --max-train-examples")
    args = p.parse_args()

    args.model_name = _abs_path(args.model_name)
    args.sexual_csv = _abs_path(args.sexual_csv)
    args.base_sae_dir = _abs_path(args.base_sae_dir)
    args.activations_dir = _abs_path(args.activations_dir)
    args.output_sae_dir = _abs_path(args.output_sae_dir)
    args.flux_config = _abs_path(args.flux_config)
    if args.replay_activations_dir:
        args.replay_activations_dir = [_abs_path(p) for p in args.replay_activations_dir]
    if args.log_file is None:
        log_dir = Path(_abs_path(DEFAULT_LOG_DIR))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log_file = str(log_dir / f"transfer_flux_doubleblock18_{ts}.log")
    else:
        args.log_file = _abs_path(args.log_file)

    args.hookpoints = [args.hookpoint]
    args.use_flux = False
    args.use_dit = False
    args.dit_config = None
    args.denoiser_attr = None
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")
    log_fh.write("\n=== run started: " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
    log_fh.write("argv: " + " ".join(sys.argv) + "\n")
    log_fh.flush()
    sys.stdout = _Tee(sys.stdout, log_fh) 
    sys.stderr = _Tee(sys.stderr, log_fh) 
    print("=== Few-shot doubleblock sexual finetune (hardcoded defaults) ===")
    print(f"log_file={args.log_file}")
    print(f"model_name={args.model_name}")
    print(f"sexual_csv={args.sexual_csv}")
    print(f"hookpoint={args.hookpoint}")
    print(f"base_sae_dir={args.base_sae_dir}")
    print(f"activations_dir={args.activations_dir}")
    print(f"output_sae_dir={args.output_sae_dir}")
    print(f"replay_activations_dir={args.replay_activations_dir}")
    print(
        f"cache(real): batch_size={args.cache_batch_size} steps={args.cache_steps} guidance={args.cache_guidance_scale} "
        f"every_n={args.cache_every_n} token_subsample={args.token_subsample} channel_proj_dim={args.channel_proj_dim} dtype={args.cache_dtype}"
    )
    print(
        f"train: decoder_only={args.train_decoder_only} replay_ratio={args.replay_ratio} max_train={args.max_train_examples} max_replay={args.max_replay_examples}"
    )
    _collect_real_activations_if_needed(args)
    activations_dir = args.activations_dir

    replay_dirs = []
    if args.replay_activations_dir:
        replay_dirs = [Path(p).expanduser() for p in args.replay_activations_dir]
        for p in replay_dirs:
            if not p.exists():
                raise FileNotFoundError(f"Replay activations dir does not exist: {p}")

    dataset_dict = load_dataset_dict_with_replay(
        Path(activations_dir).expanduser(),
        replay_dirs,
        args.hookpoints,
        parse_dtype(args.train_dtype),
        primary_max_examples=args.max_train_examples or args.train_hard_limit,
        replay_max_examples=args.max_replay_examples,
        replay_ratio=args.replay_ratio,
        seed=args.seed,
    )

    base_saes = load_base_saes(Path(args.base_sae_dir).expanduser(), args.hookpoints, device)
    sae_cfg = next(iter(base_saes.values())).cfg
    train_cfg = build_train_config(args, sae_cfg)
    train_cfg.train_decoder_only = bool(args.train_decoder_only)
    trainer = SaeTrainer(train_cfg, dataset_dict)

    for hook, sae in trainer.saes.items():
        sae.load_state_dict(base_saes[hook].state_dict())
        print(f"[init] Warm-started trainer SAE for {hook}")

    trainer.fit()
    export_saes(trainer.saes, Path(args.output_sae_dir).expanduser())
    print(f"[OK] Exported SAE(s) under: {Path(args.output_sae_dir).expanduser()}")


if __name__ == "__main__":
    main()
