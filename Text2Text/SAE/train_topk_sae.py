import argparse
import datetime
import json
import logging
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sae_model import TopKSAE
SAE_ROOT = Path(__file__).resolve().parent
DEFAULT_ACT_DIR = SAE_ROOT / "activation" / "qwen3_sae_train"
DEFAULT_SAVE_ROOT = SAE_ROOT / "SAEs"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"train_topk_sae_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
def load_layer_activations(act_dir: Path, layer: int) -> torch.Tensor:
    layer_dir = act_dir / f"layer_{layer}"
    shards = sorted(layer_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {layer_dir}")
    acts = []
    for shard in shards:
        obj = torch.load(shard, map_location="cpu")
        acts.append(obj["activations"].float())
    return torch.cat(acts, dim=0).contiguous()
def train_one_layer(args: argparse.Namespace, layer: int, device: torch.device) -> dict:
    acts = load_layer_activations(args.activation_dir, layer)
    n, d_in = acts.shape
    d_sae = args.d_sae if args.d_sae > 0 else d_in * args.expansion_factor
    if args.center:
        mean = acts.mean(dim=0, keepdim=True)
        acts = acts - mean
    else:
        mean = torch.zeros(1, d_in)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(acts),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=args.k).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(
        f"[layer {layer}] train activations={n}, d_in={d_in}, d_sae={d_sae}, k={args.k}"
    )
    t0 = time.time()
    last_stats = {}
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        total_l0 = 0.0
        total_n = 0
        pbar = tqdm(loader, desc=f"layer {layer} epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for (batch,) in pbar:
            batch = batch.to(device)
            x_hat, z = sae(batch)
            mse = F.mse_loss(x_hat, batch)
            l1 = z.abs().mean()
            loss = mse + args.l1_coef * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            bs = batch.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_mse += float(mse.detach().cpu()) * bs
            total_l1 += float(l1.detach().cpu()) * bs
            total_l0 += float((z > 0).sum(dim=1).float().mean().detach().cpu()) * bs
            total_n += bs
            pbar.set_postfix(mse=total_mse / total_n, l0=total_l0 / total_n)
        last_stats = {
            "epoch": epoch,
            "loss": total_loss / total_n,
            "mse": total_mse / total_n,
            "l1": total_l1 / total_n,
            "l0": total_l0 / total_n,
        }
        logging.info(
            f"[layer {layer}] epoch={epoch} loss={last_stats['loss']:.6f} "
            f"mse={last_stats['mse']:.6f} l1={last_stats['l1']:.6f} l0={last_stats['l0']:.2f}"
        )
    save_dir = args.save_root / f"qwen3_topk_layer_{layer}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "sae.pt"
    config = {
        "layer": layer,
        "d_in": d_in,
        "d_sae": d_sae,
        "k": args.k,
        "expansion_factor": args.expansion_factor,
        "activation_dir": str(args.activation_dir),
        "center": args.center,
        "mean_shape": list(mean.shape),
        "train_stats": last_stats,
    }
    torch.save(
        {
            "config": config,
            "state_dict": sae.cpu().state_dict(),
            "activation_mean": mean.squeeze(0),
        },
        ckpt_path,
    )
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logging.info(f"[layer {layer}] saved: {ckpt_path} elapsed={time.time() - t0:.1f}s")
    return config
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopK SAEs on Qwen3 activations.")
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACT_DIR)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--layers", type=str, default="30,32,33,34,35")
    parser.add_argument("--d-sae", type=int, default=-1)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1-coef", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    layers = parse_layers(args.layers)
    logging.info("=" * 72)
    logging.info("train_topk_sae.py — TopK SAE training")
    logging.info(f"  activation_dir   : {args.activation_dir}")
    logging.info(f"  save_root        : {args.save_root}")
    logging.info(f"  layers           : {layers}")
    logging.info(f"  expansion_factor : {args.expansion_factor}")
    logging.info(f"  k                : {args.k}")
    logging.info(f"  epochs           : {args.epochs}")
    logging.info(f"  device           : {device}")
    logging.info("=" * 72)
    all_configs = []
    for layer in layers:
        all_configs.append(train_one_layer(args, layer, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.save_root.mkdir(parents=True, exist_ok=True)
    with open(args.save_root / "qwen3_topk_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    logging.info("All done.")
if __name__ == "__main__":
    main()
