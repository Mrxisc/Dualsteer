import argparse
import datetime
import json
import logging
from pathlib import Path
import torch
from tqdm import tqdm
from sae_model import TopKSAE
SAE_ROOT = Path(__file__).resolve().parent
DEFAULT_ACT_DIR = SAE_ROOT / "activation" / "qwen3_sae_train"
DEFAULT_SAE_ROOT = SAE_ROOT / "SAEs"
DEFAULT_OUTPUT = DEFAULT_SAE_ROOT / "qwen3_harmful_features.json"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"select_harmful_features_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
def is_benign(label: str, dataset: str) -> bool:
    text = f"{label} {dataset}".lower()
    return "benign" in text
def is_harmful(label: str, dataset: str) -> bool:
    text = f"{label} {dataset}".lower()
    return "harmful" in text or dataset in {"HarmBench", "StrongREJECT"}
def load_sae(sae_root: Path, layer: int, device: torch.device) -> TopKSAE:
    ckpt_path = sae_root / f"qwen3_topk_layer_{layer}" / "sae.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device=device, dtype=torch.float32).eval()
    return sae
@torch.no_grad()
def score_layer(args: argparse.Namespace, layer: int, device: torch.device) -> dict:
    sae = load_sae(args.sae_root, layer, device)
    d_sae = sae.d_sae
    harm_sum = torch.zeros(d_sae, dtype=torch.float64)
    benign_sum = torch.zeros(d_sae, dtype=torch.float64)
    harm_count = 0
    benign_count = 0
    shard_paths = sorted((args.activation_dir / f"layer_{layer}").glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No activation shards for layer {layer}")
    for shard_path in tqdm(shard_paths, desc=f"layer {layer}", dynamic_ncols=True):
        shard = torch.load(shard_path, map_location="cpu")
        acts = shard["activations"].float()
        labels = shard.get("labels", [""] * acts.shape[0])
        datasets = shard.get("datasets", [""] * acts.shape[0])
        harm_indices = [i for i, (lab, ds) in enumerate(zip(labels, datasets)) if is_harmful(str(lab), str(ds))]
        benign_indices = [i for i, (lab, ds) in enumerate(zip(labels, datasets)) if is_benign(str(lab), str(ds))]
        for indices, target_sum, target_name in (
            (harm_indices, harm_sum, "harmful"),
            (benign_indices, benign_sum, "benign"),
        ):
            if not indices:
                continue
            x = acts[indices].to(device)
            z = sae.encode(x).detach().cpu().double()
            target_sum += z.sum(dim=0)
            if target_name == "harmful":
                harm_count += z.shape[0]
            else:
                benign_count += z.shape[0]
    if harm_count == 0 or benign_count == 0:
        raise RuntimeError(f"Layer {layer}: invalid counts harmful={harm_count}, benign={benign_count}")
    harm_mean = harm_sum / harm_count
    benign_mean = benign_sum / benign_count
    score = harm_mean - benign_mean
    positive = torch.nonzero(score > args.min_score, as_tuple=False).flatten()
    ranked = positive[torch.argsort(score[positive], descending=True)]
    selected = ranked[: args.top_k].tolist()
    features = []
    for idx in selected:
        idx = int(idx)
        features.append({
            "feature": idx,
            "score": round(float(score[idx]), 8),
            "harmful_mean": round(float(harm_mean[idx]), 8),
            "benign_mean": round(float(benign_mean[idx]), 8),
        })
    logging.info(
        f"[layer {layer}] harmful={harm_count}, benign={benign_count}, "
        f"selected={len(features)}, best_score={features[0]['score'] if features else None}"
    )
    return {
        "layer": layer,
        "harmful_count": harm_count,
        "benign_count": benign_count,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "features": features,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="Select harmful-selective SAE features.")
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACT_DIR)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=str, default="30,32,33,34,35")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()
    setup_logging()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda is not None else "cpu")
    layers = parse_layers(args.layers)
    logging.info("=" * 72)
    logging.info("select_harmful_features.py — harmful-selective SAE feature selection")
    logging.info(f"  activation_dir : {args.activation_dir}")
    logging.info(f"  sae_root       : {args.sae_root}")
    logging.info(f"  layers         : {layers}")
    logging.info(f"  top_k          : {args.top_k}")
    logging.info(f"  device         : {device}")
    logging.info("=" * 72)
    result = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "score_definition": "harmful_mean(feature_activation) - benign_mean(feature_activation)",
        "activation_dir": str(args.activation_dir),
        "sae_root": str(args.sae_root),
        "layers": {},
    }
    for layer in layers:
        layer_result = score_layer(args, layer, device)
        result["layers"][str(layer)] = layer_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved harmful feature selection to: {args.output}")
if __name__ == "__main__":
    main()
