import argparse
import datetime
import json
import logging
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
SAE_ROOT = Path(__file__).resolve().parent
DAJA_ROOT = SAE_ROOT.parent
TEST_ROOT = DAJA_ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))
from eval_no_defense import get_dataset_records
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_MODEL_PATH", "Models/Qwen3-8B"))
DEFAULT_OUTPUT_DIR = SAE_ROOT / "activation" / "qwen3_layer_scan"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging(output_dir: Path) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"collect_layer_scan_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def build_qwen_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
def load_prompts(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed)
    harmful_records: list[dict] = []
    for dataset in args.harmful_datasets:
        harmful_records.extend(get_dataset_records(dataset, wj_n_per_label=args.wj_n_per_label))
    benign_records = get_dataset_records(args.benign_dataset, wj_n_per_label=args.wj_n_per_label)
    if args.max_harmful > 0 and len(harmful_records) > args.max_harmful:
        harmful_records = rng.sample(harmful_records, args.max_harmful)
    if args.max_benign > 0 and len(benign_records) > args.max_benign:
        benign_records = rng.sample(benign_records, args.max_benign)
    logging.info(
        f"Loaded prompts: harmful={len(harmful_records)} from {args.harmful_datasets}, "
        f"benign={len(benign_records)} from {args.benign_dataset}"
    )
    return harmful_records, benign_records
class ResidPostCollector:
    def __init__(self, model, layers: list[int], token_position: str):
        self.layers = layers
        self.token_position = token_position
        self.handles = []
        self.current: dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            block = model.model.layers[layer_idx]
            handle = block.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)
    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.token_position == "last":
                act = hidden[:, -1, :]
            elif self.token_position == "mean":
                act = hidden.mean(dim=1)
            else:
                raise ValueError(f"Unknown token_position: {self.token_position}")
            self.current[layer_idx] = act.detach().float().cpu()
        return hook
    def clear(self) -> None:
        self.current = {}
    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
@torch.no_grad()
def accumulate_group(
    records: list[dict],
    group_name: str,
    tokenizer,
    model,
    collector: ResidPostCollector,
    layers: list[int],
    hidden_size: int,
    max_length: int,
    enable_thinking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(len(layers), hidden_size, dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.long)
    layer_to_pos = {layer: i for i, layer in enumerate(layers)}
    for rec in tqdm(records, desc=group_name, unit="prompt", dynamic_ncols=True):
        prompt = build_qwen_prompt(tokenizer, rec["prompt"], enable_thinking)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        collector.clear()
        _ = model(**inputs, use_cache=False)
        for layer_idx, act in collector.current.items():
            pos = layer_to_pos[layer_idx]
            sums[pos] += act.squeeze(0).to(torch.float64)
            counts[pos] += 1
    return sums, counts
def compute_scores(
    layers: list[int],
    harm_mean: torch.Tensor,
    benign_mean: torch.Tensor,
    harm_count: torch.Tensor,
    benign_count: torch.Tensor,
) -> list[dict]:
    diff = harm_mean - benign_mean
    l2 = torch.linalg.vector_norm(diff, ord=2, dim=1)
    cos_dist = 1.0 - F.cosine_similarity(harm_mean.float(), benign_mean.float(), dim=1)
    harm_norm = torch.linalg.vector_norm(harm_mean, ord=2, dim=1)
    benign_norm = torch.linalg.vector_norm(benign_mean, ord=2, dim=1)
    scores = []
    for i, layer in enumerate(layers):
        scores.append({
            "layer": int(layer),
            "l2_distance": round(float(l2[i]), 6),
            "cosine_distance": round(float(cos_dist[i]), 6),
            "harm_norm": round(float(harm_norm[i]), 6),
            "benign_norm": round(float(benign_norm[i]), 6),
            "harm_count": int(harm_count[i]),
            "benign_count": int(benign_count[i]),
        })
    return scores
def recommend_layers(scores: list[dict], top_k: int) -> dict:
    by_l2 = sorted(scores, key=lambda x: x["l2_distance"], reverse=True)
    by_cos = sorted(scores, key=lambda x: x["cosine_distance"], reverse=True)
    l2_values = [s["l2_distance"] for s in scores]
    if len(l2_values) >= 3:
        deltas = [l2_values[i] - l2_values[i - 1] for i in range(1, len(l2_values))]
        onset_idx = max(range(len(deltas)), key=lambda i: deltas[i]) + 1
        onset_layer = scores[onset_idx]["layer"]
    else:
        onset_layer = by_l2[0]["layer"] if by_l2 else None
    candidates = []
    if onset_layer is not None:
        candidates.append(onset_layer)
    for item in by_l2:
        if item["layer"] not in candidates:
            candidates.append(item["layer"])
        if len(candidates) >= top_k:
            break
    return {
        "onset_layer_by_l2_jump": onset_layer,
        "top_l2_layers": [s["layer"] for s in by_l2[:top_k]],
        "top_cosine_layers": [s["layer"] for s in by_cos[:top_k]],
        "recommended_sae_layers": candidates[:top_k],
    }
def parse_layers(layer_spec: str, num_layers: int) -> list[int]:
    if layer_spec == "all":
        return list(range(num_layers))
    layers = []
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer out of range: {layer}, num_layers={num_layers}")
    return sorted(set(layers))
def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 residual activation layer scan.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--harmful-datasets", nargs="+", default=["JBB-harmful", "HarmBench"])
    parser.add_argument("--benign-dataset", default="JBB-benign")
    parser.add_argument("--max-harmful", type=int, default=200)
    parser.add_argument("--max-benign", type=int, default=100)
    parser.add_argument("--wj-n-per-label", type=int, default=200)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--token-position", choices=["last", "mean"], default="last")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.cuda is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    setup_logging(args.output_dir)
    logging.info("=" * 72)
    logging.info("collect_layer_scan.py — Qwen3 residual activation scan")
    logging.info(f"  model_path      : {args.model_path}")
    logging.info(f"  output_dir      : {args.output_dir}")
    logging.info(f"  harmful_datasets: {args.harmful_datasets}")
    logging.info(f"  benign_dataset  : {args.benign_dataset}")
    logging.info(f"  token_position  : {args.token_position}")
    logging.info(f"  max_length      : {args.max_length}")
    logging.info(f"  enable_thinking : {args.enable_thinking}")
    logging.info("=" * 72)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    layers = parse_layers(args.layers, num_layers)
    logging.info(f"Scanning layers: {layers}")
    harmful_records, benign_records = load_prompts(args)
    collector = ResidPostCollector(model, layers, args.token_position)
    t0 = time.time()
    try:
        harm_sum, harm_count = accumulate_group(
            harmful_records, "harmful", tokenizer, model, collector,
            layers, hidden_size, args.max_length, args.enable_thinking,
        )
        benign_sum, benign_count = accumulate_group(
            benign_records, "benign", tokenizer, model, collector,
            layers, hidden_size, args.max_length, args.enable_thinking,
        )
    finally:
        collector.close()
    harm_mean = harm_sum / harm_count.clamp_min(1).unsqueeze(1)
    benign_mean = benign_sum / benign_count.clamp_min(1).unsqueeze(1)
    scores = compute_scores(layers, harm_mean, benign_mean, harm_count, benign_count)
    rec = recommend_layers(scores, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(harm_mean.float(), args.output_dir / "harm_mean.pt")
    torch.save(benign_mean.float(), args.output_dir / "benign_mean.pt")
    torch.save(harm_count, args.output_dir / "harm_count.pt")
    torch.save(benign_count, args.output_dir / "benign_count.pt")
    result = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "hook_target": "model.model.layers.{l} output[0] / resid_post",
        "token_position": args.token_position,
        "max_length": args.max_length,
        "enable_thinking": args.enable_thinking,
        "harmful_datasets": args.harmful_datasets,
        "benign_dataset": args.benign_dataset,
        "num_harmful": len(harmful_records),
        "num_benign": len(benign_records),
        "scores": scores,
        "recommendation": rec,
    }
    with open(args.output_dir / "layer_scan_scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info("Top layers by L2 distance:")
    for s in sorted(scores, key=lambda x: x["l2_distance"], reverse=True)[:args.top_k]:
        logging.info(
            f"  layer={s['layer']:02d}  l2={s['l2_distance']:.4f}  cos={s['cosine_distance']:.6f}"
        )
    logging.info(f"Recommendation: {rec}")
    logging.info(f"Saved to: {args.output_dir}")
    logging.info(f"Elapsed: {time.time() - t0:.1f}s")
if __name__ == "__main__":
    main()
