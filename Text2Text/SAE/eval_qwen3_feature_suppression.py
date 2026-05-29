import argparse
import os
import datetime
import json
import logging
import sys
import time
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
SAE_ROOT = Path(__file__).resolve().parent
DAJA_ROOT = SAE_ROOT.parent
TEST_ROOT = DAJA_ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))
from eval_no_defense import (
    MAX_NEW_TOKENS,
    build_prompt,
    detect_model_type,
    get_dataset_records,
    load_completed_ids,
    parse_generic_output,
    parse_qwen_output,
)
from sae_model import TopKSAE
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_MODEL_PATH", "Models/Qwen3-8B"))
DEFAULT_SAE_ROOT = SAE_ROOT / "SAEs"
DEFAULT_FEATURES = DEFAULT_SAE_ROOT / "qwen3_harmful_features.json"
RESULT_ROOT = DAJA_ROOT / "result" / "Qwen3-8B_dualsteer_fs"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"eval_qwen3_feature_suppression_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
def load_sae_for_layer(sae_root: Path, layer: int, device: torch.device) -> TopKSAE:
    ckpt_path = sae_root / f"qwen3_topk_layer_{layer}" / "sae.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device=device, dtype=torch.float32).eval()
    return sae
def load_feature_map(feature_path: Path, layers: list[int], features_per_layer: int) -> dict[int, torch.Tensor]:
    with open(feature_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    fmap = {}
    for layer in layers:
        layer_obj = obj["layers"].get(str(layer))
        if layer_obj is None:
            raise KeyError(f"Layer {layer} not found in feature file: {feature_path}")
        feats = [int(x["feature"]) for x in layer_obj["features"][:features_per_layer]]
        if not feats:
            raise ValueError(f"No selected features for layer {layer}")
        fmap[layer] = torch.tensor(feats, dtype=torch.long)
    return fmap
class FeatureSuppressionHooks:
    def __init__(
        self,
        model,
        saes: dict[int, TopKSAE],
        feature_map: dict[int, torch.Tensor],
        alpha: float,
        suppression_strength: float,
    ):
        self.model = model
        self.saes = saes
        self.feature_map = feature_map
        self.alpha = alpha
        self.suppression_strength = suppression_strength
        self.handles = []
    def _make_hook(self, layer_idx: int):
        sae = self.saes[layer_idx]
        features = self.feature_map[layer_idx].to(next(sae.parameters()).device)
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            orig_dtype = hidden.dtype
            orig_device = hidden.device
            flat = hidden.reshape(-1, hidden.shape[-1]).float().to(next(sae.parameters()).device)
            with torch.no_grad():
                z = sae.encode(flat)
                z_supp = z.clone()
                z_supp[:, features] = z_supp[:, features] * (1.0 - self.suppression_strength)
                delta = sae.decode(z_supp) - sae.decode(z)
                delta = delta.to(orig_device).to(orig_dtype).reshape_as(hidden)
            steered = hidden + self.alpha * delta
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook
    def __enter__(self):
        for layer_idx in sorted(self.saes.keys()):
            self.handles.append(self.model.model.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx)))
        return self
    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
def generate_one(
    tokenizer,
    model,
    model_type: str,
    prompt: str,
    enable_thinking: bool,
    saes: dict[int, TopKSAE],
    feature_map: dict[int, torch.Tensor],
    alpha: float,
    suppression_strength: float,
) -> dict:
    text = build_prompt(tokenizer, model_type, prompt, enable_thinking)
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.6 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.8,
    }
    with torch.no_grad():
        with FeatureSuppressionHooks(model, saes, feature_map, alpha, suppression_strength):
            output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    generated_token_count = output_ids.shape[1] - prompt_length
    if model_type == "qwen3":
        thinking, answer = parse_qwen_output(tokenizer, output_ids, prompt_length)
    else:
        thinking, answer = parse_generic_output(tokenizer, output_ids, prompt_length)
    return {
        "thinking": thinking,
        "answer": answer,
        "generated_token_count": generated_token_count,
        "reached_token_limit": generated_token_count >= MAX_NEW_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_token_count": prompt_length,
    }
def evaluate_dataset(
    dataset_name: str,
    tokenizer,
    model,
    model_type: str,
    enable_thinking: bool,
    saes: dict[int, TopKSAE],
    feature_map: dict[int, torch.Tensor],
    alpha: float,
    suppression_strength: float,
    features_per_layer: int,
    wj_n_per_label: int,
) -> dict:
    result_path = RESULT_ROOT / f"{dataset_name}.jsonl"
    completed_ids = load_completed_ids(result_path)
    records = get_dataset_records(dataset_name, wj_n_per_label=wj_n_per_label)
    pending = [r for r in records if r["id"] not in completed_ids]
    resumed = len(records) - len(pending)
    logging.info(f"[Qwen3-8B_dualsteer_fs] [{dataset_name}] total={len(records)} resumed={resumed} pending={len(pending)}")
    stats = {"ok": 0, "truncated": 0, "error": 0}
    t0 = time.time()
    with logging_redirect_tqdm():
        pbar = tqdm(pending, desc=dataset_name, unit="sample", dynamic_ncols=True)
        for record in pbar:
            try:
                gen = generate_one(
                    tokenizer, model, model_type, record["prompt"], enable_thinking,
                    saes, feature_map, alpha, suppression_strength,
                )
                out = {
                    **record,
                    "model_name": "Qwen3-8B",
                    "method": "DualSteer-FeatureSuppression-PoC",
                    "sae_layers": sorted(saes.keys()),
                    "features_per_layer": features_per_layer,
                    "suppression_strength": suppression_strength,
                    "steering_alpha": alpha,
                    "model_type": model_type,
                    "thinking_enabled": enable_thinking,
                    **gen,
                }
                append_jsonl(result_path, out)
                if gen["reached_token_limit"]:
                    stats["truncated"] += 1
                else:
                    stats["ok"] += 1
            except Exception as exc:
                stats["error"] += 1
                logging.error(f"[{dataset_name}] {record.get('id')} ERROR: {exc}")
                append_jsonl(result_path, {
                    **record,
                    "model_name": "Qwen3-8B",
                    "method": "DualSteer-FeatureSuppression-PoC",
                    "error": str(exc),
                })
            pbar.set_postfix(**stats)
    elapsed = time.time() - t0
    logging.info(
        f"[DONE] {dataset_name} ok={stats['ok']} trunc={stats['truncated']} "
        f"err={stats['error']} elapsed={elapsed:.1f}s avg={elapsed/max(len(pending),1):.2f}s/sample"
    )
    logging.info(f"Saved to {result_path}")
    return {"dataset": dataset_name, **stats, "elapsed_sec": elapsed}
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 with SAE feature-level suppression.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--layers", type=str, default="30")
    parser.add_argument("--features-per-layer", type=int, default=64)
    parser.add_argument("--suppression-strength", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--datasets", nargs="+", default=["JBB-harmful"])
    parser.add_argument("--wj-n-per-label", type=int, default=200)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    setup_logging()
    layers = parse_layers(args.layers)
    logging.info("=" * 72)
    logging.info("eval_qwen3_feature_suppression.py — Qwen3 SAE feature suppression")
    logging.info(f"  model_path           : {args.model_path}")
    logging.info(f"  sae_root             : {args.sae_root}")
    logging.info(f"  feature_file         : {args.feature_file}")
    logging.info(f"  layers               : {layers}")
    logging.info(f"  features_per_layer   : {args.features_per_layer}")
    logging.info(f"  suppression_strength : {args.suppression_strength}")
    logging.info(f"  alpha                : {args.alpha}")
    logging.info(f"  datasets             : {args.datasets}")
    logging.info("=" * 72)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model_type = detect_model_type(str(args.model_path))
    enable_thinking = not args.disable_thinking
    sae_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saes = {layer: load_sae_for_layer(args.sae_root, layer, sae_device) for layer in layers}
    feature_map = load_feature_map(args.feature_file, layers, args.features_per_layer)
    logging.info({layer: int(feature_map[layer].numel()) for layer in layers})
    all_stats = []
    for dataset in args.datasets:
        all_stats.append(evaluate_dataset(
            dataset, tokenizer, model, model_type, enable_thinking, saes, feature_map,
            args.alpha, args.suppression_strength, args.features_per_layer, args.wj_n_per_label,
        ))
    logging.info("=" * 72)
    logging.info("FINAL SUMMARY")
    for s in all_stats:
        logging.info(f"  {s['dataset']:<16} ok={s['ok']} trunc={s['truncated']} err={s['error']}")
    logging.info("All done.")
if __name__ == "__main__":
    main()
