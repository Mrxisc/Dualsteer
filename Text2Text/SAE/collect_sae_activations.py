import argparse
import datetime
import json
import logging
import random
import sys
import time
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
SAE_ROOT = Path(__file__).resolve().parent
DAJA_ROOT = SAE_ROOT.parent
TEST_ROOT = DAJA_ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))
from eval_no_defense import get_dataset_records
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_MODEL_PATH", "Models/Qwen3-8B"))
DEFAULT_OUTPUT_DIR = SAE_ROOT / "activation" / "qwen3_sae_train"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging(output_dir: Path) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"collect_sae_activations_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def parse_layers(layer_spec: str) -> list[int]:
    layers = []
    for part in layer_spec.split(","):
        part = part.strip()
        if part:
            layers.append(int(part))
    return sorted(set(layers))
def build_qwen_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
def load_records(args: argparse.Namespace) -> list[dict]:
    rng = random.Random(args.seed)
    all_records: list[dict] = []
    for dataset in args.datasets:
        records = get_dataset_records(dataset, wj_n_per_label=args.wj_n_per_label)
        if args.max_per_dataset > 0 and len(records) > args.max_per_dataset:
            records = rng.sample(records, args.max_per_dataset)
        all_records.extend(records)
        logging.info(f"Loaded {len(records)} records from {dataset}")
    rng.shuffle(all_records)
    if args.max_total > 0 and len(all_records) > args.max_total:
        all_records = all_records[:args.max_total]
    logging.info(f"Total records for activation collection: {len(all_records)}")
    return all_records
class LayerActivationCollector:
    def __init__(self, model, layers: list[int], token_position: str):
        self.layers = layers
        self.token_position = token_position
        self.current: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_idx in layers:
            handle = model.model.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)
    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.token_position == "last":
                act = hidden[:, -1, :]
            elif self.token_position == "mean":
                act = hidden.mean(dim=1)
            elif self.token_position == "all":
                act = hidden.reshape(-1, hidden.shape[-1])
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
def empty_buffers(layers: list[int]) -> dict[int, dict]:
    return {
        layer: {
            "activations": [],
            "record_ids": [],
            "datasets": [],
            "labels": [],
        }
        for layer in layers
    }
def flush_layer_shard(
    layer: int,
    buf: dict,
    output_dir: Path,
    shard_idx: int,
    token_position: str,
) -> int:
    if not buf["activations"]:
        return shard_idx
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    acts = torch.cat(buf["activations"], dim=0).contiguous()
    shard_path = layer_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(
        {
            "layer": layer,
            "activations": acts,
            "record_ids": buf["record_ids"],
            "datasets": buf["datasets"],
            "labels": buf["labels"],
            "token_position": token_position,
        },
        shard_path,
    )
    logging.info(f"Saved {shard_path}  shape={tuple(acts.shape)}")
    buf["activations"].clear()
    buf["record_ids"].clear()
    buf["datasets"].clear()
    buf["labels"].clear()
    return shard_idx + 1
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Qwen3 activations for SAE training.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["JBB-harmful", "HarmBench", "StrongREJECT", "JBB-benign"])
    parser.add_argument("--layers", type=str, default="30,32,33,34,35")
    parser.add_argument("--token-position", choices=["last", "mean", "all"], default="last")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-per-dataset", type=int, default=-1)
    parser.add_argument("--max-total", type=int, default=-1)
    parser.add_argument("--wj-n-per-label", type=int, default=200)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cuda is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    setup_logging(args.output_dir)
    layers = parse_layers(args.layers)
    logging.info("=" * 72)
    logging.info("collect_sae_activations.py — Qwen3 SAE activation collection")
    logging.info(f"  model_path      : {args.model_path}")
    logging.info(f"  output_dir      : {args.output_dir}")
    logging.info(f"  datasets        : {args.datasets}")
    logging.info(f"  layers          : {layers}")
    logging.info(f"  token_position  : {args.token_position}")
    logging.info(f"  shard_size      : {args.shard_size}")
    logging.info(f"  max_length      : {args.max_length}")
    logging.info(f"  enable_thinking : {args.enable_thinking}")
    logging.info("=" * 72)
    records = load_records(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    hidden_size = int(model.config.hidden_size)
    for layer in layers:
        if layer < 0 or layer >= int(model.config.num_hidden_layers):
            raise ValueError(f"Layer {layer} out of range for {model.config.num_hidden_layers} layers")
    collector = LayerActivationCollector(model, layers, args.token_position)
    buffers = empty_buffers(layers)
    shard_indices = {layer: 0 for layer in layers}
    total_by_layer = {layer: 0 for layer in layers}
    t0 = time.time()
    try:
        with torch.no_grad():
            for rec in tqdm(records, desc="collect", unit="prompt", dynamic_ncols=True):
                prompt = build_qwen_prompt(tokenizer, rec["prompt"], args.enable_thinking)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                collector.clear()
                _ = model(**inputs, use_cache=False)
                for layer in layers:
                    if layer not in collector.current:
                        raise RuntimeError(f"Missing activation for layer {layer}")
                    act = collector.current[layer]
                    n = act.shape[0]
                    buffers[layer]["activations"].append(act)
                    buffers[layer]["record_ids"].extend([rec.get("id", "")] * n)
                    buffers[layer]["datasets"].extend([rec.get("dataset", "")] * n)
                    buffers[layer]["labels"].extend([rec.get("label", "")] * n)
                    total_by_layer[layer] += n
                    buffered_n = sum(x.shape[0] for x in buffers[layer]["activations"])
                    if buffered_n >= args.shard_size:
                        shard_indices[layer] = flush_layer_shard(
                            layer,
                            buffers[layer],
                            args.output_dir,
                            shard_indices[layer],
                            args.token_position,
                        )
    finally:
        collector.close()
    for layer in layers:
        shard_indices[layer] = flush_layer_shard(
            layer,
            buffers[layer],
            args.output_dir,
            shard_indices[layer],
            args.token_position,
        )
    metadata = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "hook_target": "model.model.layers.{l} output[0] / resid_post",
        "layers": layers,
        "hidden_size": hidden_size,
        "token_position": args.token_position,
        "datasets": args.datasets,
        "num_records": len(records),
        "total_activations_by_layer": total_by_layer,
        "num_shards_by_layer": shard_indices,
        "max_length": args.max_length,
        "enable_thinking": args.enable_thinking,
        "shard_size": args.shard_size,
        "seed": args.seed,
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info("=" * 72)
    logging.info("DONE")
    logging.info(f"Saved metadata: {args.output_dir / 'metadata.json'}")
    for layer in layers:
        logging.info(
            f"  layer={layer}: activations={total_by_layer[layer]}  shards={shard_indices[layer]}"
        )
    logging.info(f"Elapsed: {time.time() - t0:.1f}s")
if __name__ == "__main__":
    main()
