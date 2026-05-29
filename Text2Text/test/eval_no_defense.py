import argparse
import os
import datetime
import json
import logging
import random
import time
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
DAJA_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = DAJA_ROOT / "dataset"
RESULT_ROOT = DAJA_ROOT / "result"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT = DAJA_ROOT / "log"
MODEL_CONFIGS: dict[str, dict] = {
    "Qwen3-8B": {
        "path": os.environ.get("QWEN3_MODEL_PATH", "Models/Qwen3-8B"),
        "enable_thinking": True,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "path": os.environ.get("DEEPSEEK_MODEL_PATH", "Models/DeepSeek-R1-Distill-Llama-8B"),
        "enable_thinking": True,
    },
}
DATASET_NAMES = ["JBB-harmful", "JBB-benign", "HarmBench", "StrongREJECT", "WildJailbreak"]
MAX_NEW_TOKENS = 1024
QWEN_THINK_END_TOKEN_ID = 151668
def setup_logging(run_tag: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_tag}.log"
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    logging.info(f"Log file: {log_path}")
def detect_model_type(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type", "unknown")
def load_model_and_tokenizer(model_path: str):
    logging.info(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logging.info("Loading model weights ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logging.info(f"Model loaded on {next(model.parameters()).device}")
    return tokenizer, model
def build_prompt(tokenizer, model_type: str, user_prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if model_type == "qwen3":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        text += "<think>\n"
    return text
def parse_qwen_output(tokenizer, output_ids, prompt_length: int):
    generated = output_ids[0][prompt_length:].tolist()
    try:
        index = len(generated) - generated[::-1].index(QWEN_THINK_END_TOKEN_ID)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(generated[:index], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(generated[index:], skip_special_tokens=True).strip("\n")
    return thinking, answer
def parse_generic_output(tokenizer, output_ids, prompt_length: int):
    generated = output_ids[0][prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    thinking, answer = "", text
    if "</think>" in text:
        thinking_part, answer_part = text.split("</think>", 1)
        thinking = thinking_part.replace("<think>", "").strip()
        answer = answer_part.strip()
    elif text.startswith("<think>"):
        thinking = text.replace("<think>", "").strip()
        answer = ""
    return thinking, answer
def generate_one(
    tokenizer,
    model,
    model_type: str,
    user_prompt: str,
    enable_thinking: bool,
) -> dict:
    text = build_prompt(tokenizer, model_type, user_prompt, enable_thinking)
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
def load_jbb(split: str) -> list[dict]:
    csv_map = {
        "harmful": DATASET_ROOT / "JBB-Behaviors" / "harmful-behaviors.csv",
        "benign": DATASET_ROOT / "JBB-Behaviors" / "benign-behaviors.csv",
    }
    df = pd.read_csv(csv_map[split])
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": f"jbb_{split}_{int(row['Index'])}",
            "dataset": f"JBB-{split}",
            "prompt": str(row["Goal"]),
            "label": split,
            "category": str(row.get("Category", "")),
            "behavior": str(row.get("Behavior", "")),
            "source": str(row.get("Source", "")),
        })
    return records
def load_harmbench() -> list[dict]:
    csv_path = (
        DATASET_ROOT / "HarmBench-main" / "data"
        / "behavior_datasets" / "harmbench_behaviors_text_test.csv"
    )
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        behavior = str(row["Behavior"])
        context = str(row.get("ContextString", "")).strip()
        func_cat = str(row.get("FunctionalCategory", "standard"))
        if context and context.lower() not in ("nan", ""):
            prompt = f"{context}\n\n{behavior}"
        else:
            prompt = behavior
        records.append({
            "id": f"harmbench_{str(row.get('BehaviorID', i))}",
            "dataset": "HarmBench",
            "prompt": prompt,
            "label": "harmful",
            "category": str(row.get("SemanticCategory", "")),
            "behavior": behavior,
            "functional_category": func_cat,
            "behavior_id": str(row.get("BehaviorID", "")),
            "source": "",
        })
    return records
def load_strongreject() -> list[dict]:
    csv_path = (
        DATASET_ROOT / "strongreject-main"
        / "strongreject_dataset" / "strongreject_dataset.csv"
    )
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        records.append({
            "id": f"strongreject_{i}",
            "dataset": "StrongREJECT",
            "prompt": str(row["forbidden_prompt"]),
            "label": "harmful",
            "category": str(row.get("category", "")),
            "behavior": str(row.get("forbidden_prompt", "")),
            "source": str(row.get("source", "")),
        })
    return records
def load_wildjailbreak(n_per_label: int = 200, seed: int = 42) -> list[dict]:
    parquet_path = DATASET_ROOT / "Wildjailbreak" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    labels = [
        "vanilla_harmful",
        "adversarial_harmful",
        "vanilla_benign",
        "adversarial_benign",
    ]
    records = []
    for label in labels:
        subset = df[df["prompt_harm_label"] == label]
        indices = list(subset.index)
        sample_indices = rng.sample(indices, min(n_per_label, len(indices)))
        for idx in sample_indices:
            row = subset.loc[idx]
            msgs = row["messages"]
            user_prompt = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = str(m.get("content", ""))
                    break
            if not user_prompt.strip():
                continue
            records.append({
                "id": f"wildjailbreak_{label}_{idx}",
                "dataset": "WildJailbreak",
                "prompt": user_prompt,
                "label": label,
                "category": label,
                "behavior": user_prompt[:120],
                "source": str(row.get("regenerated_model_type", "")),
            })
    return records
def get_dataset_records(dataset_name: str, wj_n_per_label: int = 200) -> list[dict]:
    if dataset_name == "JBB-harmful":
        return load_jbb("harmful")
    elif dataset_name == "JBB-benign":
        return load_jbb("benign")
    elif dataset_name == "HarmBench":
        return load_harmbench()
    elif dataset_name == "StrongREJECT":
        return load_strongreject()
    elif dataset_name == "WildJailbreak":
        return load_wildjailbreak(n_per_label=wj_n_per_label)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
def load_completed_ids(result_path: Path) -> set[str]:
    if not result_path.exists():
        return set()
    completed = set()
    with open(result_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add(rec["id"])
            except Exception:
                pass
    return completed
def append_result(result_path: Path, record: dict) -> None:
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
def evaluate_model_on_dataset(
    model_name: str,
    dataset_name: str,
    tokenizer,
    model,
    model_type: str,
    enable_thinking: bool,
    wj_n_per_label: int = 200,
) -> dict:
    result_path = RESULT_ROOT / f"{model_name}_no_defense" / f"{dataset_name}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids(result_path)
    logging.info(f"[{model_name}] [{dataset_name}] Loading dataset ...")
    records = get_dataset_records(dataset_name, wj_n_per_label=wj_n_per_label)
    total = len(records)
    pending = [r for r in records if r["id"] not in completed_ids]
    resumed = total - len(pending)
    logging.info(f"total={total}  resumed(skipped)={resumed}  pending={len(pending)}")
    stats = {"ok": 0, "truncated": 0, "error": 0}
    t_start = time.time()
    with logging_redirect_tqdm():
        pbar = tqdm(
            pending,
            desc=f"{dataset_name}",
            unit="sample",
            dynamic_ncols=True,
        )
        for i, record in enumerate(pbar, 1):
            prompt = record["prompt"]
            try:
                gen = generate_one(tokenizer, model, model_type, prompt, enable_thinking)
                output_record = {
                    **record,
                    "model_name": model_name,
                    "model_type": model_type,
                    "thinking_enabled": enable_thinking,
                    **gen,
                }
                append_result(result_path, output_record)
                if gen["reached_token_limit"]:
                    stats["truncated"] += 1
                    status = "TRUNCATED"
                else:
                    stats["ok"] += 1
                    status = "OK"
                pbar.set_postfix(
                    ok=stats["ok"], trunc=stats["truncated"], err=stats["error"]
                )
                logging.info(
                    f"[{i}/{len(pending)}] {record['id']}  "
                    f"gen_tokens={gen['generated_token_count']} [{status}]  "
                    f"preview={prompt[:50]!r}"
                )
            except Exception as exc:
                stats["error"] += 1
                pbar.set_postfix(
                    ok=stats["ok"], trunc=stats["truncated"], err=stats["error"]
                )
                logging.error(
                    f"[{i}/{len(pending)}] {record['id']}  ERROR: {exc}  preview={prompt[:50]!r}"
                )
                error_record = {
                    **record,
                    "model_name": model_name,
                    "model_type": model_type,
                    "thinking_enabled": enable_thinking,
                    "thinking": None,
                    "answer": None,
                    "generated_token_count": None,
                    "reached_token_limit": None,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "prompt_token_count": None,
                    "error": str(exc),
                }
                append_result(result_path, error_record)
    elapsed = time.time() - t_start
    n_pending = len(pending)
    avg_sec = elapsed / n_pending if n_pending else 0.0
    trunc_pct = stats["truncated"] / (stats["ok"] + stats["truncated"]) * 100 if (stats["ok"] + stats["truncated"]) else 0.0
    logging.info(
        f"[DATASET DONE] {model_name}/{dataset_name}  "
        f"ok={stats['ok']}  truncated={stats['truncated']} ({trunc_pct:.1f}%)  "
        f"error={stats['error']}  resumed={resumed}  "
        f"elapsed={elapsed:.1f}s  avg={avg_sec:.2f}s/sample"
    )
    logging.info(f"Saved to {result_path}")
    return {
        "model": model_name,
        "dataset": dataset_name,
        **stats,
        "resumed": resumed,
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": avg_sec,
        "truncated_pct": trunc_pct,
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-defense batch evaluation across jailbreak benchmarks."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Which models to evaluate (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="Which datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--wj-n-per-label",
        type=int,
        default=200,
        help="Samples per label from WildJailbreak (default: 200, total ~800).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking mode for all models.",
    )
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.models[0] if len(args.models) == 1 else "all"
    run_tag = f"eval_no_defense_{timestamp}_{model_slug}"
    setup_logging(run_tag)
    logging.info("=" * 72)
    logging.info("eval_no_defense.py — No-defense batch evaluation")
    logging.info(f"  models         : {args.models}")
    logging.info(f"  datasets       : {args.datasets}")
    logging.info(f"  result         : {RESULT_ROOT}")
    logging.info(f"  wj_n_per_label : {args.wj_n_per_label}")
    logging.info(f"  thinking       : {not args.disable_thinking}")
    logging.info("=" * 72)
    all_stats: list[dict] = []
    t_run_start = time.time()
    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        model_path = cfg["path"]
        enable_thinking = cfg["enable_thinking"] and not args.disable_thinking
        logging.info("=" * 72)
        logging.info(f"Loading model: {model_name}")
        logging.info(f"  path            : {model_path}")
        logging.info(f"  enable_thinking : {enable_thinking}")
        logging.info("=" * 72)
        t_model_start = time.time()
        model_type = detect_model_type(model_path)
        tokenizer, model = load_model_and_tokenizer(model_path)
        for dataset_name in args.datasets:
            ds_stats = evaluate_model_on_dataset(
                model_name=model_name,
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                model=model,
                model_type=model_type,
                enable_thinking=enable_thinking,
                wj_n_per_label=args.wj_n_per_label,
            )
            all_stats.append(ds_stats)
        model_elapsed = time.time() - t_model_start
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logging.info(f"Model {model_name} unloaded. Total model time: {model_elapsed:.1f}s")
    run_elapsed = time.time() - t_run_start
    logging.info("=" * 72)
    logging.info("FINAL SUMMARY")
    logging.info(f"  total_run_time : {run_elapsed:.1f}s ({run_elapsed/3600:.2f}h)")
    logging.info(f"  {'model':<36} {'dataset':<16} {'ok':>5} {'trunc':>6} {'trunc%':>7} {'err':>5} {'avg_s':>7}")
    logging.info(f"  {'-'*36} {'-'*16} {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*7}")
    for s in all_stats:
        logging.info(
            f"  {s['model']:<36} {s['dataset']:<16} "
            f"{s['ok']:>5} {s['truncated']:>6} {s['truncated_pct']:>6.1f}% "
            f"{s['error']:>5} {s['avg_sec_per_sample']:>7.2f}"
        )
    total_ok = sum(s["ok"] for s in all_stats)
    total_trunc = sum(s["truncated"] for s in all_stats)
    total_err = sum(s["error"] for s in all_stats)
    total_n = total_ok + total_trunc + total_err
    logging.info(f"  {'TOTAL':<36} {'':<16} {total_ok:>5} {total_trunc:>6} {'':>7} {total_err:>5}")
    logging.info(
        f"  truncated_rate={total_trunc/total_n*100:.1f}%  error_rate={total_err/total_n*100:.1f}%"
        if total_n else "  no samples processed"
    )
    logging.info("=" * 72)
    logging.info("All done.")
if __name__ == "__main__":
    main()
