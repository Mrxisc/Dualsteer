from __future__ import annotations
import os
import argparse
import datetime
import json
import logging
import math
import random
import re
import time
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
DAJA_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = DAJA_ROOT / "dataset"
RESULT_ROOT = DAJA_ROOT / "result"
LOG_ROOT = DAJA_ROOT / "log"
DEFAULT_KG_NODES = DAJA_ROOT / "KG" / "outputs" / "nodes.csv"
DEFAULT_MODEL_PATH = Path(os.environ.get("QWEN3_MODEL_PATH", "Models/Qwen3-8B"))
DEFAULT_MODEL_NAME = "Qwen3-8B"
DATASET_NAMES = ["JBB-harmful", "JBB-benign", "HarmBench", "StrongREJECT", "WildJailbreak"]
MAX_NEW_TOKENS = 1024
QWEN_THINK_END_TOKEN_ID = 151668
KG_RUNTIME_EXCLUDE = {
    "build", "make", "create", "write", "generate", "provide", "explain", "describe",
    "medical", "records", "personal", "public", "online", "access", "process",
    "claiming", "article", "social media", "campaign", "promotes", "promoting", "body",
    "individuals", "able", "colleague", "letter", "post explaining", "better", "years",
    "place", "world", "blog", "old", "entire", "use", "speech", "group", "society",
    "email", "text", "offer", "necessary", "regarding", "possible", "details", "obtain",
    "high", "human", "persuasive", "convincing", "children", "blog post", "behavior",
    "plan", "ring", "knowledge", "company", "cover", "money", "individual", "life",
    "list", "tips", "number", "large", "computer", "user", "gain", "machine",
}
def setup_logging(run_tag: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{run_tag}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())
def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
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
            "prompt": clean_text(row["Goal"]),
            "label": split,
            "category": clean_text(row.get("Category", "")),
            "behavior": clean_text(row.get("Behavior", "")),
            "source": clean_text(row.get("Source", "")),
        })
    return records
def load_harmbench() -> list[dict]:
    csv_path = DATASET_ROOT / "HarmBench-main" / "data" / "behavior_datasets" / "harmbench_behaviors_text_test.csv"
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        behavior = clean_text(row["Behavior"])
        context = clean_text(row.get("ContextString", ""))
        prompt = f"{context}\n\n{behavior}" if context and context.lower() != "nan" else behavior
        records.append({
            "id": f"harmbench_{clean_text(row.get('BehaviorID', i))}",
            "dataset": "HarmBench",
            "prompt": prompt,
            "label": "harmful",
            "category": clean_text(row.get("SemanticCategory", "")),
            "behavior": behavior,
            "functional_category": clean_text(row.get("FunctionalCategory", "standard")),
            "behavior_id": clean_text(row.get("BehaviorID", "")),
            "source": "",
        })
    return records
def load_strongreject() -> list[dict]:
    csv_path = DATASET_ROOT / "strongreject-main" / "strongreject_dataset" / "strongreject_dataset.csv"
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        prompt = clean_text(row["forbidden_prompt"])
        records.append({
            "id": f"strongreject_{i}",
            "dataset": "StrongREJECT",
            "prompt": prompt,
            "label": "harmful",
            "category": clean_text(row.get("category", "")),
            "behavior": prompt,
            "source": clean_text(row.get("source", "")),
        })
    return records
def load_wildjailbreak(n_per_label: int = 200, seed: int = 42) -> list[dict]:
    parquet_path = DATASET_ROOT / "Wildjailbreak" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    labels = ["vanilla_harmful", "adversarial_harmful", "vanilla_benign", "adversarial_benign"]
    records = []
    for label in labels:
        subset = df[df["prompt_harm_label"] == label]
        indices = list(subset.index)
        for idx in rng.sample(indices, min(n_per_label, len(indices))):
            row = subset.loc[idx]
            user_prompt = ""
            for message in row["messages"]:
                if isinstance(message, dict) and message.get("role") == "user":
                    user_prompt = clean_text(message.get("content", ""))
                    break
            if not user_prompt:
                continue
            records.append({
                "id": f"wildjailbreak_{label}_{idx}",
                "dataset": "WildJailbreak",
                "prompt": user_prompt,
                "label": label,
                "category": label,
                "behavior": user_prompt[:120],
                "source": clean_text(row.get("regenerated_model_type", "")),
            })
    return records
def get_dataset_records(dataset_name: str, wj_n_per_label: int = 200) -> list[dict]:
    if dataset_name == "JBB-harmful":
        return load_jbb("harmful")
    if dataset_name == "JBB-benign":
        return load_jbb("benign")
    if dataset_name == "HarmBench":
        return load_harmbench()
    if dataset_name == "StrongREJECT":
        return load_strongreject()
    if dataset_name == "WildJailbreak":
        return load_wildjailbreak(n_per_label=wj_n_per_label)
    raise ValueError(f"Unknown dataset: {dataset_name}")
def load_completed_ids(result_path: Path) -> set[str]:
    if not result_path.exists():
        return set()
    completed = set()
    with result_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                completed.add(json.loads(line)["id"])
            except Exception:
                pass
    return completed
def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
def load_kg_concepts(nodes_path: Path, min_risk: float, min_frequency: int, high_risk_without_alias: float) -> list[dict]:
    df = pd.read_csv(nodes_path)
    df = df[df["id"].astype(str).str.startswith("concept:")].copy()
    df = df[df["risk_score"].astype(float) >= min_risk]
    df = df[df["frequency"].astype(int) >= min_frequency]
    concepts = []
    for _, row in df.iterrows():
        label = normalize_text(row["label"])
        if len(label) < 3:
            continue
        if label in KG_RUNTIME_EXCLUDE:
            continue
        aliases = "" if pd.isna(row.get("aliases", "")) else str(row.get("aliases", ""))
        risk_score = float(row["risk_score"])
        if not aliases and risk_score < high_risk_without_alias:
            continue
        concepts.append({
            "label": str(row["label"]),
            "label_norm": label,
            "risk_score": risk_score,
            "frequency": int(row["frequency"]),
            "aliases": aliases,
            "categories": "" if pd.isna(row.get("categories", "")) else str(row.get("categories", "")),
        })
    concepts.sort(key=lambda x: (len(x["label_norm"].split()), x["risk_score"], x["frequency"]), reverse=True)
    logging.info(f"Loaded {len(concepts)} KG concepts from {nodes_path}")
    return concepts
def score_prompt_with_kg(prompt: str, concepts: list[dict], threshold: float, max_matches: int) -> dict:
    prompt_norm = normalize_text(prompt)
    raw_matches = []
    for concept in concepts:
        label = concept["label_norm"]
        if not label:
            continue
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", prompt_norm):
            raw_matches.append({
                "concept": concept["label"],
                "label_norm": label,
                "span": match.span(),
                "risk_score": round(concept["risk_score"], 4),
                "frequency": concept["frequency"],
                "aliases": concept["aliases"],
                "categories": concept["categories"],
            })
            break
        if len(raw_matches) >= max_matches * 3:
            break
    raw_matches.sort(key=lambda x: (x["span"][1] - x["span"][0], x["risk_score"], x["frequency"]), reverse=True)
    matches = []
    used_spans = []
    for item in raw_matches:
        start, end = item["span"]
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_spans):
            continue
        used_spans.append((start, end))
        item = {k: v for k, v in item.items() if k not in {"label_norm", "span"}}
        matches.append(item)
        if len(matches) >= max_matches:
            break
    if not matches:
        return {"kg_risk_score": 0.0, "kg_decision": "pass", "kg_matched_concepts": []}
    scores = [min(max(m["risk_score"], 0.0), 1.0) for m in matches]
    risk = 1.0 - math.prod([1.0 - score for score in scores])
    decision = "filter" if risk >= threshold else "review"
    return {"kg_risk_score": round(risk, 4), "kg_decision": decision, "kg_matched_concepts": matches}
def detect_model_type(model_path: Path) -> str:
    config_path = model_path / "config.json"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type", "unknown")
def load_model(model_path: Path):
    logging.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    logging.info(f"Model loaded on {next(model.parameters()).device}")
    return tokenizer, model
def build_prompt(tokenizer, model_type: str, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    if model_type == "qwen3":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        text += "<think>\n"
    return text
def parse_qwen_output(tokenizer, output_ids, prompt_length: int) -> tuple[str, str]:
    generated = output_ids[0][prompt_length:].tolist()
    try:
        index = len(generated) - generated[::-1].index(QWEN_THINK_END_TOKEN_ID)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(generated[:index], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(generated[index:], skip_special_tokens=True).strip("\n")
    return thinking, answer
def parse_generic_output(tokenizer, output_ids, prompt_length: int) -> tuple[str, str]:
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
def generate_one(tokenizer, model, model_type: str, prompt: str, enable_thinking: bool) -> dict:
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
def blocked_response(score: dict) -> str:
    concepts = ", ".join([m["concept"] for m in score.get("kg_matched_concepts", [])[:5]])
    return f"I cannot assist with this request because it matches high-risk safety concepts in the knowledge graph: {concepts}."
def evaluate_dataset(dataset_name: str, tokenizer, model, concepts: list[dict], args: argparse.Namespace) -> dict:
    result_dir = RESULT_ROOT / args.result_subdir
    result_path = result_dir / f"{dataset_name}.jsonl"
    completed_ids = load_completed_ids(result_path)
    records = get_dataset_records(dataset_name, wj_n_per_label=args.wj_n_per_label)
    pending = [record for record in records if record["id"] not in completed_ids]
    resumed = len(records) - len(pending)
    logging.info(f"[{dataset_name}] total={len(records)} resumed={resumed} pending={len(pending)}")
    stats = {"generated": 0, "kg_filtered": 0, "kg_review_generated": 0, "error": 0, "truncated": 0}
    t0 = time.time()
    with logging_redirect_tqdm():
        pbar = tqdm(pending, desc=dataset_name, unit="sample", dynamic_ncols=True)
        for record in pbar:
            prompt = record["prompt"]
            score = score_prompt_with_kg(prompt, concepts, args.kg_threshold, args.max_matches)
            output_record = {
                **record,
                "model_name": args.model_name,
                "defense": "KG-Filter",
                "kg_nodes_path": str(args.kg_nodes),
                "kg_threshold": args.kg_threshold,
                "kg_min_risk": args.kg_min_risk,
                "kg_min_frequency": args.kg_min_frequency,
                **score,
            }
            try:
                if score["kg_decision"] == "filter" and not args.generate_filtered:
                    stats["kg_filtered"] += 1
                    append_jsonl(result_path, {
                        **output_record,
                        "thinking_enabled": args.enable_thinking,
                        "thinking": "",
                        "answer": blocked_response(score),
                        "generated_token_count": 0,
                        "reached_token_limit": False,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "prompt_token_count": 0,
                        "kg_blocked": True,
                    })
                else:
                    gen = generate_one(tokenizer, model, args.model_type, prompt, args.enable_thinking)
                    stats["generated"] += 1
                    if score["kg_decision"] == "review":
                        stats["kg_review_generated"] += 1
                    if gen["reached_token_limit"]:
                        stats["truncated"] += 1
                    append_jsonl(result_path, {**output_record, "thinking_enabled": args.enable_thinking, "kg_blocked": False, **gen})
            except Exception as exc:
                stats["error"] += 1
                logging.error(f"[{record['id']}] ERROR: {exc}")
                append_jsonl(result_path, {**output_record, "answer": None, "error": str(exc), "kg_blocked": False})
            pbar.set_postfix(**stats)
    elapsed = time.time() - t0
    logging.info(f"[{dataset_name}] saved={result_path} stats={stats} elapsed={elapsed:.1f}s")
    return {"dataset": dataset_name, "resumed": resumed, "elapsed_sec": elapsed, **stats}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-8B generation with KG input filtering.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--kg-nodes", type=Path, default=DEFAULT_KG_NODES)
    parser.add_argument("--datasets", nargs="+", default=DATASET_NAMES, choices=DATASET_NAMES)
    parser.add_argument("--result-subdir", default="Qwen3-8B_kg_filter")
    parser.add_argument("--wj-n-per-label", type=int, default=200)
    parser.add_argument("--kg-threshold", type=float, default=0.6)
    parser.add_argument("--kg-min-risk", type=float, default=0.5)
    parser.add_argument("--kg-min-frequency", type=int, default=1)
    parser.add_argument("--high-risk-without-alias", type=float, default=0.88)
    parser.add_argument("--max-matches", type=int, default=20)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--generate-filtered", action="store_true", help="Generate with Qwen even when KG decision is filter.")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    args.enable_thinking = not args.disable_thinking
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(f"eval_qwen3_kg_filter_{timestamp}")
    logging.info(f"Args: {vars(args)}")
    concepts = load_kg_concepts(args.kg_nodes, args.kg_min_risk, args.kg_min_frequency, args.high_risk_without_alias)
    args.model_type = detect_model_type(args.model_path)
    logging.info(f"Detected model_type: {args.model_type}")
    tokenizer, model = load_model(args.model_path)
    all_stats = []
    for dataset_name in args.datasets:
        all_stats.append(evaluate_dataset(dataset_name, tokenizer, model, concepts, args))
    logging.info("FINAL SUMMARY")
    for stats in all_stats:
        logging.info(stats)
    logging.info("All done.")
if __name__ == "__main__":
    main()
