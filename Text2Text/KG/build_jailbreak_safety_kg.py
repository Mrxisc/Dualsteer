from __future__ import annotations
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
DAJA_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = DAJA_ROOT / "dataset"
DEFAULT_DATASETS = ["JBB-harmful", "JBB-benign", "HarmBench", "StrongREJECT", "WildJailbreak"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BASE_RISK_KEYWORDS = {
    "violence": ["kill", "murder", "weapon", "gun", "shoot", "bomb", "attack", "torture", "violence", "genocide", "war"],
    "cyber": ["hack", "hacking", "malware", "ransomware", "virus", "password", "keylogger", "phishing", "ddos", "exploit", "admin access", "unauthorized access"],
    "fraud": ["scam", "fraud", "counterfeit", "blackmail", "bribe", "forgery", "identity theft", "fake account", "fake document"],
    "hate": ["hate", "racist", "xenophobic", "sexist", "harass", "discrimination", "holocaust", "ethnic", "stereotype"],
    "sexual": ["sexual", "sex", "porn", "erotic", "explicit", "pedophilia", "trafficking", "revenge porn"],
    "self_harm": ["suicide", "self-harm", "anorexia", "bulimia", "depressed", "overdose", "mental health"],
    "privacy": ["private information", "confidential", "social security", "phone number", "personal data", "location", "medical records", "disclose"],
    "illegal": ["illegal", "illegally", "evade", "steal", "extort", "tax evasion", "insider trading", "launder", "caught", "detection"],
    "disinformation": ["fake news", "fake", "disinformation", "conspiracy", "hoax", "misinformation", "false claim", "election", "ballot"],
    "chemical_drug": ["acid", "synthesis", "chemical", "gas", "sodium", "mixture", "reaction", "cocaine", "heroin", "codeine", "morphine", "drug", "medication"],
}
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "write", "create", "generate", "provide", "explain", "describe", "give", "make", "draft",
    "develop", "design", "instructions", "step", "steps", "guide", "tutorial", "detailed",
    "sure", "help", "helpful", "want", "need", "using", "based", "specific", "people", "person",
    "rules", "effectively", "transactions", "approach", "best", "way", "feel", "driver", "habits",
    "prevents", "holding", "favorite", "emotional", "song", "game", "novel", "long", "passage",
    "lyric", "char", "guest", "table", "ent", "process", "technique", "getting", "making",
}
@dataclass
class Node:
    id: str
    label: str
    type: str
    risk_score: float
    frequency: int
    datasets: str
    categories: str
    aliases: str = ""
@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float
    evidence_count: int
    evidence: str
def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())
def normalize_phrase(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def node_id(prefix: str, label: str) -> str:
    safe = normalize_phrase(label).replace(" ", "_")[:90]
    safe = safe or "unknown"
    return f"{prefix}:{safe}"
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
        prompt = f"{context} {behavior}".strip() if context and context.lower() != "nan" else behavior
        records.append({
            "id": f"harmbench_{clean_text(row.get('BehaviorID', i))}",
            "dataset": "HarmBench",
            "prompt": prompt,
            "label": "harmful",
            "category": clean_text(row.get("SemanticCategory", "")),
            "behavior": behavior,
            "functional_category": clean_text(row.get("FunctionalCategory", "standard")),
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
def load_wildjailbreak(n_per_label: int, seed: int) -> list[dict]:
    parquet_path = DATASET_ROOT / "Wildjailbreak" / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    labels = ["vanilla_harmful", "adversarial_harmful", "vanilla_benign", "adversarial_benign"]
    records = []
    for label in labels:
        subset = df[df["prompt_harm_label"] == label]
        indices = list(subset.index)
        sample_indices = rng.sample(indices, min(n_per_label, len(indices)))
        for idx in sample_indices:
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
def load_dataset(name: str, wildjailbreak_n_per_label: int, seed: int) -> list[dict]:
    if name == "JBB-harmful":
        return load_jbb("harmful")
    if name == "JBB-benign":
        return load_jbb("benign")
    if name == "HarmBench":
        return load_harmbench()
    if name == "StrongREJECT":
        return load_strongreject()
    if name == "WildJailbreak":
        return load_wildjailbreak(wildjailbreak_n_per_label, seed)
    raise ValueError(f"Unknown dataset: {name}")
def load_records(dataset_names: list[str], wildjailbreak_n_per_label: int, seed: int) -> list[dict]:
    records = []
    for name in dataset_names:
        loaded = load_dataset(name, wildjailbreak_n_per_label, seed)
        records.extend(loaded)
        print(f"Loaded {len(loaded):5d} records from {name}")
    return [r for r in records if r.get("prompt", "").strip()]
def phrase_candidates_from_category(category: str) -> list[str]:
    category = normalize_phrase(category.replace("/", " ").replace("_", " "))
    if not category or category == "nan":
        return []
    tokens = [t for t in category.split() if t not in CUSTOM_STOP_WORDS and len(t) > 2]
    phrases = [category]
    phrases.extend(tokens)
    return list(dict.fromkeys(phrases))
def extract_tfidf_phrases(records: list[dict], top_k: int, min_df: int) -> Counter:
    prompts = [r["prompt"] for r in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=list(CUSTOM_STOP_WORDS),
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        min_df=min_df,
        max_df=0.85,
        max_features=max(top_k * 8, 2000),
    )
    matrix = vectorizer.fit_transform(prompts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(scores)[::-1]
    phrase_scores = Counter()
    for idx in order[:top_k]:
        phrase = normalize_phrase(feature_names[idx])
        if len(phrase) >= 3 and phrase not in CUSTOM_STOP_WORDS:
            phrase_scores[phrase] = float(scores[idx])
    return phrase_scores
def record_risk_score(record: dict) -> float:
    label = normalize_phrase(record.get("label", ""))
    dataset = record.get("dataset", "")
    if "benign" in label or dataset == "JBB-benign":
        return 0.15
    if "harmful" in label or dataset in {"HarmBench", "StrongREJECT", "JBB-harmful"}:
        return 0.95
    return 0.65
def keyword_risk_prior(phrase: str) -> tuple[float, list[str]]:
    phrase_norm = normalize_phrase(phrase)
    matched = []
    for risk_type, keywords in BASE_RISK_KEYWORDS.items():
        for keyword in keywords:
            if normalize_phrase(keyword) in phrase_norm or phrase_norm in normalize_phrase(keyword):
                matched.append(risk_type)
                break
    if not matched:
        return 0.5, []
    return min(1.0, 0.65 + 0.08 * len(matched)), matched
def build_nodes(records: list[dict], top_k_phrases: int, min_df: int) -> tuple[dict[str, Node], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    phrase_counts: Counter = Counter()
    phrase_datasets: dict[str, set[str]] = defaultdict(set)
    phrase_categories: dict[str, set[str]] = defaultdict(set)
    phrase_records: dict[str, set[str]] = defaultdict(set)
    phrase_risk_sum: Counter = Counter()
    tfidf_phrases = extract_tfidf_phrases(records, top_k_phrases, min_df)
    allowed_phrases = set(tfidf_phrases.keys())
    risk_keyword_phrases = {
        normalize_phrase(keyword)
        for keywords in BASE_RISK_KEYWORDS.values()
        for keyword in keywords
        if normalize_phrase(keyword)
    }
    for record in records:
        candidates = set()
        candidates.update(phrase_candidates_from_category(record.get("category", "")))
        prompt_norm = normalize_phrase(record.get("prompt", ""))
        for phrase in allowed_phrases:
            if phrase in prompt_norm:
                candidates.add(phrase)
        for phrase in risk_keyword_phrases:
            if phrase in prompt_norm:
                candidates.add(phrase)
        for phrase in candidates:
            if not phrase or len(phrase) < 3:
                continue
            phrase_counts[phrase] += 1
            phrase_datasets[phrase].add(record["dataset"])
            phrase_categories[phrase].add(clean_text(record.get("category", "")))
            phrase_records[phrase].add(record["id"])
            phrase_risk_sum[phrase] += record_risk_score(record)
    nodes: dict[str, Node] = {}
    for phrase, freq in phrase_counts.items():
        prior, aliases = keyword_risk_prior(phrase)
        empirical = phrase_risk_sum[phrase] / max(freq, 1)
        risk = round(0.7 * empirical + 0.3 * prior, 4)
        nid = node_id("concept", phrase)
        nodes[nid] = Node(
            id=nid,
            label=phrase,
            type="harmful_concept" if risk >= 0.5 else "benign_or_contextual_concept",
            risk_score=risk,
            frequency=int(freq),
            datasets="|".join(sorted(phrase_datasets[phrase])),
            categories="|".join(sorted(c for c in phrase_categories[phrase] if c and c.lower() != "nan")),
            aliases="|".join(sorted(set(aliases))),
        )
    for dataset in sorted({r["dataset"] for r in records}):
        nid = node_id("dataset", dataset)
        nodes[nid] = Node(nid, dataset, "dataset", 0.0, sum(1 for r in records if r["dataset"] == dataset), dataset, "")
    for category in sorted({clean_text(r.get("category", "")) for r in records if clean_text(r.get("category", ""))}):
        nid = node_id("category", category)
        cat_records = [r for r in records if clean_text(r.get("category", "")) == category]
        risk = sum(record_risk_score(r) for r in cat_records) / max(len(cat_records), 1)
        nodes[nid] = Node(nid, category, "risk_category", round(risk, 4), len(cat_records), "|".join(sorted({r["dataset"] for r in cat_records})), category)
    return nodes, phrase_datasets, phrase_categories, phrase_records
def add_edge(edge_map: dict[tuple[str, str, str], Edge], source: str, target: str, relation: str, weight: float, evidence: str) -> None:
    if source == target:
        return
    if source > target and relation in {"semantic_similarity", "lexical_overlap", "co_occurrence"}:
        source, target = target, source
    key = (source, target, relation)
    if key not in edge_map:
        edge_map[key] = Edge(source, target, relation, round(float(weight), 4), 1, evidence[:240])
    else:
        old = edge_map[key]
        old.weight = round(max(old.weight, float(weight)), 4)
        old.evidence_count += 1
        if evidence[:120] not in old.evidence:
            old.evidence = (old.evidence + " | " + evidence[:120])[:240]
def lexical_overlap(a: str, b: str) -> float:
    sa = set(normalize_phrase(a).split())
    sb = set(normalize_phrase(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
def build_edges(nodes: dict[str, Node], records: list[dict], phrase_records: dict[str, set[str]], semantic_threshold: float, lexical_threshold: float, top_semantic_edges: int) -> list[Edge]:
    edge_map: dict[tuple[str, str, str], Edge] = {}
    concept_nodes = [n for n in nodes.values() if n.id.startswith("concept:")]
    phrase_to_id = {n.label: n.id for n in concept_nodes}
    for node in concept_nodes:
        for dataset in node.datasets.split("|") if node.datasets else []:
            add_edge(edge_map, node.id, node_id("dataset", dataset), "observed_in_dataset", 1.0, dataset)
        for category in node.categories.split("|") if node.categories else []:
            if category:
                add_edge(edge_map, node.id, node_id("category", category), "belongs_to_category", max(node.risk_score, 0.1), category)
    record_to_phrases: dict[str, list[str]] = defaultdict(list)
    for phrase, record_ids in phrase_records.items():
        for rid in record_ids:
            record_to_phrases[rid].append(phrase)
    for record in records:
        phrases = record_to_phrases.get(record["id"], [])
        risk = record_risk_score(record)
        for i, src_phrase in enumerate(phrases):
            for dst_phrase in phrases[i + 1:]:
                add_edge(edge_map, phrase_to_id[src_phrase], phrase_to_id[dst_phrase], "co_occurrence", risk, record["id"])
    for i, src in enumerate(concept_nodes):
        for dst in concept_nodes[i + 1:]:
            overlap = lexical_overlap(src.label, dst.label)
            if overlap >= lexical_threshold:
                add_edge(edge_map, src.id, dst.id, "lexical_overlap", overlap, f"{src.label} <-> {dst.label}")
    if len(concept_nodes) >= 2:
        labels = [n.label for n in concept_nodes]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        features = normalize(vectorizer.fit_transform(labels), norm="l2", copy=False)
        sim = cosine_similarity(features)
        candidates = []
        for i in range(len(concept_nodes)):
            for j in range(i + 1, len(concept_nodes)):
                if sim[i, j] >= semantic_threshold:
                    candidates.append((sim[i, j], concept_nodes[i], concept_nodes[j]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for weight, src, dst in candidates[:top_semantic_edges]:
            add_edge(edge_map, src.id, dst.id, "semantic_similarity", float(weight), f"{src.label} <-> {dst.label}")
    return list(edge_map.values())
def export_graph(nodes: dict[str, Node], edges: list[Edge], records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_df = pd.DataFrame([asdict(n) for n in nodes.values()]).sort_values(["type", "risk_score", "frequency"], ascending=[True, False, False])
    edges_df = pd.DataFrame([asdict(e) for e in edges]).sort_values(["relation", "weight", "evidence_count"], ascending=[True, False, False])
    records_df = pd.DataFrame(records)
    nodes_df.to_csv(output_dir / "nodes.csv", index=False)
    edges_df.to_csv(output_dir / "edges.csv", index=False)
    records_df.to_csv(output_dir / "source_prompts.csv", index=False)
    graph = {
        "metadata": {
            "num_nodes": len(nodes_df),
            "num_edges": len(edges_df),
            "num_prompts": len(records_df),
            "datasets": sorted(records_df["dataset"].unique().tolist()),
        },
        "nodes": [asdict(n) for n in nodes.values()],
        "edges": [asdict(e) for e in edges],
    }
    (output_dir / "jailbreak_safety_filter_kg.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import networkx as nx
        g = nx.MultiDiGraph()
        for node in nodes.values():
            data = asdict(node)
            nid = data.pop("id")
            g.add_node(nid, **data)
        for edge in edges:
            g.add_edge(edge.source, edge.target, relation=edge.relation, weight=edge.weight, evidence_count=edge.evidence_count, evidence=edge.evidence)
        nx.write_graphml(g, output_dir / "jailbreak_safety_filter_kg.graphml")
    except Exception as exc:
        print(f"GraphML export skipped: {exc}")
    concept_nodes = nodes_df[nodes_df["id"].astype(str).str.startswith("concept:")].copy()
    concept_nodes = concept_nodes.sort_values(["risk_score", "frequency"], ascending=[False, False])
    summary = {
        "nodes_by_type": nodes_df["type"].value_counts().to_dict(),
        "edges_by_relation": edges_df["relation"].value_counts().to_dict() if len(edges_df) else {},
        "top_risk_concepts": concept_nodes.head(30).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
def score_prompt(prompt: str, nodes: dict[str, Node]) -> dict:
    prompt_norm = normalize_phrase(prompt)
    matches = []
    for node in nodes.values():
        if not node.id.startswith("concept:"):
            continue
        label = normalize_phrase(node.label)
        if label and label in prompt_norm:
            matches.append({"concept": node.label, "risk_score": node.risk_score, "frequency": node.frequency})
    if not matches:
        return {"risk_score": 0.0, "matched_concepts": [], "decision": "pass"}
    scores = [m["risk_score"] for m in matches]
    risk = 1.0 - math.prod([1.0 - min(max(s, 0.0), 1.0) for s in scores])
    return {"risk_score": round(risk, 4), "matched_concepts": matches, "decision": "filter" if risk >= 0.6 else "review"}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--wildjailbreak-n-per-label", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-phrases", type=int, default=600)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--semantic-threshold", type=float, default=0.78)
    parser.add_argument("--lexical-threshold", type=float, default=0.34)
    parser.add_argument("--top-semantic-edges", type=int, default=1500)
    parser.add_argument("--demo-prompt", default="")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    records = load_records(args.datasets, args.wildjailbreak_n_per_label, args.seed)
    print(f"Total prompts: {len(records)}")
    nodes, _, _, phrase_records = build_nodes(records, args.top_k_phrases, args.min_df)
    edges = build_edges(nodes, records, phrase_records, args.semantic_threshold, args.lexical_threshold, args.top_semantic_edges)
    output_dir = Path(args.output_dir)
    export_graph(nodes, edges, records, output_dir)
    print(f"Saved KG to: {output_dir}")
    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")
    if args.demo_prompt:
        print(json.dumps(score_prompt(args.demo_prompt, nodes), ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
