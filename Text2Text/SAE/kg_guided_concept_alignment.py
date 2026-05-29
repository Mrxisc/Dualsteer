from __future__ import annotations
import argparse
import datetime
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from sae_model import TopKSAE
SAE_ROOT = Path(__file__).resolve().parent
DAJA_ROOT = SAE_ROOT.parent
DEFAULT_ACT_DIR = SAE_ROOT / "activation" / "qwen3_sae_train"
DEFAULT_SAE_ROOT = SAE_ROOT / "SAEs"
DEFAULT_KG_DIR = DAJA_ROOT / "KG" / "outputs"
DEFAULT_OUTPUT = DEFAULT_SAE_ROOT / "qwen3_kg_sae_concept_alignment.json"
LOG_ROOT = SAE_ROOT / "logs"
def setup_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"kg_guided_concept_alignment_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Log file: {log_path}")
def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
def normalize_text(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def split_cell(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]
def is_benign(label: str, dataset: str) -> bool:
    text = f"{label} {dataset}".lower()
    return "benign" in text
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
def load_kg_tables(kg_dir: Path, min_risk: float, min_frequency: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(kg_dir / "nodes.csv")
    edges = pd.read_csv(kg_dir / "edges.csv")
    prompts = pd.read_csv(kg_dir / "source_prompts.csv")
    nodes = nodes[(nodes["risk_score"] >= min_risk) & (nodes["frequency"] >= min_frequency)].copy()
    nodes = nodes[nodes["id"].astype(str).str.startswith("concept:")].copy()
    logging.info(f"Loaded {len(nodes)} KG concept nodes after filtering")
    return nodes, edges, prompts
def build_prompt_index(prompts: pd.DataFrame) -> dict[str, dict]:
    return {str(row["id"]): row.to_dict() for _, row in prompts.iterrows()}
def build_concept_prompt_sets(nodes: pd.DataFrame, prompt_index: dict[str, dict]) -> dict[str, set[str]]:
    concept_to_records: dict[str, set[str]] = defaultdict(set)
    for _, node in nodes.iterrows():
        concept_id = str(node["id"])
        label = normalize_text(node["label"])
        aliases = [normalize_text(x) for x in split_cell(node.get("aliases", ""))]
        categories = [normalize_text(x) for x in split_cell(node.get("categories", ""))]
        terms = [x for x in [label, *aliases] if x]
        category_terms = [x for x in categories if x]
        for record_id, record in prompt_index.items():
            prompt = normalize_text(record.get("prompt", ""))
            category = normalize_text(record.get("category", ""))
            behavior = normalize_text(record.get("behavior", ""))
            matched_text = any(term and (term in prompt or term in behavior or term in category) for term in terms)
            matched_category = any(cat and (cat in category or category in cat) for cat in category_terms)
            if matched_text or matched_category:
                concept_to_records[concept_id].add(record_id)
    return concept_to_records
def build_graph_neighbor_risk(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float]:
    node_risk = {str(row["id"]): float(row["risk_score"]) for _, row in nodes.iterrows()}
    neighbor_scores: dict[str, list[float]] = defaultdict(list)
    for _, edge in edges.iterrows():
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        weight = float(edge.get("weight", 0.0))
        if src in node_risk and tgt in node_risk:
            neighbor_scores[src].append(node_risk[tgt] * weight)
            neighbor_scores[tgt].append(node_risk[src] * weight)
    graph_risk = {}
    for concept_id, risk in node_risk.items():
        neigh = neighbor_scores.get(concept_id, [])
        if neigh:
            graph_risk[concept_id] = max(0.0, min(1.0, 0.7 * risk + 0.3 * (sum(neigh) / len(neigh))))
        else:
            graph_risk[concept_id] = max(0.0, min(1.0, risk))
    return graph_risk
@torch.no_grad()
def encode_shards(args: argparse.Namespace, layer: int, device: torch.device) -> tuple[torch.Tensor, list[str], list[str], list[str]]:
    sae = load_sae(args.sae_root, layer, device)
    z_chunks = []
    record_ids = []
    labels = []
    datasets = []
    shard_paths = sorted((args.activation_dir / f"layer_{layer}").glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No activation shards for layer {layer}")
    for shard_path in tqdm(shard_paths, desc=f"encode layer {layer}", dynamic_ncols=True):
        shard = torch.load(shard_path, map_location="cpu")
        acts = shard["activations"].float().to(device)
        z = sae.encode(acts).detach().cpu().float()
        z_chunks.append(z)
        record_ids.extend([str(x) for x in shard.get("record_ids", [])])
        labels.extend([str(x) for x in shard.get("labels", [""] * z.shape[0])])
        datasets.extend([str(x) for x in shard.get("datasets", [""] * z.shape[0])])
    return torch.cat(z_chunks, dim=0), record_ids, labels, datasets
def stability_score(values: torch.Tensor, active_quantile: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros(values.shape[1])
    threshold = torch.quantile(values, active_quantile, dim=0)
    active_rate = (values >= threshold.unsqueeze(0)).float().mean(dim=0)
    cv = values.std(dim=0) / values.mean(dim=0).clamp_min(1e-6)
    return active_rate / (1.0 + cv)
def score_concepts_for_layer(
    args: argparse.Namespace,
    layer: int,
    nodes: pd.DataFrame,
    concept_prompt_sets: dict[str, set[str]],
    graph_risk: dict[str, float],
    device: torch.device,
) -> dict:
    z, record_ids, labels, datasets = encode_shards(args, layer, device)
    d_sae = z.shape[1]
    benign_indices = [i for i, (lab, ds) in enumerate(zip(labels, datasets)) if is_benign(lab, ds)]
    if not benign_indices:
        raise RuntimeError(f"Layer {layer}: no benign activations found")
    benign_mean = z[benign_indices].mean(dim=0)
    record_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record_id in enumerate(record_ids):
        record_to_indices[record_id].append(idx)
    node_meta = {str(row["id"]): row.to_dict() for _, row in nodes.iterrows()}
    concept_results = []
    for concept_id, concept_records in concept_prompt_sets.items():
        indices = []
        for record_id in concept_records:
            indices.extend(record_to_indices.get(record_id, []))
        indices = sorted(set(indices))
        if len(indices) < args.min_concept_samples:
            continue
        concept_z = z[indices]
        concept_mean = concept_z.mean(dim=0)
        sel = torch.clamp(concept_mean - benign_mean, min=0.0)
        if float(sel.max()) > 0:
            sel = sel / sel.max().clamp_min(1e-6)
        sta = stability_score(concept_z, args.active_quantile)
        risk = float(graph_risk.get(concept_id, 0.0))
        alignment = sel * sta * risk
        positive = torch.nonzero(alignment > args.min_alignment, as_tuple=False).flatten()
        if positive.numel() == 0:
            continue
        ranked = positive[torch.argsort(alignment[positive], descending=True)][: args.top_k_features]
        features = []
        for feature_idx in ranked.tolist():
            features.append({
                "feature": int(feature_idx),
                "alignment": round(float(alignment[feature_idx]), 8),
                "selectivity": round(float(sel[feature_idx]), 8),
                "stability": round(float(sta[feature_idx]), 8),
                "risk": round(risk, 8),
                "concept_mean": round(float(concept_mean[feature_idx]), 8),
                "benign_mean": round(float(benign_mean[feature_idx]), 8),
            })
        meta = node_meta.get(concept_id, {})
        concept_results.append({
            "concept_id": concept_id,
            "label": meta.get("label", ""),
            "risk_score": float(meta.get("risk_score", 0.0)),
            "graph_risk": round(risk, 8),
            "sample_count": len(indices),
            "features": features,
        })
    concept_results.sort(key=lambda item: item["features"][0]["alignment"] if item["features"] else 0.0, reverse=True)
    logging.info(f"[layer {layer}] aligned_concepts={len(concept_results)}, d_sae={d_sae}")
    return {
        "layer": layer,
        "d_sae": d_sae,
        "benign_count": len(benign_indices),
        "aligned_concepts": concept_results,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-guided KG-to-SAE concept alignment for Qwen3 text-to-text safety.")
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACT_DIR)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=str, default="30,32,33,34,35")
    parser.add_argument("--min-risk", type=float, default=0.5)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--min-concept-samples", type=int, default=3)
    parser.add_argument("--top-k-features", type=int, default=32)
    parser.add_argument("--min-alignment", type=float, default=0.0)
    parser.add_argument("--active-quantile", type=float, default=0.75)
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()
    setup_logging()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda is not None else "cpu")
    layers = parse_layers(args.layers)
    nodes, edges, prompts = load_kg_tables(args.kg_dir, args.min_risk, args.min_frequency)
    prompt_index = build_prompt_index(prompts)
    concept_prompt_sets = build_concept_prompt_sets(nodes, prompt_index)
    graph_risk = build_graph_neighbor_risk(nodes, edges)
    logging.info(f"Concepts with matched prompts: {sum(1 for v in concept_prompt_sets.values() if v)}")
    result = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "definition": "A(c,z_j)=Sel(z_j;D_c)*Sta(z_j;D_c)*Risk(c)",
        "selectivity": "normalized positive activation gap between concept-conditioned samples and benign references",
        "stability": "activation consistency over concept-conditioned prompt groups",
        "risk": "edge-weighted KG risk prior",
        "activation_dir": str(args.activation_dir),
        "sae_root": str(args.sae_root),
        "kg_dir": str(args.kg_dir),
        "layers": {},
    }
    for layer in layers:
        result["layers"][str(layer)] = score_concepts_for_layer(
            args=args,
            layer=layer,
            nodes=nodes,
            concept_prompt_sets=concept_prompt_sets,
            graph_risk=graph_risk,
            device=device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved KG-guided concept alignment to: {args.output}")
if __name__ == "__main__":
    main()
