from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required: pip install pandas") from exc


def _optional_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


DEFAULT_EXCLUDE_CATEGORIES = [
    "sexual",
    "nudity",
    "porn",
    "erotic",
    "violence",
    "self-harm",
    "self harm",
    "suicide",
    "hate",
    "harassment",
    "weapon",
    "weapons",
    "drug",
    "drugs",
    "illegal",
]


DEFAULT_KEYWORD_BLACKLIST = [
    r"\bnude\b",
    r"\bnaked\b",
    r"\bsex\b",
    r"\bsexual\b",
    r"\bporn\b",
    r"\berotic\b",
    r"\bgenital\b",
    r"\bbreasts?\b",
    r"\bvagina\b",
    r"\bpenis\b",
    r"\bblowjob\b",
    r"\bhandjob\b",
    r"\bstrip(ping|per)?\b",
    r"\blingerie\b",
    r"\bkill\b",
    r"\bmurder\b",
    r"\bstab\b",
    r"\bshot\b",
    r"\bgore\b",
    r"\bblood\b",
    r"\bcorpse\b",
    r"\bsuicide\b",
    r"\bself[- ]?harm\b",
    r"\bnazi\b",
    r"\bkkk\b",
]


DEFAULT_PREINTERCEPT_DETOX_BLACKLIST = list(DEFAULT_KEYWORD_BLACKLIST)


@dataclass
class FilterStats:
    input_rows: int
    after_nonempty_prompt: int
    after_category_filter: int
    after_keyword_filter: int
    after_length_filter: int
    after_dedup: int
    output_rows: int


@dataclass
class PreInterceptStats:
    input_rows: int
    concept_keywords: int
    kept_rows: int
    detoxed_rows: int
    thresholds: List[float]


def _cosine_sim_matrix(a, b, *, eps: float = 1e-8):
    import numpy as np


    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + eps
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + eps
    a_u = a / a_norm
    b_u = b / b_norm
    return a_u @ b_u.T


def _orthogonalize_against_concepts(x, c, *, eps: float = 1e-8):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    if c.size == 0:
        return x, np.zeros((x.shape[1], 0), dtype=np.float32)
    try:
        _, s, vt = np.linalg.svd(c, full_matrices=False)
        r = int((s > eps).sum())
        v = vt[:r].T
        proj = x @ v @ v.T
        return x - proj, v
    except Exception:
        return x, np.zeros((x.shape[1], 0), dtype=np.float32)


def _safe_float_series(df: "pd.DataFrame", col: str) -> Optional["pd.Series"]:
    if col not in df.columns:
        return None
    s = df[col]
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return None


def _split_csv_list(values: Optional[str]) -> List[str]:
    if not values:
        return []
    return [v.strip() for v in values.split(",") if v.strip()]


def _compile_patterns(patterns: Sequence[str], *, case_insensitive: bool = True) -> List[re.Pattern]:
    flags = re.IGNORECASE if case_insensitive else 0
    compiled: List[re.Pattern] = []
    for pat in patterns:
        compiled.append(re.compile(pat, flags=flags))
    return compiled


def _has_any_match(text: str, patterns: Sequence[re.Pattern]) -> bool:
    for p in patterns:
        if p.search(text):
            return True
    return False


def _normalize_prompt(p: str, *, strip_period: bool) -> str:
    p = str(p).strip()
    if strip_period and p.endswith("."):
        p = p[:-1]
    p = re.sub(r"\s+", " ", p)
    return p


def _append_reason(reasons: "pd.Series", mask: "pd.Series", reason: str) -> "pd.Series":


    return reasons.where(~mask, reasons + "|" + reason)


def _category_mask(
    categories_series: "pd.Series",
    excluded: Sequence[str],
    *,
    match_all: bool = False,
) -> "pd.Series":
    excluded = [e for e in excluded if e]
    if not excluded:
        return pd.Series([True] * len(categories_series), index=categories_series.index)
    cats = categories_series.fillna("").astype(str)
    if match_all:
        keep = pd.Series([True] * len(cats), index=cats.index)
        for ex in excluded:
            keep &= ~cats.str.contains(ex, case=False, na=False)
        return keep


    drop = pd.Series([False] * len(cats), index=cats.index)
    for ex in excluded:
        drop |= cats.str.contains(ex, case=False, na=False)
    return ~drop


def build_clean_df(
    df: "pd.DataFrame",
    *,
    prompt_col: str,
    category_col: Optional[str],
    exclude_categories: Sequence[str],
    keyword_patterns: Sequence[str],
    strip_period: bool,
    min_chars: Optional[int],
    max_chars: Optional[int],
    dedup: bool,
    max_inappropriate_pct: Optional[float],
    max_nudity_pct: Optional[float],
    max_q16_pct: Optional[float],
    max_prompt_toxicity: Optional[float],
    min_sd_safety_pct: Optional[float],
    max_hard: Optional[int],
) -> Tuple["pd.DataFrame", FilterStats]:
    if prompt_col not in df.columns:
        raise ValueError(f"Prompt column '{prompt_col}' not found. Available columns: {list(df.columns)}")


    in_rows = int(len(df))
    df = df.copy()
    df[prompt_col] = df[prompt_col].fillna("").astype(str).map(lambda x: _normalize_prompt(x, strip_period=strip_period))
    reasons = pd.Series([""] * len(df), index=df.index, dtype="string")
    keep = pd.Series([True] * len(df), index=df.index)
    empty_mask = df[prompt_col].astype(str).str.len() == 0
    reasons = _append_reason(reasons, empty_mask, "empty_prompt")
    keep &= ~empty_mask
    after_nonempty = int(keep.sum())
    if category_col and category_col in df.columns:
        keep_mask = _category_mask(df[category_col], exclude_categories)
        reasons = _append_reason(reasons, ~keep_mask, "excluded_category")
        keep &= keep_mask
    after_cat = int(keep.sum())
    compiled = _compile_patterns(keyword_patterns)
    if compiled:
        kw_hit = df[prompt_col].astype(str).map(lambda t: _has_any_match(t, compiled))
        reasons = _append_reason(reasons, kw_hit, "keyword_blacklist")
        keep &= ~kw_hit
    after_kw = int(keep.sum())


    if max_inappropriate_pct is not None:
        s = _safe_float_series(df, "inappropriate_percentage")
        if s is not None:
            reasons = _append_reason(reasons, s > float(max_inappropriate_pct), "inappropriate_pct")
            keep &= ~(s > float(max_inappropriate_pct))


    if max_nudity_pct is not None:
        s = _safe_float_series(df, "nudity_percentage")
        if s is not None:
            reasons = _append_reason(reasons, s > float(max_nudity_pct), "nudity_pct")
            keep &= ~(s > float(max_nudity_pct))


    if max_q16_pct is not None:
        s = _safe_float_series(df, "q16_percentage")
        if s is not None:
            reasons = _append_reason(reasons, s > float(max_q16_pct), "q16_pct")
            keep &= ~(s > float(max_q16_pct))


    if max_prompt_toxicity is not None:
        s = _safe_float_series(df, "prompt_toxicity")
        if s is not None:
            reasons = _append_reason(reasons, s > float(max_prompt_toxicity), "prompt_toxicity")
            keep &= ~(s > float(max_prompt_toxicity))


    if min_sd_safety_pct is not None:
        s = _safe_float_series(df, "sd_safety_percentage")
        if s is not None:
            reasons = _append_reason(reasons, s < float(min_sd_safety_pct), "sd_safety_pct")
            keep &= ~(s < float(min_sd_safety_pct))


    if max_hard is not None:
        s = _safe_float_series(df, "hard")
        if s is not None:
            reasons = _append_reason(reasons, s > float(max_hard), "hard")
            keep &= ~(s > float(max_hard))


    if min_chars is not None:
        reasons = _append_reason(
            reasons,
            df[prompt_col].astype(str).str.len() < int(min_chars),
            "min_chars",
        )
        keep &= ~(df[prompt_col].astype(str).str.len() < int(min_chars))
    if max_chars is not None:
        reasons = _append_reason(
            reasons,
            df[prompt_col].astype(str).str.len() > int(max_chars),
            "max_chars",
        )
        keep &= ~(df[prompt_col].astype(str).str.len() > int(max_chars))


    reasons = reasons.astype(str).str.lstrip("|")
    df = df[keep]
    after_len = int(keep.sum())
    if dedup:
        df = df.drop_duplicates(subset=[prompt_col])
    after_dedup = int(len(df))
    stats = FilterStats(
        input_rows=in_rows,
        after_nonempty_prompt=after_nonempty,
        after_category_filter=after_cat,
        after_keyword_filter=after_kw,
        after_length_filter=after_len,
        after_dedup=after_dedup,
        output_rows=after_dedup,
    )
    return df, stats


def _extract_keywords_tfidf(
    texts: Sequence[str],
    *,
    top_k: int,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    stop_words: str | None = "english",
) -> List[str]:
    sklearn = _optional_import("sklearn")
    if sklearn is None:
        raise SystemExit(
            "pre_intercept mode requires either sentence-transformers or scikit-learn. "
            "Install scikit-learn: pip install scikit-learn"
        )


    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        lowercase=True,
    )
    x = vec.fit_transform(list(texts))
    scores = np.asarray(x.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    idx = np.argsort(-scores)[: int(top_k)]
    kws = [str(vocab[i]) for i in idx if scores[i] > 0]
    return kws


def _embed_texts(texts: Sequence[str], *, backend: str, model_name: str, batch_size: int = 64):
    import numpy as np
    backend = str(backend).strip().lower()
    if backend == "sentence_transformers":
        st = _optional_import("sentence_transformers")
        if st is None:
            raise SystemExit(
                "sentence-transformers backend requested but not installed. "
                "Install: pip install sentence-transformers"
            )
        from sentence_transformers import SentenceTransformer


        try:
            p = Path(model_name)
            if not p.exists():
                repo_root = Path(__file__).resolve().parents[2]
                base = Path(str(model_name)).name
                candidate = repo_root / "model" / base
                if candidate.exists() and (candidate / "modules.json").exists():
                    model_name = str(candidate)
        except Exception:
            pass
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise SystemExit(
                "Failed to load sentence-transformers model. "
                "If you are offline, either (1) pass a local model path via --embedding-model, "
                "or (2) use --embedding-backend tfidf (no network required).\n"
                f"embedding_model={model_name}\n"
                f"original_error={type(exc).__name__}: {exc}"
            ) from exc
        emb = model.encode(list(texts), batch_size=int(batch_size), convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(emb, dtype=np.float32)


    if backend == "tfidf":
        sklearn = _optional_import("sklearn")
        if sklearn is None:
            raise SystemExit("TF-IDF embedding backend requires scikit-learn: pip install scikit-learn")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(lowercase=True)
        x = vec.fit_transform(list(texts))
        return x.toarray().astype(np.float32, copy=False)


    raise ValueError(f"Unknown embedding backend: {backend}")


def _risk_level(score: float, thresholds: Sequence[float]) -> int:
    s = float(score)
    for i, t in enumerate(thresholds):
        if s < float(t):
            return int(i)
    return int(len(thresholds))


def _remove_keywords_from_prompt(prompt: str, keywords: Sequence[str]) -> str:
    p = str(prompt)
    for kw in keywords:
        if not kw:
            continue
        kw_s = str(kw).strip()
        if not kw_s:
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", kw_s):
            pat = r"\\b" + re.escape(kw_s) + r"\\b"
        else:
            pat = re.escape(kw_s)
        p = re.sub(pat, " ", p, flags=re.IGNORECASE)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def _remove_regex_patterns_from_prompt(prompt: str, patterns: Sequence[re.Pattern]) -> str:
    p = str(prompt)
    for pat in patterns:
        p = pat.sub(" ", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def pre_intercept(
    *,
    input_df: "pd.DataFrame",
    input_prompt_col: str,
    source_df: "pd.DataFrame",
    source_prompt_col: str,
    target_df: "pd.DataFrame",
    target_prompt_col: str,
    keyword_extractor: str,
    keyword_top_k: int,
    keyword_min_df: int,
    keyword_max_df: float,
    thresholds: Sequence[float],
    embedding_backend: str,
    embedding_model: str,
    embedding_batch_size: int,
    topk_concepts: int,
    detox_strategy: str,
    detox_level: int,
    detox_blacklist_patterns: Sequence[str],
    detox_blacklist_always: bool,
    export_ortho: bool,
    keywords_override: Optional[Sequence[str]] = None,
) -> Tuple["pd.DataFrame", Dict, Optional[dict], PreInterceptStats]:
    import numpy as np
    for df, col, name in (
        (input_df, input_prompt_col, "input"),
        (source_df, source_prompt_col, "source"),
        (target_df, target_prompt_col, "target"),
    ):
        if col not in df.columns:
            raise ValueError(f"{name} prompt column '{col}' not found. Available columns: {list(df.columns)}")


    inp_prompts = input_df[input_prompt_col].fillna("").astype(str).tolist()
    src_prompts = source_df[source_prompt_col].fillna("").astype(str).tolist()
    tgt_prompts = target_df[target_prompt_col].fillna("").astype(str).tolist()
    used_keyword_extractor = str(keyword_extractor).strip().lower()
    if keywords_override is not None:
        keywords = [str(k).strip() for k in keywords_override if str(k).strip()]
        keywords = list(dict.fromkeys(keywords))
        used_keyword_extractor = "json"
    else:
        if used_keyword_extractor == "tfidf":
            kw_s = _extract_keywords_tfidf(
                src_prompts,
                top_k=int(keyword_top_k),
                min_df=int(keyword_min_df),
                max_df=float(keyword_max_df),
            )
            kw_t = _extract_keywords_tfidf(
                tgt_prompts,
                top_k=int(keyword_top_k),
                min_df=int(keyword_min_df),
                max_df=float(keyword_max_df),
            )
        elif used_keyword_extractor == "llm":
            raise SystemExit(
                "keyword_extractor=llm is an interface placeholder. "
                "Please export your LLM-extracted keywords to JSON and use --keywords-json instead."
            )
        else:
            raise ValueError(f"Unsupported keyword_extractor: {used_keyword_extractor}")


        keywords = list(dict.fromkeys([*kw_s, *kw_t]))
    if not keywords:
        raise RuntimeError("No keywords extracted; adjust keyword extraction settings.")
    backend_norm = str(embedding_backend).strip().lower()
    if backend_norm == "tfidf":
        sklearn = _optional_import("sklearn")
        if sklearn is None:
            raise SystemExit("TF-IDF embedding backend requires scikit-learn: pip install scikit-learn")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(lowercase=True)
        joint_corpus = list(dict.fromkeys([*keywords, *inp_prompts, *src_prompts, *tgt_prompts]))
        vec.fit(joint_corpus)
        concept_emb = vec.transform(list(keywords)).toarray().astype(np.float32, copy=False)
        prompt_emb = vec.transform(list(inp_prompts)).toarray().astype(np.float32, copy=False)
    else:
        concept_emb = _embed_texts(
            keywords,
            backend=embedding_backend,
            model_name=embedding_model,
            batch_size=int(embedding_batch_size),
        )
        prompt_emb = _embed_texts(
            inp_prompts,
            backend=embedding_backend,
            model_name=embedding_model,
            batch_size=int(embedding_batch_size),
        )

    sim = _cosine_sim_matrix(prompt_emb, concept_emb)
    max_sim = sim.max(axis=1)
    topk = int(topk_concepts)
    if topk <= 0:
        topk = 5
    top_idx = np.argpartition(-sim, kth=min(topk - 1, sim.shape[1] - 1), axis=1)[:, :topk]
    top_sorted = []
    for i in range(sim.shape[0]):
        idxs = top_idx[i]
        idxs = idxs[np.argsort(-sim[i, idxs])]
        top_sorted.append(idxs)
    top_sorted = np.stack(top_sorted, axis=0)
    rows = []
    detox_strategy = str(detox_strategy).strip().lower()
    detox_level = int(detox_level)
    detox_blacklist_always = bool(detox_blacklist_always)
    detox_blacklist_compiled = _compile_patterns(list(detox_blacklist_patterns)) if detox_blacklist_patterns else []
    for i, p in enumerate(inp_prompts):
        score = float(max_sim[i])
        level = _risk_level(score, thresholds)
        idxs = top_sorted[i].tolist()
        top_concepts = [keywords[j] for j in idxs]
        top_scores = [float(sim[i, j]) for j in idxs]
        original_prompt = p
        clean_prompt = p
        action = "allow"
        if detox_strategy in ("drop", "remove_keywords") and level >= detox_level:
            if detox_strategy == "drop":
                action = "drop"
                clean_prompt = ""
            else:
                action = "remove_keywords"
                clean_prompt = _remove_keywords_from_prompt(p, top_concepts)


        if clean_prompt and detox_blacklist_compiled and (detox_blacklist_always or action != "allow"):
            clean_prompt = _remove_regex_patterns_from_prompt(clean_prompt, detox_blacklist_compiled)
        rows.append(
            {
                "prompt": clean_prompt,
                "risk_score": score,
                "risk_level": level,
                "action": action,
                "top_concepts_json": json.dumps(top_concepts, ensure_ascii=False),
                "top_scores_json": json.dumps(top_scores, ensure_ascii=False),
                "original_prompt": original_prompt,
            }
        )


    out_df = pd.DataFrame(rows)
    ortho_artifact = None
    if export_ortho:
        ortho_emb, basis = _orthogonalize_against_concepts(prompt_emb, concept_emb)
        ortho_artifact = {
            "prompt_embedding": prompt_emb,
            "prompt_embedding_ortho": ortho_emb,
            "concept_embedding": concept_emb,
            "concept_basis": basis,
            "keywords": keywords,
        }


    concepts_artifact = {
        "keywords": keywords,
        "concept_keywords": int(len(keywords)),
        "keyword_extractor": used_keyword_extractor,
        "keyword_top_k": int(keyword_top_k),
        "keyword_min_df": int(keyword_min_df),
        "keyword_max_df": float(keyword_max_df),
        "embedding_backend": str(embedding_backend),
        "embedding_model": str(embedding_model),
        "thresholds": [float(t) for t in thresholds],
        "topk_concepts": int(topk),
    }


    stats = PreInterceptStats(
        input_rows=int(len(inp_prompts)),
        concept_keywords=int(len(keywords)),
        kept_rows=int((out_df["action"] == "allow").sum()),
        detoxed_rows=int((out_df["action"] != "allow").sum()),
        thresholds=[float(t) for t in thresholds],
    )
    return out_df, concepts_artifact, ortho_artifact, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a prompt CSV for filtering or safety pre-interception")
    p.add_argument(
        "--mode",
        choices=["filter", "pre_intercept"],
        default="filter",
        help="filter: legacy deterministic clean CSV; pre_intercept: concept-based classification+detox.",
    )
    p.add_argument("--input-csv", required=True, help="CSV to process (prompts to filter or classify)")
    p.add_argument("--output-csv", required=True, help="Output CSV path")
    p.add_argument("--prompt-col", default="prompt", help="Prompt column name")
    p.add_argument(
        "--category-col",
        default="categories",
        help="Category column name; set to empty string to disable category filtering",
    )
    p.add_argument(
        "--exclude-categories",
        default=",".join(DEFAULT_EXCLUDE_CATEGORIES),
        help="Comma-separated keywords; if category_col contains any, row is dropped",
    )
    p.add_argument(
        "--keyword-blacklist-json",
        default=None,
        help=(
            "Optional JSON file containing a list of regex patterns to exclude (overrides defaults). "
            "If omitted, built-in conservative patterns are used."
        ),
    )
    p.add_argument(
        "--extra-keyword-blacklist",
        default="",
        help="Extra comma-separated regex patterns appended to the blacklist",
    )
    p.add_argument("--strip-period", action="store_true", help="Strip trailing '.' like other scripts do")
    p.add_argument("--min-chars", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    p.add_argument("--max-inappropriate-pct", type=float, default=None, help="Drop if inappropriate_percentage > this")
    p.add_argument("--max-nudity-pct", type=float, default=None, help="Drop if nudity_percentage > this")
    p.add_argument("--max-q16-pct", type=float, default=None, help="Drop if q16_percentage > this")
    p.add_argument("--max-prompt-toxicity", type=float, default=None, help="Drop if prompt_toxicity > this")
    p.add_argument("--min-sd-safety-pct", type=float, default=None, help="Drop if sd_safety_percentage < this")
    p.add_argument("--max-hard", type=int, default=None, help="Drop if hard > this")
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="If set, randomly sample this many rows after filtering",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-format",
        choices=["prompt_only", "keep_all"],
        default="prompt_only",
        help="prompt_only writes only a 'prompt' column; keep_all preserves all columns",
    )
    p.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to write filtering statistics as JSON",
    )

    p.add_argument(
        "--source-csv",
        default=None,
        help="Source domain text CSV (for concept keyword extraction). Required for pre_intercept.",
    )
    p.add_argument(
        "--target-csv",
        default=None,
        help="Target domain (few-shot) text CSV (for concept keyword extraction). Required for pre_intercept.",
    )
    p.add_argument(
        "--source-prompt-col",
        default="prompt",
        help="Prompt column in --source-csv.",
    )
    p.add_argument(
        "--target-prompt-col",
        default="prompt",
        help="Prompt column in --target-csv.",
    )
    p.add_argument(
        "--keyword-extractor",
        choices=["tfidf", "llm"],
        default="tfidf",
        help="Keyword extraction backend. 'llm' is a placeholder (use --keywords-json).",
    )
    p.add_argument(
        "--keywords-json",
        default=None,
        help="Optional JSON list of concept keywords. If set, overrides extraction.",
    )
    p.add_argument("--keyword-top-k", type=int, default=200, help="Top-K keywords to extract per domain")
    p.add_argument("--keyword-min-df", type=int, default=2)
    p.add_argument("--keyword-max-df", type=float, default=0.9)
    p.add_argument(
        "--thresholds",
        default="0.20,0.35,0.50,0.65",
        help="Comma-separated similarity thresholds for risk binning (N thresholds => N+1 levels).",
    )
    p.add_argument(
        "--embedding-backend",
        choices=["sentence_transformers", "tfidf"],
        default="sentence_transformers",
        help="Text embedding backend.",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name for sentence-transformers backend.",
    )
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--topk-concepts", type=int, default=5, help="Top-K concepts to report per prompt")
    p.add_argument(
        "--detox-strategy",
        choices=["none", "remove_keywords", "drop"],
        default="remove_keywords",
        help="Text-level detox action for prompts at/above detox-level.",
    )
    p.add_argument(
        "--detox-level",
        type=int,
        default=2,
        help="If risk_level >= detox_level, apply detox-strategy.",
    )
    p.add_argument(
        "--detox-blacklist-json",
        default=None,
        help=(
            "Optional JSON file containing a list of REGEX patterns to forcibly remove from the cleaned prompt "
            "in pre_intercept mode. If omitted, a built-in conservative blacklist is used."
        ),
    )
    p.add_argument(
        "--extra-detox-blacklist",
        default="",
        help="Extra comma-separated REGEX patterns appended to the pre_intercept detox blacklist.",
    )
    p.add_argument(
        "--detox-blacklist-always",
        action="store_true",
        help="If set, apply the detox blacklist to ALL rows (even action=allow).",
    )
    p.add_argument(
        "--concepts-json-out",
        default=None,
        help="Optional JSON file to write concept keywords and settings.",
    )
    p.add_argument(
        "--export-ortho-embeddings",
        default=None,
        help="Optional NPZ path to export concept/prompt embeddings + orthogonalized prompt embeddings.",
    )
    p.add_argument(
        "--review-csv",
        default=None,
        help=(
            "Optional path to write DROPPED rows for manual review, including a 'drop_reason' column. "
            "(This file may contain harmful prompts; do not open if you don't want to see them.)"
        ),
    )
    p.add_argument(
        "--report-top-dropped-reasons",
        type=int,
        default=0,
        help="Print top-N drop_reason frequencies (requires --review-csv). 0 disables.",
    )
    p.add_argument(
        "--report-top-words",
        type=int,
        default=0,
        help="Print top-N token frequencies in the CLEAN output prompts. 0 disables.",
    )
    p.add_argument(
        "--word-min-len",
        type=int,
        default=3,
        help="Minimum token length for --report-top-words.",
    )
    p.add_argument(
        "--word-regex",
        default=r"[A-Za-z][A-Za-z0-9_\-']+",
        help="Regex used to extract tokens for --report-top-words.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    if args.mode == "pre_intercept":
        if args.source_csv is None or args.target_csv is None:
            raise SystemExit("pre_intercept mode requires --source-csv and --target-csv")
        src_df = pd.read_csv(Path(args.source_csv))
        tgt_df = pd.read_csv(Path(args.target_csv))
        thresholds = [float(x) for x in _split_csv_list(str(args.thresholds))]
        if not thresholds:
            raise SystemExit("--thresholds must contain at least one float")
        keywords_override = None
        if args.keywords_json:
            keywords_override = json.loads(Path(args.keywords_json).read_text())
            if not isinstance(keywords_override, list) or not all(isinstance(x, str) for x in keywords_override):
                raise SystemExit("--keywords-json must be a JSON list of strings")
        if args.detox_blacklist_json:
            patterns = json.loads(Path(args.detox_blacklist_json).read_text())
            if not isinstance(patterns, list) or not all(isinstance(x, str) for x in patterns):
                raise SystemExit("--detox-blacklist-json must be a JSON list of strings")
            detox_blacklist_patterns = patterns
        else:
            detox_blacklist_patterns = list(DEFAULT_PREINTERCEPT_DETOX_BLACKLIST)
        detox_blacklist_patterns += _split_csv_list(args.extra_detox_blacklist)
        out_df, concepts_artifact, ortho_artifact, stats = pre_intercept(
            input_df=df,
            input_prompt_col=args.prompt_col,
            source_df=src_df,
            source_prompt_col=args.source_prompt_col,
            target_df=tgt_df,
            target_prompt_col=args.target_prompt_col,
            keyword_extractor=args.keyword_extractor,
            keyword_top_k=max(1, int(args.keyword_top_k)),
            keyword_min_df=int(args.keyword_min_df),
            keyword_max_df=float(args.keyword_max_df),
            thresholds=thresholds,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=int(args.embedding_batch_size),
            topk_concepts=int(args.topk_concepts),
            detox_strategy=str(args.detox_strategy),
            detox_level=int(args.detox_level),
            detox_blacklist_patterns=detox_blacklist_patterns,
            detox_blacklist_always=bool(args.detox_blacklist_always),
            export_ortho=bool(args.export_ortho_embeddings),
            keywords_override=keywords_override,
        )


        out_df.to_csv(output_csv, index=False)
        print("=== Safety Concept Pre-Interception ===")
        print(f"input_csv={input_csv}")
        print(f"output_csv={output_csv}")
        print(f"concept_keywords={stats.concept_keywords}")
        print(f"thresholds={stats.thresholds}")
        print(f"rows_in={stats.input_rows} kept={stats.kept_rows} detoxed={stats.detoxed_rows}")


        if args.concepts_json_out:
            pth = Path(args.concepts_json_out)
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(json.dumps(concepts_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"concepts_json_out={pth}")


        if args.export_ortho_embeddings:
            import numpy as np
            npz_path = Path(args.export_ortho_embeddings)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            if ortho_artifact is None:
                raise SystemExit("Internal error: export_ortho_embeddings set but ortho_artifact missing")
            np.savez_compressed(
                npz_path,
                prompt_embedding=ortho_artifact["prompt_embedding"],
                prompt_embedding_ortho=ortho_artifact["prompt_embedding_ortho"],
                concept_embedding=ortho_artifact["concept_embedding"],
                concept_basis=ortho_artifact["concept_basis"],
                keywords=np.asarray(ortho_artifact.get("keywords", []), dtype=object),
            )
            print(f"export_ortho_embeddings={npz_path}")
        return


    category_col: Optional[str] = args.category_col
    if category_col is not None and category_col.strip() == "":
        category_col = None
    exclude_categories = _split_csv_list(args.exclude_categories)
    if args.keyword_blacklist_json:
        patterns = json.loads(Path(args.keyword_blacklist_json).read_text())
        if not isinstance(patterns, list) or not all(isinstance(x, str) for x in patterns):
            raise ValueError("--keyword-blacklist-json must be a JSON list of strings")
        keyword_patterns = patterns
    else:
        keyword_patterns = list(DEFAULT_KEYWORD_BLACKLIST)


    keyword_patterns += _split_csv_list(args.extra_keyword_blacklist)
    df_in = df.copy()
    clean_df, stats = build_clean_df(
        df,
        prompt_col=args.prompt_col,
        category_col=category_col,
        exclude_categories=exclude_categories,
        keyword_patterns=keyword_patterns,
        strip_period=bool(args.strip_period),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        dedup=not bool(args.no_dedup),
        max_inappropriate_pct=args.max_inappropriate_pct,
        max_nudity_pct=args.max_nudity_pct,
        max_q16_pct=args.max_q16_pct,
        max_prompt_toxicity=args.max_prompt_toxicity,
        min_sd_safety_pct=args.min_sd_safety_pct,
        max_hard=args.max_hard,
    )


    if args.target_size is not None:
        n = int(args.target_size)
        if n <= 0:
            raise ValueError("--target-size must be > 0")
        if len(clean_df) < n:
            raise ValueError(f"Not enough rows after filtering: {len(clean_df)} < target {n}")
        clean_df = clean_df.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)


    if args.output_format == "prompt_only":
        out = pd.DataFrame({"prompt": clean_df[args.prompt_col].astype(str)})
    else:
        out = clean_df


    if args.review_csv:
        review_path = Path(args.review_csv)
        review_path.parent.mkdir(parents=True, exist_ok=True)


        df_tmp = df_in.copy()
        df_tmp[args.prompt_col] = df_tmp[args.prompt_col].fillna("").astype(str).map(
            lambda x: _normalize_prompt(x, strip_period=bool(args.strip_period))
        )
        reasons = pd.Series([""] * len(df_tmp), index=df_tmp.index, dtype="string")
        empty_mask = df_tmp[args.prompt_col].astype(str).str.len() == 0
        reasons = _append_reason(reasons, empty_mask, "empty_prompt")
        if category_col and category_col in df_tmp.columns:
            keep_mask = _category_mask(df_tmp[category_col], exclude_categories)
            reasons = _append_reason(reasons, ~keep_mask, "excluded_category")
        compiled = _compile_patterns(keyword_patterns)
        if compiled:
            kw_hit = df_tmp[args.prompt_col].astype(str).map(lambda t: _has_any_match(t, compiled))
            reasons = _append_reason(reasons, kw_hit, "keyword_blacklist")
        if args.max_inappropriate_pct is not None:
            s = _safe_float_series(df_tmp, "inappropriate_percentage")
            if s is not None:
                reasons = _append_reason(reasons, s > float(args.max_inappropriate_pct), "inappropriate_pct")
        if args.max_nudity_pct is not None:
            s = _safe_float_series(df_tmp, "nudity_percentage")
            if s is not None:
                reasons = _append_reason(reasons, s > float(args.max_nudity_pct), "nudity_pct")
        if args.max_q16_pct is not None:
            s = _safe_float_series(df_tmp, "q16_percentage")
            if s is not None:
                reasons = _append_reason(reasons, s > float(args.max_q16_pct), "q16_pct")
        if args.max_prompt_toxicity is not None:
            s = _safe_float_series(df_tmp, "prompt_toxicity")
            if s is not None:
                reasons = _append_reason(reasons, s > float(args.max_prompt_toxicity), "prompt_toxicity")
        if args.min_sd_safety_pct is not None:
            s = _safe_float_series(df_tmp, "sd_safety_percentage")
            if s is not None:
                reasons = _append_reason(reasons, s < float(args.min_sd_safety_pct), "sd_safety_pct")
        if args.max_hard is not None:
            s = _safe_float_series(df_tmp, "hard")
            if s is not None:
                reasons = _append_reason(reasons, s > float(args.max_hard), "hard")
        if args.min_chars is not None:
            reasons = _append_reason(
                reasons,
                df_tmp[args.prompt_col].astype(str).str.len() < int(args.min_chars),
                "min_chars",
            )
        if args.max_chars is not None:
            reasons = _append_reason(
                reasons,
                df_tmp[args.prompt_col].astype(str).str.len() > int(args.max_chars),
                "max_chars",
            )
        reasons = reasons.astype(str).str.lstrip("|")
        df_tmp["drop_reason"] = reasons


        dropped = df_tmp[df_tmp["drop_reason"].astype(str).str.len() > 0]
        dropped.to_csv(review_path, index=False)
        print(f"review_csv={review_path} (dropped_rows={len(dropped)})")


        if args.report_top_dropped_reasons and int(args.report_top_dropped_reasons) > 0:
            n = int(args.report_top_dropped_reasons)
            c = Counter(dropped["drop_reason"].astype(str).tolist())
            print("=== Top dropped reasons ===")
            for reason, cnt in c.most_common(n):
                print(f"{cnt}\t{reason}")


    if args.report_top_words and int(args.report_top_words) > 0:
        n = int(args.report_top_words)
        word_min_len = int(args.word_min_len)
        token_re = re.compile(str(args.word_regex))
        c = Counter()
        for p in out["prompt"].astype(str).tolist():
            for m in token_re.findall(p):
                t = m.lower()
                if len(t) < word_min_len:
                    continue
                c[t] += 1
        print("=== Top clean tokens ===")
        for tok, cnt in c.most_common(n):
            print(f"{cnt}\t{tok}")
    out.to_csv(output_csv, index=False)
    stats.output_rows = int(len(out))
    print("=== Clean prompt CSV builder ===")
    print(f"input_csv={input_csv}")
    print(f"output_csv={output_csv}")
    print(f"rows_in={stats.input_rows}")
    print(f"after_nonempty_prompt={stats.after_nonempty_prompt}")
    if category_col:
        print(f"after_category_filter={stats.after_category_filter} (exclude_categories={len(exclude_categories)})")
    else:
        print("after_category_filter=<disabled>")
    print(f"after_keyword_filter={stats.after_keyword_filter} (keyword_patterns={len(keyword_patterns)})")
    print(f"after_length_filter={stats.after_length_filter}")
    print(f"after_dedup={stats.after_dedup}")
    print(f"rows_out={stats.output_rows}")
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))
        print(f"stats_json={stats_path}")


if __name__ == "__main__":
    main()
