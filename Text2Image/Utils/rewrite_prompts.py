from __future__ import annotations
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required: pip install pandas") from exc


DEFAULT_MODEL = ""
DEFAULT_BASE_URL = ""
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
HARDCODED_API_KEY = ""
DEFAULT_INPUT_CSV = str(WORKSPACE_ROOT / "datasets" / "")
DEFAULT_OUTPUT_CSV = str(WORKSPACE_ROOT / "datasets" / "")
DEFAULT_CACHE_JSONL = str(WORKSPACE_ROOT / "datasets" / "")


UNSAFE_PATTERNS = [
    r"\bnude\b",
    r"\bnaked\b",
    r"\bsex\b",
    r"\bsexual\b",
    r"\bporn\b",
    r"\berotic\b",
    r"\bgenital\b",
    r"\bvagina\b",
    r"\bpenis\b",
    r"\binjur(?:y|ies)\b",
    r"\bdepression\b",
    r"\bdespair\b",
    r"\bkill\b",
    r"\bmurder\b",
    r"\bsuicide\b",
    r"\bself[- ]?harm\b",
    r"\bgore\b",
    r"\bblood\b",
    r"\bbloodiness\b",
    r"\bdeath\b",
    r"\bskeleton\b",
    r"\bhanging\b",
    r"\bnazi\b",
]


SYSTEM_PROMPT = (
    "You rewrite image-generation prompts into benign, safe prompts. "
    "Remove any sexual content, nudity, violence, self-harm, hate, harassment, illegal activity, or explicit harm. "
    "Preserve non-harmful details such as art style, lighting, composition, camera, quality tags, and overall scene structure. "
    "Return ONLY the rewritten prompt, no commentary."
)


SYSTEM_PROMPT_STRICT = (
    "You rewrite image-generation prompts into benign, safe prompts. "
    "You MUST remove or replace any sexual content, nudity, violence, self-harm, hate, harassment, illegal activity, weapons, or explicit harm. "
    "Do not use suggestive euphemisms. Do not include minors. Do not include gore, injury, depression, despair, self-harm, death, skeleton, suicide, hanging, bloodiness. "
    "Preserve only non-harmful details such as art style, lighting, composition, camera, quality tags, and overall scene structure. "
    "If the original intent is unsafe or cannot be rewritten safely, output a generic safe prompt about a peaceful landscape or still life. "
    "Return ONLY the rewritten prompt, no commentary."
)


@dataclass
class RewriteResult:
    idx: int
    original: str
    rewritten: str
    ok: bool
    reason: str


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def _looks_unsafe(text: str, compiled: List[re.Pattern]) -> Optional[str]:
    for p in compiled:
        if p.search(text):
            return p.pattern
    return None


def _is_mostly_cjk(text: str) -> bool:
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    return cjk > max(10, latin)


def _generic_safe_prompt(original: str) -> str:
    if _is_mostly_cjk(original):
        return "."
    return "a peaceful landscape photo, soft natural light, high quality, detailed, balanced composition, realistic photography."


def _load_cache(cache_jsonl: Path) -> Dict[int, RewriteResult]:
    if not cache_jsonl.exists():
        return {}
    out: Dict[int, RewriteResult] = {}
    with cache_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rr = RewriteResult(
                idx=int(obj["idx"]),
                original=str(obj.get("original", "")),
                rewritten=str(obj.get("rewritten", "")),
                ok=bool(obj.get("ok", False)),
                reason=str(obj.get("reason", "")),
            )
            out[rr.idx] = rr
    return out


def _append_cache(cache_jsonl: Path, rr: RewriteResult) -> None:
    cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with cache_jsonl.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "idx": rr.idx,
                    "original": rr.original,
                    "rewritten": rr.rewritten,
                    "ok": rr.ok,
                    "reason": rr.reason,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _get_client(base_url: str):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit("openai is required: pip install -U openai") from exc
    api_key = (os.getenv("OPENAI_API_KEY", "").strip() or HARDCODED_API_KEY.strip())
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set and HARDCODED_API_KEY is empty")


    return OpenAI(api_key=api_key, base_url=base_url)


def rewrite_one(
    client,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite prompts into benign prompts via DeepSeek(OpenAI-compatible)")
    p.add_argument("--prompt", default=None, help="Single prompt to rewrite")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--keep-cols", default="", help="Comma-separated extra columns to keep in output")
    p.add_argument(
        "--keep-all-cols",
        action="store_true",
        help="Keep all input columns in output (prompt will be overwritten unless --output-prompt-col is set)",
    )
    p.add_argument(
        "--output-prompt-col",
        default="prompt",
        help="Output column name for rewritten prompt (default overwrites 'prompt')",
    )
    p.add_argument("--cache-jsonl", default=DEFAULT_CACHE_JSONL, help="JSONL cache for resume")
    p.add_argument("--max-rows", type=int, default=None, help="Optionally only process first N rows")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between requests")
    p.add_argument("--retry", type=int, default=3)
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and rewrite all rows; this will overwrite the cache file.",
    )
    p.add_argument(
        "--recheck-cache",
        action="store_true",
        help="When using cache, recompute postcheck/changed flags for cached rows (does not call API unless --ensure-safe triggers rewrites).",
    )
    p.add_argument(
        "--postcheck",
        action="store_true",
        help="Run regex post-check on rewritten prompt; failures are still written but marked ok=false in cache",
    )
    p.add_argument(
        "--ensure-safe",
        action="store_true",
        help="If postcheck fails, automatically retry with stricter prompting and finally fall back to a generic safe prompt (never drops rows)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max rewrite rounds per row when --ensure-safe is enabled (includes the initial attempt)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ensure_safe:
        args.postcheck = True
    compiled_unsafe = _compile(UNSAFE_PATTERNS)
    if args.prompt:
        client = _get_client(args.base_url)
        rewritten = rewrite_one(
            client,
            model=args.model,
            prompt=str(args.prompt),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.postcheck:
            hit = _looks_unsafe(rewritten, compiled_unsafe)
            if hit:
                print(f"[warn] postcheck matched: {hit}")
        print(rewritten)
        return


    if not args.input_csv or not args.output_csv or not args.cache_jsonl:
        raise SystemExit("For batch mode, require non-empty --input-csv --output-csv --cache-jsonl")
    in_csv = Path(args.input_csv)
    out_csv = Path(args.output_csv)
    cache_jsonl = Path(args.cache_jsonl)
    df = pd.read_csv(in_csv)
    if args.prompt_col not in df.columns:
        raise ValueError(f"prompt col '{args.prompt_col}' not found. columns={list(df.columns)}")
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    if args.force:
        cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
        cache_jsonl.write_text("", encoding="utf-8")
        cache: Dict[int, RewriteResult] = {}
    else:
        cache = _load_cache(cache_jsonl)
    client = _get_client(args.base_url)
    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    if args.keep_all_cols:
        keep_cols = [c for c in df.columns if c != args.prompt_col]
    else:
        for c in keep_cols:
            if c not in df.columns:
                raise ValueError(f"keep col '{c}' not found in input CSV")


    rewritten_prompts: List[str] = [""] * len(df)
    ok_flags: List[bool] = [False] * len(df)
    reasons: List[str] = [""] * len(df)
    changed_flags: List[bool] = [False] * len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        original = str(getattr(row, args.prompt_col)).strip()
        if i in cache:
            rr = cache[i]
            rewritten = rr.rewritten
            ok = rr.ok
            reason = rr.reason
            changed = (rewritten or "").strip() != original
            if args.recheck_cache:
                if args.postcheck:
                    hit = _looks_unsafe((rewritten or "").strip(), compiled_unsafe)
                    if hit:
                        ok = False
                        reason = f"postcheck:{hit}"
                    else:


                        ok = True if (not reason or reason.startswith("postcheck:")) else ok
                        reason = "" if reason.startswith("postcheck:") else reason
                if args.ensure_safe and (not (rewritten or "").strip() or (args.postcheck and not ok)):
                    pass
                else:
                    rewritten_prompts[i] = rewritten
                    ok_flags[i] = ok
                    reasons[i] = reason
                    changed_flags[i] = changed
                    continue
            else:
                rewritten_prompts[i] = rewritten
                ok_flags[i] = ok
                reasons[i] = reason
                changed_flags[i] = changed
                continue


        rewritten: str = ""
        ok = False
        reason = ""
        max_rounds = max(1, int(args.max_rounds))
        system_prompts = [SYSTEM_PROMPT] + [SYSTEM_PROMPT_STRICT] * max(0, max_rounds - 1)
        last_err: Optional[str] = None
        for round_idx, sys_prompt in enumerate(system_prompts[:max_rounds]):
            rewritten_round = ""
            last_err = None
            for attempt in range(int(args.retry)):
                try:
                    rewritten_round = rewrite_one(
                        client,
                        model=args.model,
                        prompt=original,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        system_prompt=sys_prompt,
                    )
                    last_err = None
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    time.sleep(1.0 + attempt)


            if last_err:
                reason = f"api_error:{last_err}"
                continue
            rewritten_round = (rewritten_round or "").strip()
            if not rewritten_round:
                reason = "empty_rewrite"
                continue
            if args.postcheck:
                hit = _looks_unsafe(rewritten_round, compiled_unsafe)
                if hit:
                    reason = f"postcheck:{hit}"
                    if args.ensure_safe:
                        continue
                    rewritten = rewritten_round
                    ok = False
                    break
            rewritten = rewritten_round
            ok = True
            reason = ""
            break


        if not rewritten:
            rewritten = _generic_safe_prompt(original)
            ok = False
            reason = reason or "fallback_generic"
        elif args.ensure_safe and not ok:
            rewritten = _generic_safe_prompt(original)
            ok = False
            reason = reason or "fallback_generic"
        rr = RewriteResult(idx=i, original=original, rewritten=rewritten, ok=ok, reason=reason)
        _append_cache(cache_jsonl, rr)
        rewritten_prompts[i] = rewritten
        ok_flags[i] = ok
        reasons[i] = reason
        changed_flags[i] = (rewritten or "").strip() != original
        if args.sleep_seconds and float(args.sleep_seconds) > 0:
            time.sleep(float(args.sleep_seconds))


    out_df = pd.DataFrame({args.output_prompt_col: rewritten_prompts})
    if keep_cols:
        for c in keep_cols:
            out_df[c] = df[c]
    out_df["rewrite_ok"] = ok_flags
    out_df["rewrite_reason"] = reasons
    out_df["rewrite_changed"] = changed_flags
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    ok_count = int(sum(1 for x in ok_flags if x))
    print("=== Rewrite done ===")
    print(f"input_csv={in_csv}")
    print(f"output_csv={out_csv}")
    print(f"cache_jsonl={cache_jsonl}")
    print(f"rows={len(out_df)} ok={ok_count} not_ok={len(out_df)-ok_count}")


if __name__ == "__main__":
    main()
