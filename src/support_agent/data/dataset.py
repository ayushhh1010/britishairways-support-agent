"""Case selection and splitting for one brand.

A "case" is (first customer message -> brand's first reply). We deliberately model
first-contact triage rather than full multi-turn dialogue; see DECISIONS.md #4.

Split discipline
----------------
`retrieval_pool`, `golden`, and `dev` are disjoint at the *thread* level. The
retriever may only see `retrieval_pool`, so a golden case can never retrieve its
own historical answer. Without this the reply scores are meaningless.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_config

# Promotional / non-support chatter that would pollute a support intent taxonomy.
_RT = re.compile(r"^\s*(rt\b|\.?@\w+\s*rt\b)", re.I)
_MOSTLY_URL = re.compile(r"^\s*(<url>\s*)+$")
_ONLY_PUNCT = re.compile(r"^[\W_]+$")


@dataclass
class Splits:
    retrieval_pool: pd.DataFrame
    golden: pd.DataFrame
    dev: pd.DataFrame

    def summary(self) -> dict[str, int]:
        return {
            "retrieval_pool": len(self.retrieval_pool),
            "golden": len(self.golden),
            "dev": len(self.dev),
        }


def _norm_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def load_cases(cfg: dict | None = None, brand: str | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    brand = brand or cfg["brand"]
    path = ROOT / cfg["paths"]["interim"] / f"cases_{brand}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python scripts/prepare_brand.py {brand}"
        )
    return pd.read_parquet(path)


def filter_cases(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Keep cases usable as supervision: real question, real answer, English, unique."""
    cfg = cfg or load_config()
    s = cfg["sampling"]
    out = df.copy()

    out = out[out["brand_reply"].notna() & out["customer_text"].notna()]
    n_len = len(out)
    out = out[
        out["customer_text"].str.len().between(s["min_customer_chars"], s["max_customer_chars"])
    ]
    out = out[out["brand_reply"].str.split().str.len() >= 5]

    out = out[~out["customer_text"].str.match(_RT)]
    out = out[~out["customer_text"].str.match(_MOSTLY_URL)]
    out = out[~out["customer_text"].str.match(_ONLY_PUNCT)]

    # Exact/near-duplicate customer messages (bot spam, copy-paste complaints).
    out["_norm"] = out["customer_text"].map(_norm_for_dedupe)
    out = out.drop_duplicates(subset="_norm").drop(columns="_norm")

    out = out.reset_index(drop=True)
    out.attrs["filter_report"] = {
        "input": len(df),
        "after_length_filter": n_len,
        "final": len(out),
    }
    return out


def _stable_bucket(thread_id: int, salt: str, n: int) -> int:
    """Deterministic hash bucket. Stable across runs and machines, unlike hash()."""
    h = hashlib.sha256(f"{salt}:{thread_id}".encode()).hexdigest()
    return int(h[:8], 16) % n


def make_splits(df: pd.DataFrame, cfg: dict | None = None) -> Splits:
    cfg = cfg or load_config()
    s = cfg["sampling"]
    rng = np.random.default_rng(s["seed"])

    df = df.copy()
    # Assign every thread a stable [0,1) coordinate so splits never drift.
    df["_u"] = [_stable_bucket(t, "split-v1", 10_000) / 10_000 for t in df["thread_id"]]
    df = df.sort_values("_u").reset_index(drop=True)

    golden_n = int(s["golden_size"])
    pool_n = int(s["retrieval_pool"])

    golden = df.iloc[:golden_n].copy()
    rest = df.iloc[golden_n:].copy()
    pool = rest.iloc[:pool_n].copy()
    dev = rest.iloc[pool_n : pool_n + 400].copy()

    for frame, name in ((golden, "golden"), (pool, "retrieval_pool"), (dev, "dev")):
        frame.drop(columns="_u", inplace=True)
        frame["split"] = name

    assert set(golden.thread_id) & set(pool.thread_id) == set(), "golden leaked into pool"
    assert set(golden.thread_id) & set(dev.thread_id) == set(), "golden leaked into dev"
    _ = rng  # splits are hash-based; rng kept for downstream sampling callers
    return Splits(retrieval_pool=pool, golden=golden, dev=dev)


def build(cfg: dict | None = None, *, write: bool = True) -> Splits:
    cfg = cfg or load_config()
    cases = load_cases(cfg)
    filtered = filter_cases(cases, cfg)
    splits = make_splits(filtered, cfg)
    if write:
        outdir = ROOT / cfg["paths"]["processed"]
        outdir.mkdir(parents=True, exist_ok=True)
        for name in ("retrieval_pool", "golden", "dev"):
            getattr(splits, name).to_parquet(outdir / f"{name}.parquet", index=False)
    return splits


if __name__ == "__main__":
    import json
    import sys

    cfg = load_config()
    cases = load_cases(cfg)
    filtered = filter_cases(cases, cfg)
    sp = make_splits(filtered, cfg)
    build(cfg)
    print(json.dumps({"filter": filtered.attrs["filter_report"], "splits": sp.summary()}, indent=2))
    sys.exit(0)
