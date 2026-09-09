"""Build and persist per-brand case tables (one union-find pass for all brands)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.data.threads import build_threads, iter_cases, load_raw  # noqa: E402


def main(brands: list[str]) -> None:
    df = load_raw(ROOT / "data" / "raw" / "twcs" / "twcs.csv")
    print(f"loaded {len(df):,} rows", file=sys.stderr)
    outdir = ROOT / "data" / "interim"
    outdir.mkdir(parents=True, exist_ok=True)
    for brand in brands:
        threads = build_threads(df, brand)
        cases = pd.DataFrame(list(iter_cases(threads)))
        path = outdir / f"cases_{brand}.parquet"
        cases.to_parquet(path, index=False)
        print(f"{brand}: {len(cases):,} cases -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:] or ["AmazonHelp", "British_Airways"])
