"""Show pipeline progress and live provider quota.

The free tier's binding limit is tokens-per-day, per model, on a rolling 24h window,
and it is NOT reported in the rate-limit headers -- it only appears in a 429 body.
This probes each model with a negligible request and reports what it says, so you can
tell at a glance whether re-running the pipeline will make progress or just stall.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def probe(model: str, key: str) -> str:
    """Probe with a REALISTIC payload, not a token-sized one.

    A near-exhausted daily budget still has room for a 20-token "hi", so a small
    probe reports "available" while every real ~1,600-token agent call is rejected.
    This mirrors the actual request size so the answer means something.
    """
    filler = "Classify this airline customer support message carefully. " * 110  # ~1.5k tokens
    body = {"model": model, "messages": [{"role": "user", "content": filler}], "max_tokens": 200}
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps(body), timeout=30,
        )
    except requests.RequestException as exc:
        return f"unreachable ({exc.__class__.__name__})"
    if r.status_code == 200:
        return "available"
    msg = ""
    try:
        msg = r.json().get("error", {}).get("message", "")
    except ValueError:
        msg = r.text[:120]
    if "per day" in msg:
        used = re.search(r"Used (\d+)", msg)
        lim = re.search(r"Limit (\d+)", msg)
        return (
            f"DAILY QUOTA SPENT ({used.group(1) if used else '?'}"
            f"/{lim.group(1) if lim else '?'} tokens) - refills on a rolling 24h window"
        )
    if "output tokens per minute" in msg:
        return "throttled (output tokens/min) - will clear within a minute"
    if "tokens per minute" in msg:
        return "throttled (tokens/min) - will clear within a minute"
    return f"HTTP {r.status_code}: {msg[:90]}"


def main() -> None:
    cfg = load_config()
    gdir = PROOT / cfg["paths"]["golden"]
    rdir = PROOT / cfg["paths"]["results"]

    golden_total = 0
    gp = PROOT / cfg["paths"]["processed"] / "golden.parquet"
    if gp.exists():
        import pandas as pd
        golden_total = len(pd.read_parquet(gp))

    print("PIPELINE")
    rows = [
        ("golden candidates", golden_total, golden_total),
        ("pre-labels", count_lines(gdir / "prelabels.jsonl"), golden_total),
        ("adjudicated labels", count_lines(gdir / "adjudication_author.jsonl"), golden_total),
        ("final golden set", count_lines(gdir / "golden_set.jsonl"), golden_total),
        ("agent predictions", count_lines(rdir / "predictions.jsonl"), golden_total),
        ("judge verdicts", count_lines(rdir / "judge_verdicts.jsonl"), 0),
        ("human reply ratings", sum(count_lines(p) for p in gdir.glob("replies_*.jsonl")), 0),
    ]
    for name, have, want in rows:
        bar = f"{have}/{want}" if want else str(have)
        mark = "OK " if (want and have >= want) or (not want and have) else "-- "
        print(f"  {mark}{name:<22} {bar}")

    cache = PROOT / cfg["paths"]["cache"]
    n_cached = sum(1 for _ in cache.rglob("*.json")) if cache.exists() else 0
    print(f"\n  cached LLM responses: {n_cached}")

    status_p = rdir / "run_status.json"
    if status_p.exists():
        print(f"  last run stopped: {status_p.read_text(encoding='utf-8').strip()}")

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("\nQUOTA: no GROQ_API_KEY set")
        return
    print("\nMODEL QUOTA (live probe)")
    seen = []
    for role in ("gen_model", "judge_model", "prelabel_model_a", "prelabel_model_b"):
        m = cfg["llm"].get(role)
        if not m or m in seen:
            continue
        seen.append(m)
        print(f"  {role:<18} {m:<24} {probe(m, key)}")


if __name__ == "__main__":
    main()
