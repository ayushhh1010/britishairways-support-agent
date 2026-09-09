"""Quantify LLM-judge self-preference bias.

The main evaluation is judged by `qwen/qwen3.8-27b`, a different family from the
generator `openai/gpt-oss-120b`. That choice is standard advice, but advice is not
evidence. This script re-judges a subsample with `gpt-oss-120b` -- the generator's
OWN family -- and reports the difference.

If the same-family judge scores the agent's replies materially higher than the
independent judge does, while both agree about the historical replies, that gap is
self-preference bias, measured rather than asserted. It also tells a reader how
much the headline reply scores would move under a different judge.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.judge import DIMENSIONS, ReplyJudge, aggregate  # noqa: E402
from support_agent.llm import LLMClient  # noqa: E402
from support_agent.retrieve import HistoricalRetriever  # noqa: E402

# Only these two systems are needed: the agent (whose family the probe judge shares)
# and the historical replies (a neutral control both judges should score alike).
SYSTEMS = ("agent", "historical")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="cases to re-judge")
    args = ap.parse_args()

    cfg = load_config()
    rdir = PROOT / cfg["paths"]["results"]
    preds = read_jsonl(rdir / "predictions.jsonl")
    if not preds:
        raise SystemExit("No predictions. Run scripts/run_eval.py first.")

    primary = {f"{v['thread_id']}::{v['system']}": v for v in read_jsonl(rdir / "judge_verdicts.jsonl")}
    # Only re-judge cases the primary judge actually scored, else there is nothing to compare.
    usable = [p for p in preds if f"{p['thread_id']}::agent" in primary]
    rng = random.Random(cfg["sampling"]["seed"])
    sample = rng.sample(usable, min(args.n, len(usable)))
    print(f"re-judging {len(sample)} cases with the same-family probe judge")

    client = LLMClient(cfg)
    retriever = HistoricalRetriever.from_disk(cfg)
    probe_model = cfg["llm"]["judge_model_secondary"]

    judge = ReplyJudge(client=client, cfg=cfg)
    judge.model = probe_model  # override: this is the whole point of the script

    ev = retriever.search_batch([p["customer_text"] for p in sample])
    jobs = []
    for p, e in zip(sample, ev):
        for sysname in SYSTEMS:
            cand = p["historical_reply"] if sysname == "historical" else p[sysname]["reply"]
            jobs.append({
                "thread_id": p["thread_id"], "system": sysname,
                "message": p["customer_text"], "candidate": cand or "",
                "historical_reply": p["historical_reply"] or "", "evidence": e,
            })

    verdicts = judge.judge_batch(jobs, desc="probe-judge")
    with (rdir / "judge_verdicts_probe.jsonl").open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.as_dict(), ensure_ascii=False) + "\n")

    probe = {f"{v.thread_id}::{v.system}": v for v in verdicts}
    report: dict = {
        "independent_judge": cfg["llm"]["judge_model"],
        "probe_judge_same_family_as_generator": probe_model,
        "n_cases": len(sample),
        "per_system": {},
    }
    for sysname in SYSTEMS:
        keys = [k for k in probe if k.endswith(f"::{sysname}") and k in primary]
        if not keys:
            continue
        rows = []
        for k in keys:
            ind = primary[k]
            pr = probe[k]
            rows.append({
                **{f"ind_{d}": ind[d] for d in DIMENSIONS},
                **{f"pro_{d}": getattr(pr, d) for d in DIMENSIONS},
                "ind_mean": float(np.mean([ind[d] for d in DIMENSIONS])),
                "pro_mean": pr.mean_score,
                "ind_acc": bool(ind["acceptable"]),
                "pro_acc": bool(pr.acceptable),
            })
        df = pd.DataFrame(rows)
        report["per_system"][sysname] = {
            "n": len(df),
            "independent_mean": round(float(df.ind_mean.mean()), 3),
            "probe_mean": round(float(df.pro_mean.mean()), 3),
            "probe_minus_independent": round(float((df.pro_mean - df.ind_mean).mean()), 3),
            "independent_acceptable_rate": round(float(df.ind_acc.mean()), 4),
            "probe_acceptable_rate": round(float(df.pro_acc.mean()), 4),
            "per_dimension_gap": {
                d: round(float((df[f"pro_{d}"] - df[f"ind_{d}"]).mean()), 3) for d in DIMENSIONS
            },
        }

    ps = report["per_system"]
    if "agent" in ps and "historical" in ps:
        # The control subtracts out any general leniency of the probe judge, leaving
        # the part of the gap that is specific to its own family's output.
        report["self_preference_estimate"] = round(
            ps["agent"]["probe_minus_independent"] - ps["historical"]["probe_minus_independent"], 3
        )
        report["interpretation"] = (
            "Gap on agent replies minus gap on historical replies. Positive means the "
            "same-family judge favours its own family's output beyond its general leniency."
        )

    dest = rdir / "judge_bias.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
