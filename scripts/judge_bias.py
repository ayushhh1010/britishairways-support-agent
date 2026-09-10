"""Judge reliability: does a second, independent judge reach the same verdicts?

The primary judge is `gpt-oss-120b` on Groq. This re-grades a subsample with a
Gemini model on Google's stack and reports how far the two agree.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
It measures judge **reliability** (do two independent judges agree?), not judge
**validity** (are they right?). Two judges can agree and both be wrong, and neither
is a substitute for human ratings -- see `scripts/judge_agreement.py`, which is the
real validation and needs a human rater.

It also gives a **self-preference** estimate. The probe judge shares a family with the
generator (both Gemini) while the primary does not, so the probe grading agent replies
more generously than it grades the historical control is evidence of family bias
rather than of general leniency. `historical` is the control that separates the two.
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
from support_agent.evaluation.judge import DIMENSIONS, ReplyJudge  # noqa: E402
from support_agent.evaluation.metrics import agreement, quadratic_weighted_kappa  # noqa: E402
from support_agent.retrieve import HistoricalRetriever  # noqa: E402

# The agent (whose family the probe shares) plus a neutral control both judges
# should score alike if neither is simply more generous than the other.
SYSTEMS = ("agent", "historical")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="cases to re-judge")
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    rdir = PROOT / cfg["paths"]["results"]
    preds = read_jsonl(rdir / "predictions.jsonl")
    if not preds:
        raise SystemExit("No predictions. Run scripts/run_eval.py first.")

    primary = {f"{v['thread_id']}::{v['system']}": v
               for v in read_jsonl(rdir / "judge_verdicts.jsonl") if not v.get("error")}
    if not primary:
        raise SystemExit("No successful primary verdicts. Run scripts/run_eval.py first.")

    usable = [p for p in preds if f"{p['thread_id']}::agent" in primary]
    sample = random.Random(cfg["sampling"]["seed"]).sample(usable, min(args.n, len(usable)))
    print(f"re-judging {len(sample)} cases with the probe judge")

    # Build the probe judge on the MAIN provider (Gemini), overriding the judge
    # provider split so the probe really does run on the generator's own stack.
    probe_model = cfg["llm"]["judge_model_secondary"]
    pcfg = {**cfg, "llm": {**cfg["llm"]}}
    pcfg["llm"]["judge_provider"] = cfg["llm"]["provider"]
    pcfg["llm"]["judge_model"] = probe_model
    judge = ReplyJudge(cfg=pcfg)
    print(f"  primary: {cfg['llm']['judge_model']} via {cfg['llm'].get('judge_provider')}")
    print(f"  probe  : {judge.model} via {judge.client.provider}")

    retriever = HistoricalRetriever.from_disk(cfg)
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

    verdicts = judge.judge_batched(jobs, group_size=args.group_size, desc="probe-judge")
    with (rdir / "judge_verdicts_probe.jsonl").open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.as_dict(), ensure_ascii=False) + "\n")

    probe = {f"{v.thread_id}::{v.system}": v for v in verdicts if not v.error}
    report: dict = {
        "primary_judge": cfg["llm"]["judge_model"],
        "primary_provider": cfg["llm"].get("judge_provider"),
        "probe_judge": probe_model,
        "probe_provider": judge.client.provider,
        "n_requested": len(sample),
        "probe_failures": sum(1 for v in verdicts if v.error),
        "caveat": (
            "Judge-vs-judge agreement measures RELIABILITY, not validity. Two judges "
            "can agree and both be wrong. This does not replace human ratings."
        ),
        "per_system": {},
    }

    for sysname in SYSTEMS:
        keys = [k for k in probe if k.endswith(f"::{sysname}") and k in primary]
        if not keys:
            continue
        rows = []
        for k in keys:
            a, b = primary[k], probe[k]
            rows.append({
                **{f"pri_{d}": a[d] for d in DIMENSIONS},
                **{f"pro_{d}": getattr(b, d) for d in DIMENSIONS},
                "pri_mean": float(np.mean([a[d] for d in DIMENSIONS])),
                "pro_mean": b.mean_score,
                "pri_acc": bool(a["acceptable"]),
                "pro_acc": bool(b.acceptable),
            })
        df = pd.DataFrame(rows)
        report["per_system"][sysname] = {
            "n": len(df),
            "primary_mean": round(float(df.pri_mean.mean()), 3),
            "probe_mean": round(float(df.pro_mean.mean()), 3),
            "probe_minus_primary": round(float((df.pro_mean - df.pri_mean).mean()), 3),
            "mean_absolute_difference": round(float((df.pro_mean - df.pri_mean).abs().mean()), 3),
            "acceptable_agreement": agreement(df.pri_acc.tolist(), df.pro_acc.tolist()),
            "per_dimension_qwk": {
                d: round(quadratic_weighted_kappa(df[f"pri_{d}"], df[f"pro_{d}"]), 4)
                for d in DIMENSIONS
            },
        }

    ps = report["per_system"]
    if "agent" in ps and "historical" in ps:
        report["self_preference_estimate"] = round(
            ps["agent"]["probe_minus_primary"] - ps["historical"]["probe_minus_primary"], 3
        )
        report["self_preference_interpretation"] = (
            "Gap on agent replies minus gap on historical replies. The probe judge shares "
            "a family with the generator; the primary does not. Positive means the probe "
            "favours its own family's output beyond its general leniency."
        )

    dest = rdir / "judge_bias.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
