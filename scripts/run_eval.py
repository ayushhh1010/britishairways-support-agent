"""Main evaluation harness.

Runs three systems over the golden set and scores all of them identically:

  B0_trivial  majority intent / fixed triage / canned reply
  B1_simple   TF-IDF+LR intent (out-of-fold) / keyword triage / nearest historical reply
  agent       retrieval-grounded LLM: intent + triage + drafted reply

plus `historical`, the real BA reply, scored by the same judge as a reference point.

Outputs
-------
  reports/results/eval_results.json   every metric, with bootstrap CIs
  reports/results/predictions.jsonl   per-case predictions from every system
  reports/results/judge_verdicts.jsonl
  reports/results/judge_sample.jsonl  blind sample for a human to rate
  reports/results/RESULTS.md          the headline table
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import SupportAgent  # noqa: E402
from support_agent.baselines import SimpleBaseline, TrivialBaseline  # noqa: E402
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.judge import ReplyJudge, aggregate  # noqa: E402
from support_agent.evaluation.metrics import answer_rate, score_intents, score_triage  # noqa: E402
from support_agent.llm import LLMClient  # noqa: E402
from support_agent.retrieve import HistoricalRetriever  # noqa: E402
from support_agent.taxonomy import load_taxonomy  # noqa: E402

SYSTEMS = ("B0_trivial", "B1_simple", "agent")


def load_golden(cfg) -> pd.DataFrame:
    path = PROOT / cfg["paths"]["golden"] / "golden_set.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "Build it first:\n"
            "  python scripts/prelabel_golden.py\n"
            "  python scripts/build_golden.py"
        )
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    missing = {"thread_id", "customer_text", "intent", "action"} - set(df.columns)
    if missing:
        raise SystemExit(f"golden_set.jsonl missing columns: {missing}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (metrics only)")
    ap.add_argument("--judge-n", type=int, default=0, help="judge only N cases per system (0=all)")
    ap.add_argument("--ablation", action="store_true", help="also run the no-retrieval ablation")
    ap.add_argument("--allow-partial", action="store_true",
                    help="publish metrics even if the provider quota cut the run short")
    args = ap.parse_args()

    cfg = load_config()
    tax = load_taxonomy()
    results_dir = PROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    golden = load_golden(cfg)
    if args.limit:
        golden = golden.head(args.limit)
    messages = golden["customer_text"].tolist()
    y_intent = golden["intent"].tolist()
    y_action = golden["action"].tolist()
    print(f"golden set: {len(golden)} cases, {golden['intent'].nunique()} intents present")

    retriever = HistoricalRetriever.from_disk(cfg)
    client = LLMClient(cfg)

    preds: dict[str, pd.DataFrame] = {}

    # ---- B0 trivial ----
    b0 = TrivialBaseline().fit(y_intent, y_action)
    preds["B0_trivial"] = b0.predict(messages)

    # ---- B1 simple ----
    b1 = SimpleBaseline(retriever, seed=cfg["sampling"]["seed"])
    preds["B1_simple"] = b1.predict(messages, y_intent)

    # ---- agent ----
    agent = SupportAgent(client=client, retriever=retriever, taxonomy=tax, cfg=cfg)
    outs = agent.run_batch(golden.to_dict("records"), desc="agent")
    preds["agent"] = pd.DataFrame([o.as_dict() for o in outs])

    # A quota wall is not a prediction. Publishing metrics over cases the agent never
    # actually saw would silently understate it, so stop and say so instead.
    blocked = sum(1 for o in outs if (o.error or "").startswith("QUOTA"))
    if blocked and not args.allow_partial:
        done = len(outs) - blocked
        (results_dir / "run_status.json").write_text(json.dumps({
            "stage": "agent", "completed": done, "blocked_by_quota": blocked,
            "total": len(outs),
        }, indent=2), encoding="utf-8")
        msg = [
            "",
            "STOPPED: provider daily token quota reached.",
            f"  {done}/{len(outs)} agent cases completed and cached.",
            "  Re-run `make eval` once the quota window refills; cached cases replay",
            f"  instantly and only the remaining {blocked} will hit the API.",
            f"  (Use --allow-partial to publish metrics over the {done} completed cases.)",
        ]
        print("\n".join(msg))
        raise SystemExit(3)

    systems = list(SYSTEMS)
    if args.ablation:
        abl = agent.run_batch(golden.to_dict("records"), classify_only=True, desc="ablation(no-RAG)")
        preds["agent_no_retrieval"] = pd.DataFrame([o.as_dict() for o in abl])
        systems.append("agent_no_retrieval")

    # ---- automated metrics ----
    results: dict = {"n_cases": len(golden), "systems": {}}
    tri_cfg = cfg["triage"]
    for name in systems:
        p = preds[name]
        results["systems"][name] = {
            "intent": score_intents(y_intent, p["intent"].tolist(), tax.names).as_dict(),
            "triage": score_triage(
                y_action,
                p["action"].tolist(),
                false_auto_handle_cost=tri_cfg["false_auto_handle_cost"],
                false_escalate_cost=tri_cfg["false_escalate_cost"],
            ).as_dict(),
        }

    # ---- per-case predictions ----
    with (results_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in golden.reset_index(drop=True).iterrows():
            rec = {
                "thread_id": int(row["thread_id"]),
                "customer_text": row["customer_text"],
                "gold_intent": row["intent"],
                "gold_action": row["action"],
                "historical_reply": row.get("brand_reply"),
            }
            for name in systems:
                p = preds[name].iloc[i]
                rec[name] = {
                    "intent": p["intent"],
                    "action": p["action"],
                    "reply": p.get("reply", ""),
                    "escalation_rationale": p.get("escalation_rationale", ""),
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- LLM judge ----
    if not args.no_judge:
        judge = ReplyJudge(client=client, cfg=cfg)
        ev_all = retriever.search_batch(messages)
        idx = list(range(len(golden)))
        if args.judge_n and args.judge_n < len(idx):
            idx = random.Random(cfg["sampling"]["seed"]).sample(idx, args.judge_n)

        jobs = []
        for i in idx:
            row = golden.iloc[i]
            hist = row.get("brand_reply") or ""
            for name in ("B0_trivial", "B1_simple", "agent"):
                cand = preds[name].iloc[i].get("reply", "")
                jobs.append({
                    "thread_id": int(row["thread_id"]), "system": name,
                    "message": row["customer_text"], "candidate": cand,
                    "historical_reply": hist, "evidence": ev_all[i],
                })
            jobs.append({
                "thread_id": int(row["thread_id"]), "system": "historical",
                "message": row["customer_text"], "candidate": hist,
                "historical_reply": hist, "evidence": ev_all[i],
            })

        verdicts = judge.judge_batch(jobs, desc="judge")
        jblocked = sum(1 for v in verdicts if (v.error or "").startswith("QUOTA"))
        if jblocked and not args.allow_partial:
            print(
                f"\nSTOPPED during judging: {len(verdicts) - jblocked}/{len(verdicts)} "
                "verdicts completed and cached. Re-run `make eval` after the quota "
                "window refills."
            )
            raise SystemExit(3)
        with (results_dir / "judge_verdicts.jsonl").open("w", encoding="utf-8") as fh:
            for v in verdicts:
                fh.write(json.dumps(v.as_dict(), ensure_ascii=False) + "\n")
        results["judge"] = aggregate(verdicts)
        results["judge_model"] = cfg["llm"]["judge_model"]

        # Blind sample for a human rater: system labels stripped and order shuffled.
        rng = random.Random(cfg["sampling"]["seed"])
        sample = rng.sample(jobs, min(60, len(jobs)))
        with (results_dir / "judge_sample.jsonl").open("w", encoding="utf-8") as fh:
            for j in sample:
                ev = j["evidence"][:2]
                fh.write(json.dumps({
                    "key": f"{j['thread_id']}::{j['system']}",
                    "thread_id": j["thread_id"],
                    "message": j["message"],
                    "candidate": j["candidate"],
                    "evidence_preview": " | ".join(e.brand_reply[:110] for e in ev),
                }, ensure_ascii=False) + "\n")

    results["usage"] = client.usage.as_dict()
    results["gen_model"] = cfg["llm"]["gen_model"]
    (results_dir / "eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_markdown(results, results_dir / "RESULTS.md", systems)
    print(f"\nwrote {results_dir/'eval_results.json'} and RESULTS.md")
    print(json.dumps({k: v["intent"]["accuracy"] for k, v in results["systems"].items()}, indent=2))


def write_markdown(results: dict, path: Path, systems: list[str]) -> None:
    L = [f"# Results ({results['n_cases']} golden cases)", ""]
    L += ["## Intent classification", "",
          "| system | accuracy | 95% CI | macro-F1 | 95% CI |",
          "|---|---|---|---|---|"]
    for s in systems:
        m = results["systems"][s]["intent"]
        L.append(
            f"| `{s}` | {m['accuracy']:.3f} | [{m['accuracy_ci95'][0]:.3f}, {m['accuracy_ci95'][1]:.3f}] "
            f"| {m['macro_f1']:.3f} | [{m['macro_f1_ci95'][0]:.3f}, {m['macro_f1_ci95'][1]:.3f}] |"
        )

    L += ["", "## Triage (escalate vs auto-handle)", "",
          "| system | accuracy | 95% CI | esc. precision | esc. recall | false auto-handle | cost/case |",
          "|---|---|---|---|---|---|---|"]
    for s in systems:
        m = results["systems"][s]["triage"]
        L.append(
            f"| `{s}` | {m['accuracy']:.3f} | [{m['accuracy_ci95'][0]:.3f}, {m['accuracy_ci95'][1]:.3f}] "
            f"| {m['escalate_precision']:.3f} | {m['escalate_recall']:.3f} "
            f"| {m['false_auto_handle']} | {m['cost_per_case']:.3f} |"
        )

    L += ["", "## Answer rate (of cases the system chose to auto-handle)", "",
          "A deflection here is a system claiming it can handle a case and then not "
          "answering it. The judge rubric does not punish this; that is the point.", "",
          "| system | auto-handled | answered | deflected |", "|---|---|---|---|"]
    for s in systems:
        m = results["systems"][s].get("answer_rate", {})
        if not m or not m.get("n_auto_handled"):
            L.append(f"| `{s}` | 0 | - | - |")
            continue
        L.append(
            f"| `{s}` | {m['n_auto_handled']} | {m['answer_rate']:.1%} | {m['deflection_rate']:.1%} |"
        )

    if "judge" in results:
        L += ["", f"## Reply quality (LLM judge: `{results.get('judge_model')}`)", "",
              "| system | grounded | helpful | tone | safety | mean | acceptable | beats historical |",
              "|---|---|---|---|---|---|---|---|"]
        for s, m in results["judge"].items():
            L.append(
                f"| `{s}` | {m['groundedness']:.2f} | {m['helpfulness']:.2f} | {m['tone']:.2f} "
                f"| {m['safety']:.2f} | {m['mean_score']:.2f} | {m['acceptable_rate']:.1%} "
                f"| {m['beats_historical_rate']:.1%} |"
            )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
