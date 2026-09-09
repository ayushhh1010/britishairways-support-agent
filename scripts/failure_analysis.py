"""Mine the eval outputs for failure modes, with real examples attached.

Produces reports/results/failures.md: the confusion pairs that actually cost
accuracy, every unsafe triage decision, and the lowest-scoring replies with the
judge's critique. This is the raw material for the report's failure analysis --
the hypotheses in REPORT.md are written by hand from what is surfaced here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.judge import DIMENSIONS  # noqa: E402

TARGET = "agent"
TOP_N = 6


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def esc(text: str, n: int = 200) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")[:n]


def main() -> None:
    cfg = load_config()
    rdir = PROOT / cfg["paths"]["results"]
    preds = read_jsonl(rdir / "predictions.jsonl")
    if not preds:
        raise SystemExit("No predictions. Run scripts/run_eval.py first.")
    verdicts = read_jsonl(rdir / "judge_verdicts.jsonl")
    by_key = {f"{v['thread_id']}::{v['system']}": v for v in verdicts}

    L: list[str] = ["# Failure analysis (system: `agent`)", ""]

    # ---- 1. intent confusions ----
    confusions = Counter()
    examples: dict[tuple, list] = defaultdict(list)
    for r in preds:
        got = r[TARGET]["intent"]
        gold = r["gold_intent"]
        if got != gold:
            confusions[(gold, got)] += 1
            examples[(gold, got)].append(r)

    total_wrong = sum(confusions.values())
    L += [f"## 1. Intent confusions ({total_wrong} of {len(preds)} cases wrong)", ""]
    for (gold, got), n in confusions.most_common(8):
        share = n / total_wrong if total_wrong else 0
        L += [f"### `{gold}` -> `{got}`  ({n} cases, {share:.0%} of all errors)", ""]
        for ex in examples[(gold, got)][:3]:
            L.append(f"- {esc(ex['customer_text'])}")
        L.append("")

    # ---- 2. unsafe triage ----
    false_auto = [r for r in preds if r["gold_action"] == "escalate" and r[TARGET]["action"] == "auto_handle"]
    false_esc = [r for r in preds if r["gold_action"] == "auto_handle" and r[TARGET]["action"] == "escalate"]
    L += [f"## 2. Triage errors", "",
          f"- **False auto-handle (costly): {len(false_auto)}** -- needed a human, the bot answered.",
          f"- False escalate (cheap): {len(false_esc)} -- answerable, but sent to a human.", ""]
    L += ["### Every false auto-handle", ""]
    for r in false_auto[:20]:
        L += [f"- **{esc(r['customer_text'])}**",
              f"  - agent said: _{esc(r[TARGET].get('escalation_rationale'), 160)}_",
              f"  - drafted: _{esc(r[TARGET].get('reply'), 200)}_"]
    L.append("")

    # ---- 3. worst replies by judge score ----
    scored = []
    for r in preds:
        v = by_key.get(f"{r['thread_id']}::{TARGET}")
        if v:
            scored.append((sum(v[d] for d in DIMENSIONS) / len(DIMENSIONS), r, v))
    scored.sort(key=lambda t: t[0])
    L += ["## 3. Lowest-scoring drafted replies", ""]
    for mean, r, v in scored[:TOP_N]:
        dims = " ".join(f"{d[:4]}={v[d]}" for d in DIMENSIONS)
        L += [f"### mean {mean:.2f}  ({dims})", "",
              f"- customer: {esc(r['customer_text'])}",
              f"- agent:    {esc(r[TARGET].get('reply'), 300)}",
              f"- historical: {esc(r.get('historical_reply'), 200)}",
              f"- judge critique: _{esc(v.get('critique'), 200)}_", ""]

    # ---- 4. groundedness vs retrieval quality ----
    L += ["## 4. Groundedness by judge dimension (agent)", ""]
    agent_v = [v for v in verdicts if v["system"] == TARGET]
    if agent_v:
        for d in DIMENSIONS:
            dist = Counter(v[d] for v in agent_v)
            row = "  ".join(f"{s}:{dist.get(s,0)}" for s in (1, 2, 3, 4, 5))
            L.append(f"- `{d}` score distribution -> {row}")
    L.append("")

    dest = rdir / "failures.md"
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    print(f"intent errors: {total_wrong}/{len(preds)} | false auto-handle: {len(false_auto)} | false escalate: {len(false_esc)}")


if __name__ == "__main__":
    main()
