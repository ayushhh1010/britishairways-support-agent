"""Dump the golden set for human adjudication in a compact, readable form.

Prints every case with both model proposals so an annotator can decide in one pass.
Disagreements and ambiguity flags are marked, because those are the cases where the
pre-labels carry no authority and a human must decide.

  python scripts/adjudicate_dump.py                # all cases
  python scripts/adjudicate_dump.py --only-disputed # just the ones needing a decision
  python scripts/adjudicate_dump.py --start 0 --limit 60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-disputed", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    path = PROOT / cfg["paths"]["golden"] / "prelabels.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run scripts/prelabel_golden.py first")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.only_disputed:
        rows = [r for r in rows if r["needs_adjudication"]]
    rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    for i, r in enumerate(rows, args.start + 1):
        a, b = r["model_a"], r["model_b"]
        mark = "DISPUTED" if r["needs_adjudication"] else "agreed  "
        amb = " AMBIG" if r["ambiguous"] else ""
        print(f"\n--- [{i}] tid={r['thread_id']} {mark}{amb}")
        print(f"  MSG: {r['customer_text'][:230]}")
        print(f"  A: {a['intent']} / {a['action']} ({a.get('escalation_reason_id') or '-'})")
        print(f"  B: {b['intent']} / {b['action']} ({b.get('escalation_reason_id') or '-'})")
        if a.get("second_choice"):
            print(f"  A alt: {a['second_choice']}")
        if a.get("rationale"):
            print(f"  why A: {a['rationale'][:150]}")

    disputed = sum(1 for r in rows if r["needs_adjudication"])
    print(f"\n{'='*70}\n{len(rows)} shown | {disputed} need a human decision")
    print("model A intents:", dict(Counter(r["model_a"]["intent"] for r in rows).most_common()))
    print("model B intents:", dict(Counter(r["model_b"]["intent"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
