"""Assemble the final golden set from pre-labels + human adjudication.

Precedence, highest first:
  1. golden/intents_<annotator>.jsonl   -- labels typed in the labelling CLI
  2. golden/adjudication_author.jsonl   -- the author's case-by-case adjudication
  3. golden/prelabels.jsonl             -- dual-model consensus, ONLY where both
                                           models agreed and neither flagged ambiguity

Every row records `label_source`, so the report can state exactly how many labels
were human-decided versus model-consensus. A case that reaches step 3 while still
marked `needs_adjudication` is dropped rather than guessed -- an unadjudicated
ambiguous case is not ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.taxonomy import load_taxonomy  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-consensus", action="store_true", default=True,
                    help="accept dual-model consensus for un-adjudicated, unambiguous cases")
    ap.add_argument("--strict", action="store_true",
                    help="human labels only; drop everything not adjudicated")
    args = ap.parse_args()

    cfg = load_config()
    tax = load_taxonomy()
    gdir = PROOT / cfg["paths"]["golden"]
    golden_df = pd.read_parquet(PROOT / cfg["paths"]["processed"] / "golden.parquet")
    base = {str(r["thread_id"]): r for _, r in golden_df.iterrows()}

    labels: dict[str, dict] = {}
    sources = Counter()

    # 3 - lowest precedence
    if args.allow_consensus and not args.strict:
        for r in read_jsonl(gdir / "prelabels.jsonl"):
            if r.get("needs_adjudication"):
                continue
            if r.get("intent") in tax.names and r.get("action") in ("auto_handle", "escalate"):
                labels[r["key"]] = {
                    "intent": r["intent"], "action": r["action"],
                    "escalation_reason_id": r.get("escalation_reason_id"),
                    "label_source": "dual_model_consensus", "note": None,
                }

    # 2
    for r in read_jsonl(gdir / "adjudication_author.jsonl"):
        if r.get("intent") in tax.names:
            labels[str(r["key"])] = {
                "intent": r["intent"], "action": r["action"],
                "escalation_reason_id": r.get("escalation_reason_id"),
                "label_source": "author_adjudicated", "note": r.get("note"),
            }

    # 1 - highest precedence
    for p in sorted(gdir.glob("intents_*.jsonl")):
        annot = p.stem.replace("intents_", "")
        for r in read_jsonl(p):
            if r.get("intent") in tax.names:
                labels[str(r["key"])] = {
                    "intent": r["intent"], "action": r["action"],
                    "escalation_reason_id": r.get("escalation_reason_id"),
                    "label_source": f"human:{annot}", "note": r.get("note"),
                }

    out = []
    for key, lab in labels.items():
        row = base.get(key)
        if row is None:
            continue
        sources[lab["label_source"]] += 1
        out.append({
            "thread_id": int(row["thread_id"]),
            "customer_text": row["customer_text"],
            "brand_reply": row["brand_reply"],
            **lab,
        })
    out.sort(key=lambda r: r["thread_id"])

    dest = gdir / "golden_set.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total": len(out),
        "candidates_available": len(base),
        "dropped_unadjudicated": len(base) - len(out),
        "by_source": dict(sources),
        "intent_distribution": dict(Counter(r["intent"] for r in out).most_common()),
        "action_distribution": dict(Counter(r["action"] for r in out)),
        "escalation_reasons": dict(
            Counter(r["escalation_reason_id"] for r in out if r["action"] == "escalate").most_common()
        ),
    }
    (PROOT / cfg["paths"]["results"] / "golden_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
