"""Human (golden) vs model (pre-label) label agreement.

This is the annotation-quality evidence for the golden set. The author's labels were
written BLIND -- before looking at any pre-label -- so this is a genuine comparison
of two independent annotators, not a measure of how often a human clicked "accept".

It is NOT the headline agent result. Pre-labels come from a different prompt with no
retrieval and no reply drafting, so treat these numbers as a difficulty floor for the
task, not as the agent's score.

Reported separately for intent and for triage action, because they behave very
differently and the difference is the interesting part.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.metrics import agreement  # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    cfg = load_config()
    gdir = PROOT / cfg["paths"]["golden"]
    human = {str(r["thread_id"]): r for r in read(gdir / "golden_set.jsonl")}
    pre = {str(r["thread_id"]): r for r in read(gdir / "prelabels.jsonl")}

    pairs = [(human[k], pre[k]) for k in human if k in pre and pre[k]["model_a"]["intent"]]
    if not pairs:
        raise SystemExit("no paired cases -- run prelabel_golden.py and build_golden.py")

    hi = [h["intent"] for h, _ in pairs]
    mi = [p["model_a"]["intent"] for _, p in pairs]
    ha = [h["action"] for h, _ in pairs]
    ma = [p["model_a"]["action"] for _, p in pairs]

    # `other` was added to the taxonomy DURING annotation, after these pre-labels were
    # generated, so the model could not have selected it. Reporting only the "all"
    # figure would understate model agreement; reporting only the subset would flatter it.
    sub = [(h, p) for h, p in pairs if h["intent"] != "other"]

    conf = Counter((a, b) for a, b in zip(hi, mi) if a != b)
    mism = [(h, m) for h, m in zip(ha, ma) if h != m]

    report = {
        "pre_label_model": pairs[0][1]["model_a"]["model"],
        "n_paired": len(pairs),
        "intent_all": agreement(hi, mi),
        "intent_excluding_other": agreement(
            [h["intent"] for h, _ in sub], [p["model_a"]["intent"] for _, p in sub]
        ),
        "action": agreement(ha, ma),
        "action_mismatches": {
            "total": len(mism),
            "human_escalate_model_auto": sum(1 for h, _ in mism if h == "escalate"),
            "human_auto_model_escalate": sum(1 for h, _ in mism if h == "auto_handle"),
        },
        "top_intent_confusions": [
            {"human": h, "model": m, "n": n} for (h, m), n in conf.most_common(10)
        ],
        "note": (
            "Author labels written blind, before any pre-label was inspected. "
            "Pre-labels use a different prompt with no retrieval; these figures "
            "indicate task difficulty, not agent performance."
        ),
    }
    dest = PROOT / cfg["paths"]["results"] / "label_agreement.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
