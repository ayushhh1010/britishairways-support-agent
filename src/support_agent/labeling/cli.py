"""Terminal labelling tool for the golden set and for human reply ratings.

Two modes:

  intents  -- label each case with an intent + triage action (+ escalation reason)
  replies  -- rate candidate replies 1-5 on the same rubric the LLM judge uses,
              which is what makes judge-vs-human agreement measurable

Both write JSONL incrementally and resume where you left off, so a session can be
interrupted. `--blind` hides model pre-labels to remove anchoring bias.

Usage
-----
  python -m support_agent.labeling.cli intents --annotator ayush --blind
  python -m support_agent.labeling.cli replies --annotator ayush --limit 50
  python -m support_agent.labeling.cli status
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from ..config import ROOT, load_config
from ..evaluation.judge import DIMENSIONS
from ..taxonomy import ESCALATION_REASON_IDS, load_taxonomy

SEP = "=" * 78


def _labels_path(cfg: dict, annotator: str, mode: str) -> Path:
    d = ROOT / cfg["paths"]["golden"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mode}_{annotator}.jsonl"


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[str(rec.get("key"))] = rec
    return out


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[saved; rerun the same command to resume]")
        sys.exit(0)


# ---------------------------------------------------------------------------
# intents
# ---------------------------------------------------------------------------


def label_intents(args: argparse.Namespace) -> None:
    cfg = load_config()
    tax = load_taxonomy()
    golden = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "golden.parquet")

    prelabels: dict[str, dict] = {}
    pre_path = ROOT / cfg["paths"]["golden"] / "prelabels.jsonl"
    if pre_path.exists() and not args.blind:
        prelabels = {str(r["key"]): r for r in map(json.loads, pre_path.read_text(encoding="utf-8").splitlines()) if r}

    out_path = _labels_path(cfg, args.annotator, "intents")
    done = _load_done(out_path)
    todo = [r for _, r in golden.iterrows() if str(r["thread_id"]) not in done]
    if args.shuffle:
        random.Random(cfg["sampling"]["seed"]).shuffle(todo)
    todo = todo[: args.limit] if args.limit else todo

    names = tax.names
    print(f"\n{len(done)} already labelled, {len(todo)} to go -> {out_path}")
    print("Intents:")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}. {n:<26} {tax.get(n).short}")
    print("Actions: [a]uto_handle  [e]scalate     Commands: s=skip  q=quit\n")

    for n_i, row in enumerate(todo, 1):
        key = str(row["thread_id"])
        print(SEP)
        print(f"[{n_i}/{len(todo)}]  thread {key}")
        print(f"\n  CUSTOMER: {row['customer_text']}")
        if args.show_reply:
            print(f"\n  (BA actually replied: {row['brand_reply']})")
        pre = prelabels.get(key)
        if pre:
            flag = "  <-- model flagged AMBIGUOUS" if pre.get("ambiguous") else ""
            print(f"\n  suggestion: {pre.get('intent')} / {pre.get('action')}{flag}")
            if pre.get("second_choice"):
                print(f"  or maybe:   {pre.get('second_choice')}")

        raw = _prompt("\n  intent # (enter=accept suggestion): ")
        if raw.lower() == "q":
            break
        if raw.lower() == "s":
            continue
        if raw == "" and pre:
            intent = pre["intent"]
        elif raw.isdigit() and 1 <= int(raw) <= len(names):
            intent = names[int(raw) - 1]
        else:
            print("  ! invalid, skipping")
            continue

        araw = _prompt("  action [a/e] (enter=accept suggestion): ").lower()
        if araw.startswith("a"):
            action = "auto_handle"
        elif araw.startswith("e"):
            action = "escalate"
        elif araw == "" and pre:
            action = pre["action"]
        else:
            print("  ! invalid, skipping")
            continue

        reason = None
        if action == "escalate":
            print("  reasons: " + ", ".join(f"{i}={r}" for i, r in enumerate(ESCALATION_REASON_IDS, 1)))
            rraw = _prompt("  reason # (enter=accept suggestion): ")
            if rraw.isdigit() and 1 <= int(rraw) <= len(ESCALATION_REASON_IDS):
                reason = ESCALATION_REASON_IDS[int(rraw) - 1]
            elif pre:
                reason = pre.get("escalation_reason_id")

        note = _prompt("  note (optional): ") or None
        _append(out_path, {
            "key": key, "thread_id": int(row["thread_id"]), "annotator": args.annotator,
            "intent": intent, "action": action, "escalation_reason_id": reason,
            "note": note, "blind": bool(args.blind),
            "customer_text": row["customer_text"],
        })
        print(f"  saved: {intent} / {action}")

    print(f"\nDone. {len(_load_done(out_path))} labels in {out_path}")


# ---------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------


def rate_replies(args: argparse.Namespace) -> None:
    cfg = load_config()
    src = ROOT / cfg["paths"]["results"] / "judge_sample.jsonl"
    if not src.exists():
        print(f"missing {src}\nRun: python scripts/run_eval.py  (it writes the blind rating sample)")
        sys.exit(1)
    jobs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_path = _labels_path(cfg, args.annotator, "replies")
    done = _load_done(out_path)
    todo = [j for j in jobs if str(j["key"]) not in done]
    todo = todo[: args.limit] if args.limit else todo

    print(f"\n{len(done)} already rated, {len(todo)} to go -> {out_path}")
    print("Score each 1-5. The system that wrote the reply is hidden on purpose.\n")
    print("  groundedness: are all factual claims supported by evidence/the message?")
    print("  helpfulness : does it move the customer forward?")
    print("  tone        : does it sound like a good BA agent?")
    print("  safety      : avoids promising money/timescales, no public PII requests?\n")

    for n_i, job in enumerate(todo, 1):
        print(SEP)
        print(f"[{n_i}/{len(todo)}]")
        print(f"\n  CUSTOMER : {job['message']}")
        print(f"\n  CANDIDATE: {job['candidate']}")
        if job.get("evidence_preview"):
            print(f"\n  evidence : {job['evidence_preview']}")
        scores = {}
        bad = False
        for dim in DIMENSIONS:
            # Re-prompt on bad input. Previously a stray Enter ended the whole
            # session, silently discarding the rest of a 40-item run.
            while True:
                raw = _prompt(f"  {dim:<13}[1-5] (q=quit): ")
                if raw.lower() == "q":
                    bad = True
                    break
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    scores[dim] = int(raw)
                    break
                print("    ! enter a number 1-5, or q to quit")
            if bad:
                break
        if bad:
            break
        # Validate explicitly: this used to treat any non-"y" input as "no", so a
        # mistyped "5" was silently recorded as an unacceptable reply.
        while True:
            araw = _prompt("  send as-is to a real customer? [y/n]: ").strip().lower()
            if araw in ("y", "yes", "n", "no"):
                acc = araw.startswith("y")
                break
            print("    ! answer y or n")
        _append(out_path, {
            "key": str(job["key"]), "thread_id": job["thread_id"], "annotator": args.annotator,
            **scores, "acceptable": acc,
        })
        print(f"  saved: mean={sum(scores.values())/len(scores):.2f} acceptable={acc}")

    print(f"\nDone. {len(_load_done(out_path))} ratings in {out_path}")


def status(_: argparse.Namespace) -> None:
    cfg = load_config()
    d = ROOT / cfg["paths"]["golden"]
    if not d.exists():
        print("no golden/ directory yet")
        return
    for p in sorted(d.glob("*.jsonl")):
        # Count lines, not `_load_done` keys: files written by other stages (e.g.
        # golden_set.jsonl) have no `key` field, so keying would collapse every row
        # onto None and report "1 record" for a 220-row file.
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  {p.name:<34} {n:>4} records")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("intents", help="label intents + triage actions")
    a.add_argument("--annotator", required=True)
    a.add_argument("--blind", action="store_true", help="hide model pre-labels")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--shuffle", action="store_true")
    a.add_argument("--show-reply", action="store_true", help="reveal the historical BA reply")
    a.set_defaults(func=label_intents)

    b = sub.add_parser("replies", help="rate candidate replies 1-5")
    b.add_argument("--annotator", required=True)
    b.add_argument("--limit", type=int, default=0)
    b.set_defaults(func=rate_replies)

    c = sub.add_parser("status", help="show label counts")
    c.set_defaults(func=status)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
