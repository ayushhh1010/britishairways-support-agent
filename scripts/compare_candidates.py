"""Compare finalist brands on conversation structure, not just reply text."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.data.threads import build_threads, clean_text, load_raw  # noqa: E402

CANDIDATES = ["Delta", "AmericanAir", "British_Airways", "AmazonHelp", "SpotifyCares", "SouthwestAir"]

DEFLECTION = re.compile(
    r"\b(dm|direct message|private message|send us a (dm|message|private)|click .{0,15}message)\b", re.I
)
THANKS = re.compile(r"\b(thanks|thank you|thx|ty|appreciate|cheers|sorted|resolved|worked|fixed|great)\b", re.I)
LINK = re.compile(r"https?://\S+")

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("loading raw csv...", file=sys.stderr)
    df = load_raw(ROOT / "data" / "raw" / "twcs" / "twcs.csv")
    print(f"  {len(df):,} rows", file=sys.stderr)

    rows = []
    for brand in CANDIDATES:
        print(f"threading {brand}...", file=sys.stderr)
        threads = build_threads(df, brand)
        n_threads = len(threads)
        if not n_threads:
            continue
        turns = [t.n_turns for t in threads]
        multi = sum(1 for t in turns if t >= 3)

        substantive = 0
        deflect = 0
        with_reply = 0
        thanked = 0
        reply_lens = []
        for th in threads:
            rep = th.first_brand_reply()
            if rep is None:
                continue
            with_reply += 1
            body = clean_text(rep.text)
            reply_lens.append(len(body))
            if DEFLECTION.search(body):
                deflect += 1
            else:
                # substantive = has content beyond a bare link/apology
                stripped = LINK.sub("", body)
                if len(stripped.split()) >= 12:
                    substantive += 1
            # customer expresses satisfaction in a later turn
            later = [t for t in th.turns if t.inbound and t.created_at is not None
                     and rep.created_at is not None and t.created_at > rep.created_at]
            if any(THANKS.search(t.text) for t in later):
                thanked += 1

        rows.append({
            "brand": brand,
            "threads": n_threads,
            "threads_with_reply": with_reply,
            "median_turns": float(pd.Series(turns).median()),
            "multiturn_rate": round(multi / n_threads, 3),
            "deflection_rate": round(deflect / with_reply, 3),
            "substantive_reply_rate": round(substantive / with_reply, 3),
            "customer_thanks_rate": round(thanked / with_reply, 3),
            "median_reply_chars": float(pd.Series(reply_lens).median()),
        })
        del threads

    out = pd.DataFrame(rows).sort_values("substantive_reply_rate", ascending=False)
    out.to_csv(ROOT / "reports" / "results" / "candidate_comparison.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
