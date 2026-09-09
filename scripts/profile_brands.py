"""Profile brands in the twcs corpus to choose one with evidence, not vibes.

Emits reports/results/brand_profile.csv + a ranked shortlist.
Streams the 2.8M-row CSV in chunks so it runs in <2GB RAM.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
OUT = ROOT / "reports" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# A brand reply that only pushes the user to a private channel carries no
# resolution content -- it is unusable as grounding evidence for a drafted reply.
DEFLECTION = re.compile(
    r"\b(dm|d\.m\.|direct message|private message|pm us|send us a (dm|message|private)|"
    r"click .{0,15}message|shoot us a|slide into)\b",
    re.I,
)
LINK = re.compile(r"https?://\S+")
THANKS = re.compile(r"\b(thanks|thank you|thx|ty|appreciate it|cheers|sorted|resolved|worked|fixed)\b", re.I)

CHUNK = 400_000


def main() -> None:
    inbound_to = Counter()          # brand -> inbound customer tweets addressed to it
    outbound = Counter()            # brand -> outbound brand tweets
    deflect = Counter()             # brand -> outbound tweets that are pure deflection
    has_link = Counter()
    reply_chars = defaultdict(int)
    trigrams = defaultdict(Counter)  # brand -> trigram counter (capped)
    signed = Counter()               # brand -> replies ending in an agent signature (-KC)

    sig_re = re.compile(r"[-–—~]\s?[A-Z]{1,3}[a-z]?\.?\s*$")
    mention_re = re.compile(r"@(\w+)")

    reader = pd.read_csv(
        RAW,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={"author_id": "string", "text": "string"},
        chunksize=CHUNK,
    )
    n = 0
    for chunk in reader:
        n += len(chunk)
        chunk["text"] = chunk["text"].fillna("")
        # Brand accounts are the non-numeric author_ids on outbound tweets.
        out = chunk[chunk["inbound"] == False]  # noqa: E712
        inb = chunk[chunk["inbound"] == True]   # noqa: E712

        for brand, txt in zip(out["author_id"], out["text"]):
            outbound[brand] += 1
            reply_chars[brand] += len(txt)
            if DEFLECTION.search(txt):
                deflect[brand] += 1
            if LINK.search(txt):
                has_link[brand] += 1
            if sig_re.search(txt.strip()):
                signed[brand] += 1
            if len(trigrams[brand]) < 200_000:
                toks = re.findall(r"[a-z']+", txt.lower())
                for i in range(len(toks) - 2):
                    trigrams[brand][(toks[i], toks[i + 1], toks[i + 2])] += 1

        # Inbound tweets name their brand via @mention.
        for txt in inb["text"]:
            m = mention_re.match(txt.strip())
            if m:
                inbound_to[m.group(1)] += 1
        print(f"  ...{n:,} rows", file=sys.stderr, flush=True)

    rows = []
    for brand, n_out in outbound.most_common(40):
        tg = trigrams[brand]
        total_tg = sum(tg.values())
        # Repetition ratio: share of trigram mass held by the top 100 trigrams.
        top100 = sum(c for _, c in tg.most_common(100))
        rows.append(
            {
                "brand": brand,
                "outbound_replies": n_out,
                "inbound_mentions": inbound_to.get(brand, 0),
                "deflection_rate": round(deflect[brand] / n_out, 4),
                "link_rate": round(has_link[brand] / n_out, 4),
                "signature_rate": round(signed[brand] / n_out, 4),
                "avg_reply_chars": round(reply_chars[brand] / n_out, 1),
                "trigram_repetition": round(top100 / total_tg, 4) if total_tg else 0.0,
                "distinct_trigrams": len(tg),
            }
        )

    df = pd.DataFrame(rows).sort_values("outbound_replies", ascending=False)
    df.to_csv(OUT / "brand_profile.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
