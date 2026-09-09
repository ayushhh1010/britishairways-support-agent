"""Reconstruct conversation threads from the flat twcs tweet table.

The raw CSV is an edge list: each row carries `in_response_to_tweet_id`. Threads
are the connected components of that graph. We union-find over the edges once,
then keep only components that a given brand participated in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

RAW_DEFAULT = Path("data/raw/twcs/twcs.csv")

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p.get(root, root) != root:
            root = p[root]
        # Path compression, iterative to avoid recursion limits on long chains.
        while p.get(x, x) != x:
            x, p[x] = p[x], root
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class Turn:
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: pd.Timestamp
    text: str


@dataclass
class Thread:
    thread_id: int
    brand: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def first_customer_turn(self) -> Turn | None:
        return next((t for t in self.turns if t.inbound), None)

    def first_brand_reply(self) -> Turn | None:
        """The brand's first reply *after* the first customer turn."""
        seen_customer = False
        for t in self.turns:
            if t.inbound:
                seen_customer = True
            elif seen_customer and t.author_id == self.brand:
                return t
        return None


_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str, *, strip_mentions: bool = True) -> str:
    """Normalise tweet text. Anonymised user ids (@115712) carry no meaning."""
    out = _URL.sub("<url>", text)
    if strip_mentions:
        out = _MENTION.sub("", out)
    out = out.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return _WS.sub(" ", out).strip()


def load_raw(path: Path | str = RAW_DEFAULT, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype={"author_id": "string", "text": "string"},
        nrows=nrows,
    )
    df["text"] = df["text"].fillna("")
    return df


def build_threads(df: pd.DataFrame, brand: str) -> list[Thread]:
    """Return every thread in which `brand` posted at least one tweet."""
    uf = _UnionFind()
    tid = df["tweet_id"].to_numpy()
    parent = df["in_response_to_tweet_id"].to_numpy()

    for a, b in zip(tid, parent):
        if b == b:  # not NaN
            uf.union(int(a), int(b))

    roots = [uf.find(int(t)) for t in tid]
    df = df.assign(_root=roots)

    brand_roots = set(df.loc[df["author_id"] == brand, "_root"].unique())
    sub = df[df["_root"].isin(brand_roots)].copy()
    sub["created_at"] = pd.to_datetime(
        sub["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    sub = sub.sort_values(["_root", "created_at", "tweet_id"])

    threads: list[Thread] = []
    for root, grp in sub.groupby("_root", sort=False):
        turns = [
            Turn(
                tweet_id=int(r.tweet_id),
                author_id=str(r.author_id),
                inbound=bool(r.inbound),
                created_at=r.created_at,
                text=str(r.text),
            )
            for r in grp.itertuples()
        ]
        threads.append(Thread(thread_id=int(root), brand=brand, turns=turns))
    return threads


def iter_cases(threads: Iterable[Thread]) -> Iterator[dict]:
    """Flatten threads into (first customer message -> brand reply) supervision cases."""
    for th in threads:
        cust = th.first_customer_turn()
        rep = th.first_brand_reply()
        if cust is None:
            continue
        yield {
            "thread_id": th.thread_id,
            "brand": th.brand,
            "customer_id": cust.author_id,
            "created_at": cust.created_at.isoformat() if cust.created_at is not None else None,
            "customer_text_raw": cust.text,
            "customer_text": clean_text(cust.text),
            "brand_reply_raw": rep.text if rep else None,
            "brand_reply": clean_text(rep.text) if rep else None,
            "n_turns": th.n_turns,
            "n_customer_turns": sum(1 for t in th.turns if t.inbound),
            "n_brand_turns": sum(1 for t in th.turns if not t.inbound),
        }
