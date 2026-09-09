"""Retrieval over historical resolved cases.

TF-IDF + cosine, not embeddings. Rationale (DECISIONS.md #9): the corpus is 12k
short, jargon-dense tweets ("BA2553", "Avios", "T5", "LHR"). Exact lexical overlap
on flight codes and product nouns is precisely the signal we want, an embedding API
is another dependency and cost, and the retriever is measured on downstream reply
groundedness rather than on retrieval alone.

The index is built ONLY from `retrieval_pool`, which is disjoint from `golden`, so
a golden case can never retrieve the reply it is being scored against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import ROOT, load_config


@dataclass(frozen=True)
class Evidence:
    thread_id: int
    customer_text: str
    brand_reply: str
    score: float

    def render(self, idx: int) -> str:
        return (
            f"[{idx}] (similarity {self.score:.2f})\n"
            f"    past customer: {self.customer_text}\n"
            f"    BA replied:    {self.brand_reply}"
        )


class HistoricalRetriever:
    def __init__(self, pool: pd.DataFrame, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        rc = self.cfg["retrieval"]
        self.k = int(rc["k"])
        self.min_similarity = float(rc["min_similarity"])
        self.pool = pool.reset_index(drop=True)

        vc = rc["vectorizer"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(vc["ngram_range"]),
            min_df=int(vc["min_df"]),
            max_features=int(vc["max_features"]),
            sublinear_tf=bool(vc["sublinear_tf"]),
            strip_accents="unicode",
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.pool["customer_text"])

    @classmethod
    def from_disk(cls, cfg: dict | None = None) -> "HistoricalRetriever":
        cfg = cfg or load_config()
        path = ROOT / cfg["paths"]["processed"] / "retrieval_pool.parquet"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} missing. Run: python -m support_agent.data.dataset"
            )
        return cls(pd.read_parquet(path), cfg)

    def search(self, query: str, k: int | None = None) -> list[Evidence]:
        k = k or self.k
        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.matrix).ravel()
        if not sims.size:
            return []
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        out: list[Evidence] = []
        for i in top:
            score = float(sims[i])
            if score < self.min_similarity:
                continue
            row = self.pool.iloc[i]
            out.append(
                Evidence(
                    thread_id=int(row["thread_id"]),
                    customer_text=str(row["customer_text"]),
                    brand_reply=str(row["brand_reply"]),
                    score=score,
                )
            )
        return out

    def search_batch(self, queries: list[str], k: int | None = None) -> list[list[Evidence]]:
        """Vectorised search. Much faster than looping `search` over a whole eval set."""
        k = k or self.k
        qv = self.vectorizer.transform(queries)
        sims = linear_kernel(qv, self.matrix)
        results: list[list[Evidence]] = []
        for row_sims in sims:
            kk = min(k, row_sims.shape[0])
            top = np.argpartition(-row_sims, kk - 1)[:kk]
            top = top[np.argsort(-row_sims[top])]
            hits: list[Evidence] = []
            for i in top:
                score = float(row_sims[i])
                if score < self.min_similarity:
                    continue
                r = self.pool.iloc[i]
                hits.append(
                    Evidence(
                        thread_id=int(r["thread_id"]),
                        customer_text=str(r["customer_text"]),
                        brand_reply=str(r["brand_reply"]),
                        score=score,
                    )
                )
            results.append(hits)
        return results


def render_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no similar historical cases found)"
    return "\n".join(e.render(i + 1) for i, e in enumerate(evidence))
