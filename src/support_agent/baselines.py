"""Baselines the LLM system has to beat.

B0 trivial  -- majority-class intent, a fixed triage policy, one canned reply.
B1 simple   -- TF-IDF + logistic regression intent, keyword-rule triage, and the
               nearest historical reply copied verbatim.

B1 is deliberately not a strawman. It is given a real advantage the LLM system does
not get: it is *trained on the golden labels* (scored out-of-fold via stratified
5-fold CV), while the LLM system is zero-shot and never sees a golden label. If the
LLM only ties this, that is a genuine result, not a win. See REPORT.md.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .retrieve import HistoricalRetriever

# The single most common real BA holding reply pattern, used as the trivial reply.
CANNED_REPLY = (
    "Sorry to hear this. Please send us a direct message with your booking reference "
    "and flight number and we'll take a look for you."
)

# ---------------------------------------------------------------------------
# B0: trivial
# ---------------------------------------------------------------------------


@dataclass
class TrivialBaseline:
    """Majority intent + always-escalate + one canned reply."""

    majority_intent: str = "flight_disruption"
    triage_policy: str = "escalate"  # "escalate" | "auto_handle" | "majority"

    def fit(self, intents: list[str], actions: list[str]) -> "TrivialBaseline":
        self.majority_intent = Counter(intents).most_common(1)[0][0]
        if self.triage_policy == "majority":
            self.triage_policy = Counter(actions).most_common(1)[0][0]
        return self

    def predict(self, messages: list[str]) -> pd.DataFrame:
        n = len(messages)
        return pd.DataFrame(
            {
                "intent": [self.majority_intent] * n,
                "action": [self.triage_policy] * n,
                "reply": [CANNED_REPLY] * n,
                "escalation_rationale": ["fixed policy: always " + self.triage_policy] * n,
            }
        )


# ---------------------------------------------------------------------------
# B1: simple
# ---------------------------------------------------------------------------

_MONEY = re.compile(
    r"(\brefund|\bcompensat|\breimburs|\bmoney back|\bovercharg|\bcharged twice|"
    r"\bdouble charg|\beu ?261|\bec ?261|\bvoucher|\bclaim|£|\$|€|\bexpenses)",
    re.I,
)
# A 6-char booking reference must contain a digit, otherwise shouty words like
# "THANKS" or "REFUND" match and the rule fires on every angry tweet.
_BOOKING_REF = re.compile(
    r"(\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b"
    r"|\bbooking (ref|reference)\b|\bpnr\b|\bticket number\b|\b\d{8,}\b)"
)
_LOST_PROPERTY = re.compile(
    r"(\blost\b|\bmissing\b|\bdidn'?t arrive|\bnever arrived|\bhasn'?t arrived|"
    r"\bhas not arrived|\bnot arrived|\bdamaged\b|\bleft (my|it|behind)|"
    r"\bstill waiting for my (bag|case|luggage|suitcase))",
    re.I,
)
_URGENT = re.compile(
    r"\b(stranded|stuck|right now|boarding|at the (airport|gate)|today|tonight|"
    r"in an hour|tomorrow morning|asap|urgent)\b",
    re.I,
)
_LEGAL = re.compile(r"\b(solicitor|lawyer|legal|sue|ombudsman|caa|small claims|press|bbc)\b", re.I)
_DISTRESS = re.compile(r"\b(disgust|appall|furious|outrage|never fly|worst|unacceptable|shame)\b", re.I)


def rule_triage(message: str) -> tuple[str, str, str]:
    """Keyword escalation rules. Returns (action, reason_id, rationale)."""
    checks = [
        (_MONEY, "money_at_stake", "mentions money, a refund, or a compensation claim"),
        (_LEGAL, "legal_safety_medical", "mentions legal or regulatory escalation"),
        (_LOST_PROPERTY, "property_irregularity", "reports lost, missing, or damaged property"),
        (_BOOKING_REF, "needs_account_access", "contains a booking or ticket identifier"),
        (_URGENT, "time_critical", "appears to be travelling imminently"),
        (_DISTRESS, "high_distress", "shows strong dissatisfaction"),
    ]
    for pattern, reason, why in checks:
        if pattern.search(message):
            return "escalate", reason, why
    return "auto_handle", "", "no escalation keyword matched"


class SimpleBaseline:
    """TF-IDF + logistic regression intent, rule triage, nearest-neighbour reply."""

    def __init__(self, retriever: HistoricalRetriever, seed: int = 20260909) -> None:
        self.retriever = retriever
        self.seed = seed
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, C=4.0, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )

    def cv_predict_intents(self, messages: list[str], intents: list[str], n_splits: int = 5) -> list[str]:
        """Out-of-fold predictions, so nothing is scored on its own training data.

        Classes with fewer members than n_splits cannot be stratified; they are
        assigned the global majority label rather than being silently dropped.
        """
        X = np.asarray(messages, dtype=object)
        y = np.asarray(intents, dtype=object)
        counts = Counter(y)
        rare = {c for c, n in counts.items() if n < n_splits}
        mask = np.array([c not in rare for c in y])
        preds = np.array([Counter(y).most_common(1)[0][0]] * len(y), dtype=object)

        Xf, yf = X[mask], y[mask]
        # Nothing is stratifiable (tiny set, or every class rarer than n_splits).
        # Fall back to the majority label rather than crashing the whole sweep.
        if len(yf) == 0 or len(set(yf)) < 2:
            return preds.tolist()

        n_splits = min(n_splits, int(min(Counter(yf).values())))
        if n_splits < 2:
            return preds.tolist()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        idx_map = np.where(mask)[0]
        for train_idx, test_idx in skf.split(Xf, yf):
            self.pipeline.fit(Xf[train_idx], yf[train_idx])
            preds[idx_map[test_idx]] = self.pipeline.predict(Xf[test_idx])
        return preds.tolist()

    def nearest_replies(self, messages: list[str]) -> list[str]:
        hits = self.retriever.search_batch(messages, k=1)
        return [h[0].brand_reply if h else CANNED_REPLY for h in hits]

    def predict(self, messages: list[str], intents: list[str]) -> pd.DataFrame:
        triage = [rule_triage(m) for m in messages]
        return pd.DataFrame(
            {
                "intent": self.cv_predict_intents(messages, intents),
                "action": [t[0] for t in triage],
                "escalation_reason_id": [t[1] or None for t in triage],
                "escalation_rationale": [t[2] for t in triage],
                "reply": self.nearest_replies(messages),
            }
        )
