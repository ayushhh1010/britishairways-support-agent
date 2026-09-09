"""Automated metrics.

Every headline number ships with a bootstrap CI. With n=220 and 12 classes, a bare
point estimate hides roughly +/-6 accuracy points of sampling noise, and quoting one
without the interval is the single easiest way to mislead a reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

BOOTSTRAP_N = 2000
SEED = 20260909


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn,
    *,
    n: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap over cases. Resamples pairs, not predictions."""
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    m = len(yt)
    if m == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        try:
            stats[i] = metric_fn(yt[idx], yp[idx])
        except ValueError:
            stats[i] = np.nan
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class IntentMetrics:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    macro_f1: float
    macro_f1_ci: tuple[float, float]
    weighted_f1: float
    per_class: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)
    confusion: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci95": [round(c, 4) for c in self.accuracy_ci],
            "macro_f1": round(self.macro_f1, 4),
            "macro_f1_ci95": [round(c, 4) for c in self.macro_f1_ci],
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": self.per_class,
            "labels": self.labels,
            "confusion": self.confusion,
        }


def _macro_f1(a, b) -> float:
    return f1_score(a, b, average="macro", zero_division=0)


def score_intents(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> IntentMetrics:
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        k: {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for k, v in rep.items()
        if k in labels
    }
    return IntentMetrics(
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        accuracy_ci=bootstrap_ci(y_true, y_pred, accuracy_score),
        macro_f1=_macro_f1(y_true, y_pred),
        macro_f1_ci=bootstrap_ci(y_true, y_pred, _macro_f1),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        per_class=per_class,
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


@dataclass
class TriageMetrics:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    escalate_precision: float
    escalate_recall: float
    escalate_f1: float
    false_auto_handle: int
    false_escalate: int
    weighted_cost: float
    cost_per_case: float
    base_rate_escalate: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci95": [round(c, 4) for c in self.accuracy_ci],
            "escalate_precision": round(self.escalate_precision, 4),
            "escalate_recall": round(self.escalate_recall, 4),
            "escalate_f1": round(self.escalate_f1, 4),
            "false_auto_handle": self.false_auto_handle,
            "false_escalate": self.false_escalate,
            "weighted_cost": round(self.weighted_cost, 2),
            "cost_per_case": round(self.cost_per_case, 4),
            "base_rate_escalate": round(self.base_rate_escalate, 4),
        }


def score_triage(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    false_auto_handle_cost: float = 3.0,
    false_escalate_cost: float = 1.0,
) -> TriageMetrics:
    """`escalate` is the positive class.

    The two errors are not symmetric. Escalating something the bot could have
    answered wastes an agent's minute. Auto-handling something that needed a human
    reaches the customer with a wrong or harmful answer, so it is weighted 3x.
    """
    yt = np.asarray([str(v) for v in y_true])
    yp = np.asarray([str(v) for v in y_pred])
    p, r, f, _ = precision_recall_fscore_support(
        yt, yp, labels=["escalate"], average="binary", pos_label="escalate", zero_division=0
    )
    false_auto = int(np.sum((yt == "escalate") & (yp == "auto_handle")))
    false_esc = int(np.sum((yt == "auto_handle") & (yp == "escalate")))
    cost = false_auto * false_auto_handle_cost + false_esc * false_escalate_cost
    return TriageMetrics(
        n=len(yt),
        accuracy=accuracy_score(yt, yp),
        accuracy_ci=bootstrap_ci(yt, yp, accuracy_score),
        escalate_precision=float(p),
        escalate_recall=float(r),
        escalate_f1=float(f),
        false_auto_handle=false_auto,
        false_escalate=false_esc,
        weighted_cost=float(cost),
        cost_per_case=float(cost / max(len(yt), 1)),
        base_rate_escalate=float(np.mean(yt == "escalate")),
    )


_DEFLECTION = re.compile(
    r"(\bDM\b|direct message|private message|\bPM us\b|send us (a|your)|drop us a"
    r"|get in touch|contact us|reach out to us|let us know (your|the) (booking|ref)"
    r"|share (your|the) (booking|details|reference)|we'?ll look into|colleague will"
    r"|pass (this|it) (on|to))",
    re.I,
)


def answer_rate(replies: Sequence[str], actions: Sequence[str]) -> dict:
    """Of the cases a system chose to AUTO-HANDLE, how many actually answered?

    This exists because the judge rubric can be gamed by a safe non-answer. A canned
    "sorry, please DM us" scores well on groundedness and safety -- there is nothing
    in it to be wrong about -- so a system that deflects everything can post a
    respectable mean judge score while resolving nothing. Answer rate is a
    deliberately dumb, non-LLM check on that failure, and it is reported next to the
    judge scores rather than instead of them.

    A deflection is only counted against a reply the system said it would AUTO-HANDLE.
    Routing to DM is the correct behaviour when the system escalated.
    """
    auto = [(r or "") for r, a in zip(replies, actions) if a == "auto_handle"]
    if not auto:
        return {"n_auto_handled": 0, "answer_rate": float("nan"), "deflection_rate": float("nan")}
    deflected = sum(1 for r in auto if _DEFLECTION.search(r))
    return {
        "n_auto_handled": len(auto),
        "deflection_rate": round(deflected / len(auto), 4),
        "answer_rate": round(1 - deflected / len(auto), 4),
    }


def agreement(a: Sequence, b: Sequence) -> dict:
    """Cohen's kappa + raw agreement between two annotators/raters."""
    a = [str(x) for x in a]
    b = [str(x) for x in b]
    raw = float(np.mean([x == y for x, y in zip(a, b)]))
    try:
        kappa = float(cohen_kappa_score(a, b))
    except ValueError:
        kappa = float("nan")
    lo, hi = bootstrap_ci(a, b, lambda p, q: float(np.mean(p == q)))
    return {
        "n": len(a),
        "raw_agreement": round(raw, 4),
        "raw_agreement_ci95": [round(lo, 4), round(hi, 4)],
        "cohens_kappa": round(kappa, 4),
        "kappa_interpretation": interpret_kappa(kappa),
    }


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands. Conventional, and coarse -- quoted as a label only."""
    if np.isnan(k):
        return "undefined"
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """For ordinal 1-5 judge scores: penalises distant disagreements more."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if len(a) == 0:
        return float("nan")
    try:
        return float(cohen_kappa_score(a, b, weights="quadratic"))
    except ValueError:
        return float("nan")
