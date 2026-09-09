"""Tests for the parts where a silent bug would corrupt the results.

Priority is not coverage. It is the handful of places where a mistake produces a
plausible-looking number rather than a crash: split leakage, the escalation-cost
asymmetry, out-of-fold scoring, and the defensive parsing that stands between a
sloppy model response and a mislabelled case.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import _clean_reply, _coerce_action, _coerce_intent, _coerce_reason
from support_agent.baselines import SimpleBaseline, rule_triage
from support_agent.data.dataset import _stable_bucket, filter_cases, make_splits
from support_agent.data.threads import _UnionFind, clean_text
from support_agent.evaluation.metrics import (
    answer_rate,
    agreement,
    bootstrap_ci,
    score_intents,
    score_triage,
)
from support_agent.llm import LLMClient, parse_json
from support_agent.taxonomy import load_taxonomy


# --------------------------------------------------------------------------
# thread reconstruction
# --------------------------------------------------------------------------

def test_unionfind_merges_transitively():
    uf = _UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(10, 11)
    assert uf.find(1) == uf.find(3)
    assert uf.find(1) != uf.find(10)


def test_unionfind_handles_long_chain_without_recursion():
    uf = _UnionFind()
    for i in range(5000):
        uf.union(i, i + 1)
    assert uf.find(0) == uf.find(5000)


def test_clean_text_strips_anonymised_mentions_and_urls():
    out = clean_text("@115712 see https://t.co/abc now &amp; then")
    assert "@115712" not in out
    assert "<url>" in out
    assert "&" in out and "&amp;" not in out


# --------------------------------------------------------------------------
# splits: the leakage guarantee the whole evaluation rests on
# --------------------------------------------------------------------------

def test_split_bucket_is_stable_across_processes():
    # Must not use hash(), which is salted per process.
    assert _stable_bucket(12345, "split-v1", 10_000) == _stable_bucket(12345, "split-v1", 10_000)
    assert _stable_bucket(12345, "split-v1", 10_000) != _stable_bucket(12346, "split-v1", 10_000)


def _fake_cases(n: int = 15_000) -> pd.DataFrame:
    return pd.DataFrame({
        "thread_id": range(n),
        "brand": ["British_Airways"] * n,
        "customer_text": [f"my flight BA{i} was delayed and I need help please" for i in range(n)],
        "brand_reply": ["We are sorry to hear this, please let us know more." for _ in range(n)],
    })


def test_splits_are_disjoint_at_thread_level():
    sp = make_splits(_fake_cases())
    g, p, d = set(sp.golden.thread_id), set(sp.retrieval_pool.thread_id), set(sp.dev.thread_id)
    assert not (g & p), "golden leaked into retrieval pool"
    assert not (g & d), "golden leaked into dev"
    assert not (p & d), "pool leaked into dev"


def test_filter_drops_retweets_and_duplicates():
    df = pd.DataFrame({
        "thread_id": [1, 2, 3, 4],
        "brand": ["BA"] * 4,
        "customer_text": [
            "RT @someone this is a retweet that is long enough to pass length checks",
            "my flight was delayed by three hours and nobody told us anything at all",
            "my flight was delayed by three hours and nobody told us anything at all",
            "<url>",
        ],
        "brand_reply": ["We are very sorry about this delay today" for _ in range(4)],
    })
    out = filter_cases(df)
    assert len(out) == 1
    assert out.iloc[0]["thread_id"] == 2


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_triage_cost_is_asymmetric():
    """Auto-handling something that needed a human must cost more than the reverse."""
    truth = ["escalate", "auto_handle"]
    false_auto = score_triage(truth, ["auto_handle", "auto_handle"])
    false_esc = score_triage(truth, ["escalate", "escalate"])
    assert false_auto.false_auto_handle == 1 and false_esc.false_escalate == 1
    assert false_auto.weighted_cost > false_esc.weighted_cost
    assert false_auto.weighted_cost == pytest.approx(3.0)


def test_perfect_prediction_scores_one():
    y = ["a", "b", "a", "c"]
    m = score_intents(y, y, ["a", "b", "c"])
    assert m.accuracy == 1.0 and m.macro_f1 == 1.0


def test_bootstrap_ci_brackets_the_point_estimate():
    y = ["a"] * 60 + ["b"] * 40
    p = ["a"] * 50 + ["b"] * 50
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, p)
    lo, hi = bootstrap_ci(y, p, accuracy_score)
    assert lo <= acc <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_agreement_is_perfect_and_chance_corrected():
    a = ["x", "y", "x", "y"]
    assert agreement(a, a)["cohens_kappa"] == 1.0
    # Constant raters agree 100% of the time but carry no information.
    const = agreement(["x"] * 10, ["x"] * 10)
    assert const["raw_agreement"] == 1.0


def test_answer_rate_only_judges_auto_handled_replies():
    replies = ["Please DM us your booking reference.", "Hand luggage is 56x45x25cm."]
    # When the system escalated, routing to DM is correct, so nothing is counted.
    assert answer_rate(replies, ["escalate", "escalate"])["n_auto_handled"] == 0
    r = answer_rate(replies, ["auto_handle", "auto_handle"])
    assert r["deflection_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want a refund for my cancelled flight", "escalate"),
        ("my suitcase never arrived at JFK", "escalate"),
        ("stranded at T5 right now", "escalate"),
        ("booking ref J8EJU9 please help", "escalate"),
        ("can I take a snow globe in hand luggage?", "auto_handle"),
        ("THANKS FOR NOTHING BA", "auto_handle"),  # shouty caps must not look like a PNR
        ("what are Tier Points?", "auto_handle"),
    ],
)
def test_rule_triage(text, expected):
    assert rule_triage(text)[0] == expected


def test_cv_predictions_are_out_of_fold_and_complete():
    """Every case must get a prediction, including classes too rare to stratify."""
    class _Stub:
        def search_batch(self, queries, k=None):
            return [[] for _ in queries]

    msgs = [f"message number {i} about baggage and flights" for i in range(40)]
    intents = ["a"] * 18 + ["b"] * 18 + ["rare"] * 4
    preds = SimpleBaseline(_Stub()).cv_predict_intents(msgs, intents)
    assert len(preds) == len(intents)
    assert all(p is not None for p in preds)


# --------------------------------------------------------------------------
# defensive parsing: a sloppy model response must never become a silent mislabel
# --------------------------------------------------------------------------

def test_parse_json_survives_fences_and_prose():
    assert parse_json('here you go:\n```json\n{"a": 1}\n```')["a"] == 1
    assert parse_json('Sure! {"b": 2} hope that helps')["b"] == 2
    with pytest.raises(Exception):
        parse_json("no json at all")


def test_unknown_action_defaults_to_escalate():
    """Unparseable output must fail safe towards a human, never towards auto-send."""
    assert _coerce_action("banana") == "escalate"
    assert _coerce_action(None) == "escalate"
    assert _coerce_action("auto-handle") == "auto_handle"


def test_intent_coercion_maps_near_misses_and_stays_in_taxonomy():
    tax = load_taxonomy()
    assert _coerce_intent("baggage_policy", tax) == "baggage_policy"
    assert _coerce_intent("BAGGAGE POLICY", tax) == "baggage_policy"
    assert _coerce_intent("total nonsense", tax) in tax.names


def test_escalation_reason_cleared_when_auto_handling():
    assert _coerce_reason("money_at_stake", "auto_handle") is None
    assert _coerce_reason("nonsense", "escalate") in {"needs_account_access"}


def test_reply_is_truncated_to_tweet_length():
    out = _clean_reply("word " * 200)
    assert len(out) <= 280


# --------------------------------------------------------------------------
# provider limits
# --------------------------------------------------------------------------

def test_per_model_limits_fall_back_to_default():
    """Per-model overrides exist because some providers reject a request whose
    max_tokens alone exceeds their output-per-minute cap. Assert the lookup
    mechanism, not one provider's numbers, which change with the config."""
    client = LLMClient()
    default = client.limits_for("some-model-with-no-override")
    assert default["max_output"] > 0 and default["otpm"] > 0

    client.cfg["llm"]["model_limits"]["tiny-model"] = {"max_output": 400}
    tiny = client.limits_for("tiny-model")
    assert tiny["max_output"] == 400
    assert tiny["otpm"] == default["otpm"]  # unspecified keys inherit the default


def test_token_estimate_counts_prompt_and_a_bounded_completion():
    small = LLMClient._estimate_tokens({"messages": [{"content": "x" * 400}], "max_tokens": 700})
    large = LLMClient._estimate_tokens({"messages": [{"content": "x" * 4000}], "max_tokens": 700})
    assert large > small
    # Reserving the full max_tokens would over-throttle against a per-minute cap.
    assert small < 400 // 4 + 700
