"""Dual-model pre-labelling of the golden set.

Protocol (DECISIONS.md #10):
  1. Two DIFFERENT model families independently propose intent + triage action.
  2. Where they agree and neither flags ambiguity -> a consensus pre-label.
  3. Where they disagree, or either flags ambiguity -> marked `needs_adjudication`.
  4. A human adjudicates. Pre-labels are NEVER ground truth on their own.

Model-model agreement is also reported: it is a useful upper-ish bound on how
well-defined the taxonomy is. If two strong models cannot agree on a boundary, a
human annotator will not be consistent on it either, and the class needs redefining.

Requests are BATCHED (default 10 messages per call). The taxonomy + policy block is
~1,300 tokens and identical every time; sending it once per message burned ~90% of
the provider's daily token budget on repeated boilerplate. Batching is used only
here, never for the agent under evaluation -- see DECISIONS.md #21.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.metrics import agreement  # noqa: E402
from support_agent.llm import LLMClient, LLMError, QuotaExhausted  # noqa: E402
from support_agent.prompts import (  # noqa: E402
    PRELABEL_BATCH_SYSTEM,
    PRELABEL_BATCH_USER,
    PROMPT_VERSION,
)
from support_agent.taxonomy import ESCALATION_REASON_IDS, load_taxonomy  # noqa: E402

EMPTY = {
    "intent": None, "second_choice": None, "ambiguous": True,
    "action": None, "escalation_reason_id": None, "rationale": "",
}


def _normalise(raw: dict, tax) -> dict:
    intent = str(raw.get("intent") or "").strip().lower()
    action = str(raw.get("action") or "").strip().lower()
    reason = raw.get("escalation_reason_id")
    reason = str(reason).strip().lower() if reason else None
    return {
        "intent": intent if intent in tax.names else None,
        "second_choice": raw.get("second_choice"),
        "ambiguous": bool(raw.get("ambiguous")),
        "action": action if action in ("auto_handle", "escalate") else None,
        "escalation_reason_id": reason if reason in ESCALATION_REASON_IDS else None,
        "rationale": str(raw.get("rationale") or "")[:200],
    }


def propose_batch(client: LLMClient, model: str, rows: list[dict], tax, style: str) -> list[dict]:
    """Label a batch. On a malformed response, returns EMPTY for the whole batch;
    those cases then route to mandatory human adjudication, the safe default."""
    system = PRELABEL_BATCH_SYSTEM.format(
        taxonomy=tax.render_for_prompt(style=style),
        triage_policy=tax.render_triage_policy(),
        reason_ids=", ".join(ESCALATION_REASON_IDS),
    )
    block = "\n".join(f'{i + 1}. """{r["customer_text"]}"""' for i, r in enumerate(rows))
    msgs = [
        {"role": "system", "content": f"[prompt {PROMPT_VERSION}]\n" + system},
        {"role": "user", "content": PRELABEL_BATCH_USER.format(block=block, n=len(rows))},
    ]
    try:
        raw = client.chat_json(msgs, model=model, max_tokens=85 * len(rows) + 150)
    except QuotaExhausted:
        raise
    except LLMError:
        return [dict(EMPTY) for _ in rows]

    items = raw.get("items")
    if not isinstance(items, list):
        return [dict(EMPTY) for _ in rows]

    # Map by declared id where possible; fall back to positional order.
    by_id: dict[int, dict] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            by_id[int(it.get("id"))] = it
        except (TypeError, ValueError):
            continue

    out = []
    for i in range(len(rows)):
        it = by_id.get(i + 1)
        if it is None and len(items) == len(rows) and isinstance(items[i], dict):
            it = items[i]
        out.append(_normalise(it, tax) if it else dict(EMPTY))
    return out


def batch_size_for(client, model: str, requested: int) -> int:
    """Batch size a model can actually answer in one response.

    Providers reject a request whose max_tokens exceeds their output-per-minute cap,
    so the batch has to fit inside the model's output ceiling, not just its context.
    ~85 tokens per item plus JSON wrapper, with headroom.
    """
    ceiling = int(client.limits_for(model)["max_output"])
    fits = max(3, (ceiling - 150) // 85)
    return max(1, min(requested, fits))


def run_model(client, model, rows, tax, style, batch_size, workers, desc) -> list[dict]:
    batch_size = batch_size_for(client, model, batch_size)
    desc = f"{desc} b={batch_size}"
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(
            pool.map(lambda b: propose_batch(client, model, b, tax, style), batches),
            total=len(batches), desc=desc,
        ):
            results.extend(res)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    tax = load_taxonomy()
    client = LLMClient(cfg)
    golden = pd.read_parquet(PROOT / cfg["paths"]["processed"] / "golden.parquet")

    model_a = cfg["llm"]["prelabel_model_a"]
    model_b = cfg["llm"]["prelabel_model_b"]
    style = cfg["llm"].get("taxonomy_style", "compact")
    workers = int(cfg["llm"].get("concurrency", 3))
    rows = golden.to_dict("records")

    # A and B hit different models, hence different provider quota pools, so
    # running them together roughly halves wall-clock time.
    with ThreadPoolExecutor(max_workers=2) as outer:
        fa = outer.submit(run_model, client, model_a, rows, tax, style,
                          args.batch_size, workers, f"A({model_a.split('/')[-1]})")
        fb = outer.submit(run_model, client, model_b, rows, tax, style,
                          args.batch_size, workers, f"B({model_b.split('/')[-1]})")
        a, b = fa.result(), fb.result()

    out_path = PROOT / cfg["paths"]["golden"] / "prelabels.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row, ra, rb in zip(rows, a, b):
        intent_agree = ra["intent"] is not None and ra["intent"] == rb["intent"]
        action_agree = ra["action"] is not None and ra["action"] == rb["action"]
        needs_adj = (
            not intent_agree or not action_agree
            or bool(ra.get("ambiguous")) or bool(rb.get("ambiguous"))
        )
        records.append({
            "key": str(row["thread_id"]),
            "thread_id": int(row["thread_id"]),
            "customer_text": row["customer_text"],
            "brand_reply": row["brand_reply"],
            "intent": ra["intent"] or rb["intent"],
            "action": ra["action"] or rb["action"],
            "escalation_reason_id": ra.get("escalation_reason_id") or rb.get("escalation_reason_id"),
            "second_choice": ra.get("second_choice"),
            "ambiguous": bool(ra.get("ambiguous")) or bool(rb.get("ambiguous")),
            "model_a": {"model": model_a, **ra},
            "model_b": {"model": model_b, **rb},
            "intent_agree": intent_agree,
            "action_agree": action_agree,
            "needs_adjudication": needs_adj,
        })

    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    both_i = [r for r in records if r["model_a"]["intent"] and r["model_b"]["intent"]]
    both_a = [r for r in records if r["model_a"]["action"] and r["model_b"]["action"]]
    stats = {
        "n": len(records),
        "model_a": model_a,
        "model_b": model_b,
        "batch_size": args.batch_size,
        "n_with_both_intents": len(both_i),
        "intent_agreement": agreement(
            [r["model_a"]["intent"] for r in both_i], [r["model_b"]["intent"] for r in both_i]
        ) if both_i else None,
        "action_agreement": agreement(
            [r["model_a"]["action"] for r in both_a], [r["model_b"]["action"] for r in both_a]
        ) if both_a else None,
        "needs_adjudication": sum(r["needs_adjudication"] for r in records),
        "flagged_ambiguous": sum(r["ambiguous"] for r in records),
        "usage": client.usage.as_dict(),
    }
    (PROOT / cfg["paths"]["results"] / "prelabel_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
