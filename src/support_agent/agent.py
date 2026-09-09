"""The support agent: retrieve -> classify -> triage -> draft, in one LLM turn.

One call per ticket rather than three. It is what a real deployment would do on
latency and cost, and it lets the drafted reply be conditioned on the agent's own
triage decision instead of a separately-predicted one. `classify_only` exists as an
ablation so we can measure what retrieval actually contributes.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Iterable

from tqdm import tqdm

from .config import load_config
from .llm import LLMClient, LLMError, QuotaExhausted
from .prompts import (
    AGENT_SYSTEM,
    AGENT_USER,
    CLASSIFY_ONLY_SYSTEM,
    CLASSIFY_ONLY_USER,
    PROMPT_VERSION,
)
from .retrieve import Evidence, HistoricalRetriever, render_evidence
from .taxonomy import ESCALATION_REASON_IDS, Taxonomy, load_taxonomy

MAX_REPLY_CHARS = 280


@dataclass
class AgentOutput:
    thread_id: int
    intent: str
    intent_confidence: float
    action: str
    escalation_reason_id: str | None
    escalation_rationale: str
    reply: str
    evidence_used: list[int] = field(default_factory=list)
    evidence_thread_ids: list[int] = field(default_factory=list)
    top_similarity: float = 0.0
    n_evidence: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _coerce_action(value: object) -> str:
    v = str(value or "").strip().lower()
    if v in ("escalate", "escalation", "human", "handoff"):
        return "escalate"
    if v in ("auto_handle", "auto-handle", "autohandle", "auto", "bot"):
        return "auto_handle"
    return "escalate"  # policy default when the model returns junk


def _coerce_intent(value: object, taxonomy: Taxonomy) -> str:
    v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in taxonomy.names:
        return v
    # Models occasionally return a near-miss like "baggage" or "flight_delay".
    for name in taxonomy.names:
        if v and (v in name or name in v):
            return name
    return "service_complaint"  # broadest bucket; counted as an error if wrong


def _coerce_reason(value: object, action: str) -> str | None:
    if action == "auto_handle":
        return None
    v = str(value or "").strip().lower()
    return v if v in ESCALATION_REASON_IDS else "needs_account_access"


_WS = re.compile(r"\s+")


def _clean_reply(text: object) -> str:
    out = _WS.sub(" ", str(text or "")).strip().strip('"')
    if len(out) > MAX_REPLY_CHARS:
        out = out[: MAX_REPLY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return out


class SupportAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        retriever: HistoricalRetriever | None = None,
        taxonomy: Taxonomy | None = None,
        cfg: dict | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)
        self.retriever = retriever or HistoricalRetriever.from_disk(self.cfg)
        self.taxonomy = taxonomy or load_taxonomy()
        self._reason_ids = ", ".join(ESCALATION_REASON_IDS)

    # ---------------- prompt building ----------------

    def _system(self, template: str) -> str:
        style = self.cfg["llm"].get("taxonomy_style", "full")
        return template.format(
            taxonomy=self.taxonomy.render_for_prompt(style=style),
            triage_policy=self.taxonomy.render_triage_policy(),
            reason_ids=self._reason_ids,
        )

    # ---------------- single case ----------------

    def run_one(self, thread_id: int, message: str, evidence: list[Evidence] | None = None) -> AgentOutput:
        ev = self.retriever.search(message) if evidence is None else evidence
        messages = [
            {"role": "system", "content": f"[prompt {PROMPT_VERSION}]\n" + self._system(AGENT_SYSTEM)},
            {"role": "user", "content": AGENT_USER.format(evidence=render_evidence(ev), message=message)},
        ]
        try:
            raw = self.client.chat_json(messages)
        except QuotaExhausted as exc:
            # Not a prediction failure. Marked so the caller can stop, keep what is
            # cached, and resume when the provider's daily window refills.
            return AgentOutput(
                thread_id=thread_id, intent="service_complaint", intent_confidence=0.0,
                action="escalate", escalation_reason_id="unparseable",
                escalation_rationale="", reply="", n_evidence=len(ev),
                error=f"QUOTA:{exc}"[:200],
            )
        except LLMError as exc:
            return AgentOutput(
                thread_id=thread_id,
                intent="service_complaint",
                intent_confidence=0.0,
                action="escalate",
                escalation_reason_id="unparseable",
                escalation_rationale="model call failed; defaulted to human handoff",
                reply="",
                n_evidence=len(ev),
                error=str(exc)[:200],
            )

        action = _coerce_action(raw.get("action"))
        used = raw.get("evidence_used") or []
        used = [int(u) for u in used if str(u).strip().lstrip("-").isdigit()]
        return AgentOutput(
            thread_id=thread_id,
            intent=_coerce_intent(raw.get("intent"), self.taxonomy),
            intent_confidence=float(raw.get("intent_confidence") or 0.0),
            action=action,
            escalation_reason_id=_coerce_reason(raw.get("escalation_reason_id"), action),
            escalation_rationale=str(raw.get("escalation_rationale") or "").strip()[:300],
            reply=_clean_reply(raw.get("reply")),
            evidence_used=used,
            evidence_thread_ids=[e.thread_id for e in ev],
            top_similarity=ev[0].score if ev else 0.0,
            n_evidence=len(ev),
        )

    def classify_only(self, thread_id: int, message: str) -> AgentOutput:
        """Ablation: no retrieval, no reply drafting."""
        messages = [
            {"role": "system", "content": f"[prompt {PROMPT_VERSION}]\n" + self._system(CLASSIFY_ONLY_SYSTEM)},
            {"role": "user", "content": CLASSIFY_ONLY_USER.format(message=message)},
        ]
        try:
            raw = self.client.chat_json(messages)
        except QuotaExhausted as exc:
            return AgentOutput(
                thread_id=thread_id, intent="service_complaint", intent_confidence=0.0,
                action="escalate", escalation_reason_id="unparseable",
                escalation_rationale="", reply="", error=f"QUOTA:{exc}"[:200],
            )
        except LLMError as exc:
            return AgentOutput(
                thread_id=thread_id, intent="service_complaint", intent_confidence=0.0,
                action="escalate", escalation_reason_id="unparseable",
                escalation_rationale="model call failed", reply="", error=str(exc)[:200],
            )
        action = _coerce_action(raw.get("action"))
        return AgentOutput(
            thread_id=thread_id,
            intent=_coerce_intent(raw.get("intent"), self.taxonomy),
            intent_confidence=float(raw.get("intent_confidence") or 0.0),
            action=action,
            escalation_reason_id=_coerce_reason(raw.get("escalation_reason_id"), action),
            escalation_rationale=str(raw.get("escalation_rationale") or "").strip()[:300],
            reply="",
        )

    # ---------------- batch ----------------

    def run_batch(
        self,
        rows: Iterable[dict],
        *,
        classify_only: bool = False,
        desc: str = "agent",
    ) -> list[AgentOutput]:
        rows = list(rows)
        if not classify_only:
            all_ev = self.retriever.search_batch([r["customer_text"] for r in rows])
        else:
            all_ev = [None] * len(rows)

        workers = int(self.cfg["llm"].get("concurrency", 4))

        def _work(pair):
            row, ev = pair
            if classify_only:
                return self.classify_only(int(row["thread_id"]), row["customer_text"])
            return self.run_one(int(row["thread_id"]), row["customer_text"], ev)

        out: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(_work, zip(rows, all_ev)), total=len(rows), desc=desc):
                out.append(res)
        return out
