"""The support agent: retrieve -> classify -> triage -> draft, in one LLM turn.

One call per ticket rather than three. It is what a real deployment would do on
latency and cost, and it lets the drafted reply be conditioned on the agent's own
triage decision instead of a separately-predicted one. `classify_only` exists as an
ablation so we can measure what retrieval actually contributes.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Iterable

from tqdm import tqdm

from .config import load_config
from .llm import LLMClient, LLMError, QuotaExhausted, parse_json, salvage_items
from .prompts import (
    AGENT_BATCH_SYSTEM,
    AGENT_BATCH_TICKET,
    AGENT_BATCH_USER,
    CLASSIFY_BATCH_SYSTEM,
    CLASSIFY_BATCH_TICKET,
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

    # ---------------- grouped (N tickets per request) ----------------

    def _parse_group(self, rows: list[dict], evs: list, raw: dict) -> list[AgentOutput]:
        items = raw.get("items")
        if not isinstance(items, list):
            items = []
        by_id: dict[int, dict] = {}
        for it in items:
            if isinstance(it, dict):
                try:
                    by_id[int(it.get("id"))] = it
                except (TypeError, ValueError):
                    continue

        out: list[AgentOutput] = []
        for i, (row, ev) in enumerate(zip(rows, evs)):
            it = by_id.get(i + 1)
            if it is None and len(items) == len(rows) and isinstance(items[i], dict):
                it = items[i]
            if it is None:
                out.append(AgentOutput(
                    thread_id=int(row["thread_id"]), intent="service_complaint",
                    intent_confidence=0.0, action="escalate",
                    escalation_reason_id="unparseable",
                    escalation_rationale="model returned no item for this ticket",
                    reply="", n_evidence=len(ev), error="missing item in batch response",
                ))
                continue
            action = _coerce_action(it.get("action"))
            used = [int(u) for u in (it.get("evidence_used") or [])
                    if str(u).strip().lstrip("-").isdigit()]
            out.append(AgentOutput(
                thread_id=int(row["thread_id"]),
                intent=_coerce_intent(it.get("intent"), self.taxonomy),
                intent_confidence=float(it.get("intent_confidence") or 0.0),
                action=action,
                escalation_reason_id=_coerce_reason(it.get("escalation_reason_id"), action),
                escalation_rationale=str(it.get("escalation_rationale") or "").strip()[:300],
                reply=_clean_reply(it.get("reply")),
                evidence_used=used,
                evidence_thread_ids=[e.thread_id for e in ev],
                top_similarity=ev[0].score if ev else 0.0,
                n_evidence=len(ev),
            ))
        return out

    def run_group(self, rows: list[dict], evs: list) -> list[AgentOutput]:
        """Handle N tickets in ONE request. See prompts.AGENT_BATCH_SYSTEM."""
        blocks = "\n".join(
            AGENT_BATCH_TICKET.format(
                id=i + 1, evidence=render_evidence(ev), message=row["customer_text"]
            )
            for i, (row, ev) in enumerate(zip(rows, evs))
        )
        messages = [
            {"role": "system",
             "content": f"[prompt {PROMPT_VERSION}]\n" + self._system(AGENT_BATCH_SYSTEM)},
            {"role": "user", "content": AGENT_BATCH_USER.format(blocks=blocks, n=len(rows))},
        ]
        # Output scales with the batch: ~150 tokens per ticket, plus room for the
        # hidden reasoning these models emit before any content at all.
        budget = 240 * len(rows) + 1500
        try:
            text = self.client.chat(messages, json_object=True, max_tokens=budget)
            try:
                raw = parse_json(text)
            except LLMError:
                # Truncated mid-array: keep the tickets that did come back whole.
                raw = {"items": salvage_items(text)}
                if not raw["items"]:
                    raise
        except QuotaExhausted as exc:
            return [AgentOutput(
                thread_id=int(r["thread_id"]), intent="service_complaint",
                intent_confidence=0.0, action="escalate",
                escalation_reason_id="unparseable", escalation_rationale="",
                reply="", error=f"QUOTA:{exc}"[:200],
            ) for r in rows]
        except LLMError as exc:
            return [AgentOutput(
                thread_id=int(r["thread_id"]), intent="service_complaint",
                intent_confidence=0.0, action="escalate",
                escalation_reason_id="unparseable",
                escalation_rationale="model call failed; defaulted to human handoff",
                reply="", error=str(exc)[:200],
            ) for r in rows]
        return self._parse_group(rows, evs, raw)

    def classify_group(self, rows: list[dict]) -> list[AgentOutput]:
        """Ablation counterpart of `run_group`: same task, no evidence, no reply."""
        blocks = "\n".join(
            CLASSIFY_BATCH_TICKET.format(id=i + 1, message=r["customer_text"])
            for i, r in enumerate(rows)
        )
        messages = [
            {"role": "system",
             "content": f"[prompt {PROMPT_VERSION}]\n" + self._system(CLASSIFY_BATCH_SYSTEM)},
            {"role": "user", "content": AGENT_BATCH_USER.format(blocks=blocks, n=len(rows))},
        ]
        budget = 130 * len(rows) + 1200
        try:
            text = self.client.chat(messages, json_object=True, max_tokens=budget)
            try:
                raw = parse_json(text)
            except LLMError:
                raw = {"items": salvage_items(text)}
                if not raw["items"]:
                    raise
        except LLMError as exc:
            tag = "QUOTA:" if isinstance(exc, QuotaExhausted) else ""
            return [AgentOutput(
                thread_id=int(r["thread_id"]), intent="service_complaint",
                intent_confidence=0.0, action="escalate",
                escalation_reason_id="unparseable",
                escalation_rationale="model call failed", reply="",
                error=f"{tag}{exc}"[:200],
            ) for r in rows]
        return self._parse_group(rows, [[] for _ in rows], raw)

    def run_batched(
        self,
        rows: Iterable[dict],
        *,
        group_size: int = 15,
        classify_only: bool = False,
        desc: str = "agent(grouped)",
    ) -> list[AgentOutput]:
        rows = list(rows)
        if classify_only:
            groups = [(rows[i : i + group_size], None) for i in range(0, len(rows), group_size)]
            work = lambda g: self.classify_group(g[0])  # noqa: E731
        else:
            all_ev = self.retriever.search_batch([r["customer_text"] for r in rows])
            groups = [
                (rows[i : i + group_size], all_ev[i : i + group_size])
                for i in range(0, len(rows), group_size)
            ]
            work = lambda g: self.run_group(g[0], g[1])  # noqa: E731
        workers = max(1, int(self.cfg["llm"].get("concurrency", 2)))
        out: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(work, groups), total=len(groups), desc=desc):
                out.extend(res)
        return out

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

        # Once the provider's daily budget is gone, every remaining call will fail the
        # same way. Marching through the rest wastes ~12 minutes per retry, so the
        # first quota refusal short-circuits the batch. Cached cases still replay.
        quota_hit = threading.Event()

        def _work(pair):
            row, ev = pair
            if quota_hit.is_set():
                return AgentOutput(
                    thread_id=int(row["thread_id"]), intent="service_complaint",
                    intent_confidence=0.0, action="escalate",
                    escalation_reason_id="unparseable", escalation_rationale="",
                    reply="", error="QUOTA:skipped after daily budget exhausted",
                )
            if classify_only:
                out = self.classify_only(int(row["thread_id"]), row["customer_text"])
            else:
                out = self.run_one(int(row["thread_id"]), row["customer_text"], ev)
            if (out.error or "").startswith("QUOTA"):
                quota_hit.set()
            return out

        out: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(_work, zip(rows, all_ev)), total=len(rows), desc=desc):
                out.append(res)
        return out
