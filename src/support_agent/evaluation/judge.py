"""LLM-as-judge for reply quality.

Two deliberate design choices:

1. Different model family from the generator. Replies are drafted by
   `openai/gpt-oss-120b` and graded by `qwen/qwen3.6-27b` (Alibaba). A judge from
   the same family as the generator inflates scores through self-preference bias;
   `scripts/judge_bias.py` measures that effect rather than assuming it.

2. The judge is blind to which system produced a reply. Candidates from the agent,
   both baselines, and the real historical BA reply are all graded with the same
   prompt and no system label, so any ordering it produces is about the text.

The judge is itself evaluated against human ratings -- see `scripts/judge_agreement.py`.
A judge whose agreement with a human is unknown is decoration, not evidence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Sequence

from tqdm import tqdm

from ..config import load_config
from ..llm import LLMClient, LLMError, QuotaExhausted
from ..prompts import JUDGE_SYSTEM, JUDGE_USER, PROMPT_VERSION
from ..retrieve import Evidence, render_evidence

DIMENSIONS = ("groundedness", "helpfulness", "tone", "safety")


@dataclass
class JudgeVerdict:
    thread_id: int
    system: str
    groundedness: int
    helpfulness: int
    tone: int
    safety: int
    acceptable: bool
    beats_historical: bool
    critique: str
    error: str | None = None

    @property
    def mean_score(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mean_score"] = round(self.mean_score, 4)
        return d


def _clip(value: object, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


class ReplyJudge:
    def __init__(self, client: LLMClient | None = None, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)
        self.model = self.cfg["llm"]["judge_model"]

    def judge_one(
        self,
        thread_id: int,
        system: str,
        message: str,
        candidate: str,
        historical_reply: str,
        evidence: Sequence[Evidence] = (),
    ) -> JudgeVerdict:
        if not (candidate or "").strip():
            return JudgeVerdict(
                thread_id, system, 1, 1, 1, 1, False, False, "empty reply", None
            )
        messages = [
            {"role": "system", "content": f"[prompt {PROMPT_VERSION}]\n" + JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    message=message,
                    evidence=render_evidence(list(evidence)),
                    historical_reply=historical_reply or "(none recorded)",
                    candidate=candidate,
                ),
            },
        ]
        try:
            raw = self.client.chat_json(messages, model=self.model, max_tokens=400)
        except QuotaExhausted as exc:
            return JudgeVerdict(
                thread_id, system, 1, 1, 1, 1, False, False, "", f"QUOTA:{exc}"[:200]
            )
        except LLMError as exc:
            return JudgeVerdict(
                thread_id, system, 1, 1, 1, 1, False, False, "judge call failed", str(exc)[:200]
            )
        return JudgeVerdict(
            thread_id=thread_id,
            system=system,
            groundedness=_clip(raw.get("groundedness")),
            helpfulness=_clip(raw.get("helpfulness")),
            tone=_clip(raw.get("tone")),
            safety=_clip(raw.get("safety")),
            acceptable=bool(raw.get("acceptable")),
            beats_historical=bool(raw.get("beats_historical")),
            critique=str(raw.get("critique") or "").strip()[:300],
        )

    def judge_batch(self, jobs: list[dict], desc: str = "judge") -> list[JudgeVerdict]:
        """`jobs` items: thread_id, system, message, candidate, historical_reply, evidence."""
        workers = int(self.cfg["llm"].get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = pool.map(
                lambda j: self.judge_one(
                    j["thread_id"], j["system"], j["message"], j["candidate"],
                    j.get("historical_reply", ""), j.get("evidence", ()),
                ),
                jobs,
            )
            return list(tqdm(futures, total=len(jobs), desc=desc))


def aggregate(verdicts: Sequence[JudgeVerdict]) -> dict:
    """Per-system means. Empty groups return NaN rather than crashing a sweep."""
    import numpy as np

    out: dict[str, dict] = {}
    systems = sorted({v.system for v in verdicts})
    for sysname in systems:
        rows = [v for v in verdicts if v.system == sysname]
        if not rows:
            continue
        out[sysname] = {
            "n": len(rows),
            **{d: round(float(np.mean([getattr(r, d) for r in rows])), 3) for d in DIMENSIONS},
            "mean_score": round(float(np.mean([r.mean_score for r in rows])), 3),
            "acceptable_rate": round(float(np.mean([r.acceptable for r in rows])), 4),
            "beats_historical_rate": round(float(np.mean([r.beats_historical for r in rows])), 4),
            "errors": sum(1 for r in rows if r.error),
        }
    return out
