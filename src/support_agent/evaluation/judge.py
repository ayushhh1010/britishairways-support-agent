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
from ..llm import LLMClient, LLMError, QuotaExhausted, parse_json, salvage_items
from ..prompts import (
    JUDGE_BATCH_ITEM,
    JUDGE_BATCH_SYSTEM,
    JUDGE_SYSTEM,
    JUDGE_USER,
    PROMPT_VERSION,
)
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
    """Grades replies on a judge model that may live on a different provider.

    Independence is the point: the judge should share neither a model family nor an
    inference stack with the generator. When `judge_provider` differs from the main
    provider, this builds its own client against that endpoint rather than reusing
    the generator's.
    """

    def __init__(self, client: LLMClient | None = None, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        lc = self.cfg["llm"]
        self.model = lc["judge_model"]

        jp = lc.get("judge_provider")
        if jp and jp != lc["provider"]:
            jcfg = {**self.cfg, "llm": {**lc}}
            jcfg["llm"]["provider"] = jp
            jcfg["llm"]["base_url"] = lc.get("judge_base_url", lc["base_url"])
            jcfg["llm"]["request_timeout"] = int(lc.get("judge_timeout", 90))
            # Rate limits belong to the JUDGE's provider, not the generator's. Copying
            # the generator's caps across throttled a 3-second judge to one request per
            # minute, which looks exactly like a hang.
            jcfg["llm"]["rpm_limit"] = int(lc.get("judge_rpm_limit", 25))
            jcfg["llm"]["tpm_limit"] = int(lc.get("judge_tpm_limit", 7000))
            cap = int(lc.get("judge_max_output", 2500))
            jcfg["llm"]["model_limits"] = {
                **(lc.get("model_limits") or {}),
                self.model: {"max_output": cap, "otpm": int(lc.get("judge_otpm", 20000))},
            }
            self.client = LLMClient(jcfg)
        else:
            self.client = client or LLMClient(self.cfg)

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
            raw = self.client.chat_json(messages, model=self.model, max_tokens=1500)  # Gemma prepends a <thought> block; leave room for it
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

    def judge_group(self, jobs: list[dict]) -> list[JudgeVerdict]:
        """Grade N (message, candidate) pairs in ONE request.

        Pairs from different systems are interleaved by the caller and carry no system
        label, so the judge cannot tell whose reply it is grading.
        """
        blocks = "\n".join(
            JUDGE_BATCH_ITEM.format(
                id=i + 1, message=j["message"],
                evidence=render_evidence(list(j.get("evidence") or ())),
                historical_reply=j.get("historical_reply") or "(none recorded)",
                candidate=j["candidate"] or "(empty reply)",
            )
            for i, j in enumerate(jobs)
        )
        tail = f"\n--- END ---\nReturn the JSON object with exactly {len(jobs)} items."
        messages = [
            {"role": "system",
             "content": f"[prompt {PROMPT_VERSION}]\n" + JUDGE_BATCH_SYSTEM},
            {"role": "user", "content": blocks + tail},
        ]
        budget = 160 * len(jobs) + 1500
        try:
            text = self.client.chat(messages, model=self.model, json_object=True, max_tokens=budget)
            try:
                raw = parse_json(text)
            except LLMError:
                raw = {"items": salvage_items(text)}
                if not raw["items"]:
                    raise
        except LLMError as exc:
            tag = "QUOTA:" if isinstance(exc, QuotaExhausted) else ""
            return [JudgeVerdict(j["thread_id"], j["system"], 1, 1, 1, 1, False, False,
                                 "", f"{tag}{exc}"[:200]) for j in jobs]

        items = raw.get("items") if isinstance(raw.get("items"), list) else []
        by_id = {}
        for it in items:
            if isinstance(it, dict):
                try:
                    by_id[int(it.get("id"))] = it
                except (TypeError, ValueError):
                    continue
        out = []
        for i, j in enumerate(jobs):
            it = by_id.get(i + 1)
            if it is None and len(items) == len(jobs) and isinstance(items[i], dict):
                it = items[i]
            if it is None:
                out.append(JudgeVerdict(j["thread_id"], j["system"], 1, 1, 1, 1, False, False,
                                        "", "missing item in batch response"))
                continue
            out.append(JudgeVerdict(
                thread_id=j["thread_id"], system=j["system"],
                groundedness=_clip(it.get("groundedness")),
                helpfulness=_clip(it.get("helpfulness")),
                tone=_clip(it.get("tone")), safety=_clip(it.get("safety")),
                acceptable=bool(it.get("acceptable")),
                beats_historical=bool(it.get("beats_historical")),
                critique=str(it.get("critique") or "").strip()[:300],
            ))
        return out

    def judge_batched(
        self, jobs: list[dict], *, group_size: int = 20, desc: str = "judge(grouped)"
    ) -> list[JudgeVerdict]:
        groups = [jobs[i : i + group_size] for i in range(0, len(jobs), group_size)]
        workers = max(1, int(self.cfg["llm"].get("concurrency", 2)))
        out: list[JudgeVerdict] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(self.judge_group, groups), total=len(groups), desc=desc):
                out.extend(res)
        return out

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
        # A failed judge call is MISSING DATA, not a score of 1. Averaging the
        # fail-safe 1/1/1/1 placeholder in drags every system towards 1.0 and makes
        # real human replies look terrible -- which is exactly how a broken judge run
        # produced a plausible-looking table on a previous attempt.
        scored = [r for r in rows if not r.error]
        n_err = len(rows) - len(scored)
        if not scored:
            out[sysname] = {"n": 0, "errors": n_err, "note": "no successful judgements"}
            continue
        out[sysname] = {
            "n": len(scored),
            **{d: round(float(np.mean([getattr(r, d) for r in scored])), 3) for d in DIMENSIONS},
            "mean_score": round(float(np.mean([r.mean_score for r in scored])), 3),
            "acceptable_rate": round(float(np.mean([r.acceptable for r in scored])), 4),
            "beats_historical_rate": round(float(np.mean([r.beats_historical for r in scored])), 4),
            "errors": n_err,
        }
    return out
