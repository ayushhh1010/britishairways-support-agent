"""Provider-agnostic chat client with an on-disk cache.

Design notes
------------
* Every call is keyed by sha256(provider|model|params|messages). A cache hit costs
  no network. This is what makes the README's "reproduce in <15 min" claim honest:
  a fresh clone with the shipped cache replays the exact run offline.
* Rate limiting is a simple token bucket sized for the Groq free tier; 429s are
  retried with exponential backoff that honours `retry-after`.
* `json_object` asks the provider for strict JSON; we still parse defensively
  because no provider is reliable here.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from .config import ROOT, api_key, load_config


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Daily token budget for a model is gone. Waiting minutes will not help."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class _RateLimiter:
    """Rolling-window limiter over BOTH requests and tokens, per model.

    On the Groq free tier the binding constraint is tokens-per-minute (8k), not
    requests-per-minute. A request-only limiter sails past the RPM cap and then
    collects a wall of 429s. We reserve an estimated token cost up front and sleep
    until the rolling 60s window has room.
    """

    def __init__(self, rpm: int, tpm: int, otpm: int) -> None:
        self._rpm = max(rpm, 1)
        self._tpm = max(tpm, 1)
        self._otpm = max(otpm, 1)
        self._lock = threading.Lock()
        self._events: list[tuple[float, int, int]] = []  # (ts, total_tokens, out_tokens)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._events = [e for e in self._events if e[0] > cutoff]

    def acquire(self, est_tokens: int, est_output: int) -> None:
        est_tokens = max(1, min(est_tokens, self._tpm))
        est_output = max(1, min(est_output, self._otpm))
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used_total = sum(e[1] for e in self._events)
                used_out = sum(e[2] for e in self._events)
                if (
                    used_total + est_tokens <= self._tpm
                    and used_out + est_output <= self._otpm
                    and len(self._events) + 1 <= self._rpm
                ):
                    self._events.append((now, est_tokens, est_output))
                    return
                oldest = self._events[0][0] if self._events else now
                wait = max(0.05, 60.0 - (now - oldest))
            time.sleep(min(wait, 5.0))


class LLMClient:
    def __init__(self, cfg: dict | None = None, *, cache_dir: Path | None = None) -> None:
        self.cfg = cfg or load_config()
        lc = self.cfg["llm"]
        self.provider: str = lc["provider"]
        self.base_url: str = lc["base_url"].rstrip("/")
        self.timeout: int = int(lc.get("request_timeout", 90))
        self.max_retries: int = int(lc.get("max_retries", 6))
        self.cache_dir = cache_dir or (ROOT / self.cfg["paths"]["cache"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Quotas are per model on this provider, so one shared limiter would throttle
        # the judge against the generator's spend for no reason.
        self._rpm = int(lc.get("rpm_limit", 28))
        self._tpm = int(lc.get("tpm_limit", 8000))
        self._limiters: dict[str, _RateLimiter] = {}
        self._limiter_lock = threading.Lock()
        self.reasoning_effort: str | None = lc.get("reasoning_effort") or None
        # requests.Session is NOT thread-safe. Sharing one across worker threads
        # intermittently wedges a request inside urllib3's connection pool, past any
        # socket timeout -- observed here twice as a run that simply stopped making
        # progress with idle CPU. One session per thread, created on first use.
        self._thread_local = threading.local()
        self.usage = Usage()
        self._usage_lock = threading.Lock()

    @property
    def _session(self) -> requests.Session:
        s = getattr(self._thread_local, "session", None)
        if s is None:
            s = requests.Session()
            self._thread_local.session = s
        return s

    def limits_for(self, model: str) -> dict:
        ml = self.cfg["llm"].get("model_limits") or {}
        base = dict(ml.get("default", {"max_output": 2000, "otpm": 6000}))
        base.update(ml.get(model, {}))
        return base

    def _limiter_for(self, model: str) -> _RateLimiter:
        with self._limiter_lock:
            if model not in self._limiters:
                otpm = int(self.limits_for(model)["otpm"])
                self._limiters[model] = _RateLimiter(self._rpm, self._tpm, otpm)
            return self._limiters[model]

    # ---------------- cache ----------------

    def _key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(
            {"provider": self.provider, "payload": payload}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _cache_put(self, key: str, value: dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    # ---------------- transport ----------------

    def _endpoint(self) -> str:
        if self.provider == "ollama":
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = api_key(self.provider)
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    # Reserving the full `max_tokens` throttles us against a TPM cap far harder than
    # reality warrants: with reasoning_effort=low these calls complete in ~150-250
    # tokens against a 700 ceiling. We reserve a realistic completion instead, which
    # is safe because the limiter is a soft guard and 429s are still retried.
    COMPLETION_RESERVE = 300
    MAX_RETRY_SLEEP = 75.0  # seconds; see _post()

    @classmethod
    def _estimate_tokens(cls, payload: dict) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
        ceiling = int(payload.get("max_tokens", 512))
        return chars // 4 + min(ceiling, cls.COMPLETION_RESERVE) + 32

    def _post(self, payload: dict) -> dict:
        last_err: Exception | None = None
        est = self._estimate_tokens(payload)
        est_out = int(payload.get("max_tokens", self.COMPLETION_RESERVE))
        limiter = self._limiter_for(str(payload.get("model", "default")))
        for attempt in range(self.max_retries):
            limiter.acquire(est, est_out)
            try:
                resp = self._session.post(
                    self._endpoint(),
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (408, 429, 500, 502, 503, 529):
                # A tokens-per-DAY 429 will not clear by waiting a few seconds, and
                # the provider does not expose TPD in its rate-limit headers -- the
                # only signal is this message. Retrying it burns the retry budget and
                # hides the real cause behind a slow stall, so fail loudly instead.
                body_l = resp.text.lower().replace("_", "").replace("-", "")
                daily = any(k in body_l for k in ("per day", "perday", "requestsperday"))
                if resp.status_code == 429 and daily:
                    raise QuotaExhausted(
                        f"daily token quota exhausted for {payload.get('model')}: "
                        f"{resp.text[:220]}"
                    )
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    delay = float(retry_after)
                elif resp.status_code == 503:
                    # "Model is experiencing high demand" is transient congestion, not a
                    # quota. Exponential backoff from a 1s base gives up after ~30s and
                    # throws the request away; a linear 15s-per-attempt ramp actually
                    # outlasts these spikes.
                    delay = 15.0 * (attempt + 1) + random.random() * 5
                else:
                    delay = 2 ** attempt + random.random()
                # Cap it. This provider reports the DAILY window in `retry-after`,
                # so an uncapped honour of that header parks a worker thread for
                # hours on what is usually a per-minute throttle.
                delay = min(delay, self.MAX_RETRY_SLEEP)
                last_err = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if os.getenv("LLM_DEBUG"):
                    print(
                        f"[llm] {resp.status_code} on {payload.get('model')} "
                        f"attempt {attempt + 1}/{self.max_retries}, sleeping {delay:.1f}s "
                        f"(retry-after={retry_after or 'none'})",
                        flush=True,
                    )
                time.sleep(delay)
                continue
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        raise LLMError(f"exhausted {self.max_retries} retries: {last_err}")

    # ---------------- public API ----------------

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_object: bool = False,
        use_cache: bool = True,
    ) -> str:
        lc = self.cfg["llm"]
        payload: dict[str, Any] = {
            "model": model or lc["gen_model"],
            "messages": list(messages),
            "temperature": lc["temperature"] if temperature is None else temperature,
        }
        if self.provider == "ollama":
            payload["stream"] = False
            payload["options"] = {
                "temperature": payload.pop("temperature"),
                "num_predict": max_tokens or lc["max_tokens"],
            }
            if json_object:
                payload["format"] = "json"
        else:
            requested = max_tokens or lc["max_tokens"]
            ceiling = int(self.limits_for(payload["model"])["max_output"])
            payload["max_tokens"] = min(requested, ceiling)
            no_json_mode = set(lc.get("no_json_mode_models") or [])
            if json_object and not any(m in payload["model"] for m in no_json_mode):
                payload["response_format"] = {"type": "json_object"}
            # Thinking models spend part of `max_tokens` on hidden reasoning before
            # emitting content, which silently TRUNCATES JSON when the budget is
            # tight. Measured on gemini-3.6-flash: 669 thinking tokens against a 700
            # budget left 27 for the answer and produced unparseable output.
            # reasoning_effort="low" cut thinking 729 -> 99 with identical answers.
            # Only these families accept the parameter; others reject it with a 400.
            if self.reasoning_effort and any(
                fam in payload["model"] for fam in ("gpt-oss", "gemini")
            ):
                payload["reasoning_effort"] = self.reasoning_effort

        key = self._key(payload)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                with self._usage_lock:
                    self.usage.cache_hits += 1
                return cached["content"]

        raw = self._post(payload)
        if self.provider == "ollama":
            content = raw["message"]["content"]
            pt = raw.get("prompt_eval_count", 0)
            ct = raw.get("eval_count", 0)
        else:
            # Thinking models can return a message with no `content` when the token
            # budget is spent on reasoning, so this must not KeyError.
            content = (raw["choices"][0].get("message") or {}).get("content")
            u = raw.get("usage", {}) or {}
            pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

        content = content or ""
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += pt
            self.usage.completion_tokens += ct
        self._cache_put(key, {"content": content, "prompt_tokens": pt, "completion_tokens": ct})
        return content

    def chat_json(self, messages: Sequence[dict[str, str]], **kw) -> dict:
        text = self.chat(messages, json_object=True, **kw)
        return parse_json(text)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
# Gemma-family models prepend a reasoning block that itself contains braces, which
# derails naive first-brace-to-last-brace extraction. Strip it before parsing.
_THOUGHT = re.compile(r"<thought>.*?(?:</thought>|$)", re.S | re.I)


def parse_json(text: str) -> dict:
    """Best-effort JSON extraction. Models wrap JSON in prose and fences constantly."""
    text = _THOUGHT.sub(" ", (text or "")).strip()
    if not text:
        raise LLMError("empty model response")
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return {"items": obj}
    raise LLMError(f"could not parse JSON from: {text[:300]}")


def _candidates(text: str) -> Iterable[str]:
    yield text
    m = _FENCE.search(text)
    if m:
        yield m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start : end + 1]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        yield text[start : end + 1]


def healthcheck(client: LLMClient | None = None) -> dict:
    client = client or LLMClient()
    cfg = client.cfg["llm"]
    out = {
        "provider": client.provider,
        "gen_model": cfg["gen_model"],
        "judge_model": cfg["judge_model"],
    }
    if client.provider != "ollama" and not api_key(client.provider):
        out["status"] = "MISSING_API_KEY"
        return out
    try:
        reply = client.chat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=256,  # thinking models spend a small budget entirely on reasoning
            use_cache=False,
        )
        out["status"] = "ok"
        out["sample"] = reply.strip()[:40]
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"ERROR: {exc}"
    return out


_ITEM_OBJ = re.compile(r"\{[^{}]*\}", re.S)


def salvage_items(text: str) -> list[dict]:
    """Recover whatever complete objects survive in a truncated `items` array.

    A batched request that gets cut off mid-JSON still contains valid results for
    most of its tickets. Throwing the whole response away wastes both those answers
    and one of a small daily request budget, so we parse out every well-formed
    object and let the caller match them up by id.
    """
    text = _THOUGHT.sub(" ", text or "")
    out: list[dict] = []
    for m in _ITEM_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "id" in obj:
            out.append(obj)
    return out
