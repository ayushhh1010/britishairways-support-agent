"""Config loading. YAML defaults, env-var overrides, no hidden globals."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

_ENV_OVERRIDES = {
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "GEN_MODEL": ("llm", "gen_model"),
    "JUDGE_MODEL": ("llm", "judge_model"),
    "BRAND": ("brand",),
}


class Config(dict):
    """dict with dotted access: cfg["llm"]["gen_model"] or cfg.get_path("paths.results")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def abs_path(self, dotted: str) -> Path:
        rel = self.get_path(dotted)
        if rel is None:
            raise KeyError(dotted)
        return ROOT / rel


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    for env_key, target in _ENV_OVERRIDES.items():
        val = os.getenv(env_key)
        if not val:
            continue
        node = raw
        for part in target[:-1]:
            node = node.setdefault(part, {})
        node[target[-1]] = val

    return Config(raw)


def api_key(provider: str) -> str | None:
    return {
        "cerebras": os.getenv("CEREBRAS_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "openai_compatible": os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
        "ollama": None,
    }.get(provider)
