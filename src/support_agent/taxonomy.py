"""Load and render the intent taxonomy + triage policy."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import ROOT


@dataclass(frozen=True)
class Intent:
    name: str
    short: str
    definition: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: int
    brand: str
    intents: tuple[Intent, ...]
    escalate_rules: tuple[tuple[str, str], ...]
    auto_handle_rules: tuple[str, ...]
    default_action: str

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.intents]

    def get(self, name: str) -> Intent | None:
        return next((i for i in self.intents if i.name == name), None)

    # The per-intent "do NOT use when" clauses repeat the same handful of boundary
    # rules 13 times over, costing ~340 tokens. Stating each rule once costs ~90 and
    # says the same thing. Against a 200k tokens/day cap that is ~40 extra cases/day.
    _BOUNDARIES = (
        "BOUNDARY RULES (apply in order):\n"
        "- Customer asks for money (refund/compensation/reimbursement/disputed charge)"
        " -> compensation_refund, whatever the underlying incident was.\n"
        "- A SPECIFIC bag or item is missing/damaged -> baggage_lost_damaged."
        " A general rules question -> baggage_policy.\n"
        "- The site or app erroring IS the subject -> digital_technical."
        " Otherwise classify by what the customer was trying to do.\n"
        "- Disruption to a flight (delay/cancellation/missed connection) with no money"
        " asked -> flight_disruption. A voluntary date change -> booking_change_cancel.\n"
        "- A diffuse grievance with no specific actionable ask -> service_complaint.\n"
        "- Praise, photos, retweets, banter -> praise_or_chatter."
        " A message that is ONLY a channel request -> contact_channel_request.\n"
        "- A genuine support question none of the above fits -> other."
    )

    def render_for_prompt(self, *, style: str = "full", with_examples: bool = True) -> str:
        """Render the taxonomy for a prompt.

        `compact` keeps the name, the one-line gloss, and the boundary rules, and
        drops the definition paragraph, the includes, and the examples. It costs
        ~500 tokens against ~2000 for `full`. On a provider capped at 8k
        tokens/minute that is the difference between 3 and 8 calls per minute, and
        the boundary rules are the part that actually decides hard cases.
        """
        if style == "terse":
            names = "\n".join(f"- {i.name}: {i.short}" for i in self.intents)
            return names + "\n\n" + self._BOUNDARIES

        lines: list[str] = []
        for i in self.intents:
            lines.append(f"- {i.name}: {i.short}")
            if style == "compact":
                if i.excludes:
                    lines.append("    NOT: " + "; ".join(i.excludes))
                continue
            lines.append(f"    definition: {' '.join(i.definition.split())}")
            if i.includes:
                lines.append("    use when: " + "; ".join(i.includes))
            if i.excludes:
                lines.append("    do NOT use when: " + "; ".join(i.excludes))
            if with_examples and i.examples:
                lines.append(f'    example: "{i.examples[0]}"')
        return "\n".join(lines)

    # Terse gloss per rule id. The full prose lives in configs/taxonomy.yaml and in
    # the annotation codebook; the model only needs the discriminating clause, and
    # this costs ~90 tokens against ~275 for the full rendering.
    _TERSE = {
        "needs_account_access": "needs a booking/ticket/Executive Club lookup",
        "money_at_stake": "asks for money: refund, compensation, reimbursement, disputed charge",
        "legal_safety_medical": "EU261, legal threat, regulator/media, safety, medical, accessibility",
        "property_irregularity": "a specific bag or item is lost, delayed or damaged",
        "time_critical": "travelling within ~24h and blocked (stranded, cannot check in)",
        "high_distress": "severe distress, abuse, or a vulnerable passenger",
        "unparseable": "cannot be understood safely, or is not in English",
    }

    def render_triage_policy(self, *, style: str = "terse") -> str:
        if style != "terse":
            lines = ["ESCALATE if ANY of these hold:"]
            for rid, rule in self.escalate_rules:
                lines.append(f"  - [{rid}] {' '.join(rule.split())}")
            lines.append("AUTO-HANDLE only if ALL of these hold:")
            for rule in self.auto_handle_rules:
                lines.append(f"  - {' '.join(rule.split())}")
            lines.append(f"If unclear, default to: {self.default_action}.")
            return "\n".join(lines)
        parts = [f"{rid} = {self._TERSE.get(rid, rid)}" for rid, _ in self.escalate_rules]
        return (
            "ESCALATE if ANY apply -- " + "; ".join(parts) + ".\n"
            "AUTO-HANDLE only if the answer comes from public policy alone: no booking "
            "or account lookup, no money, no legal claim, no physical property, and a "
            f"wrong answer would merely inconvenience.\nIf unsure: {self.default_action}."
        )


@lru_cache(maxsize=2)
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    path = Path(path) if path else ROOT / "configs" / "taxonomy.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    intents = tuple(
        Intent(
            name=i["name"],
            short=i["short"],
            definition=i["definition"],
            includes=tuple(i.get("includes", ())),
            excludes=tuple(i.get("excludes", ())),
            examples=tuple(i.get("examples", ())),
        )
        for i in raw["intents"]
    )
    tri = raw["triage"]
    return Taxonomy(
        version=raw["version"],
        brand=raw["brand"],
        intents=intents,
        escalate_rules=tuple((r["id"], r["rule"]) for r in tri["escalate_if_any"]),
        auto_handle_rules=tuple(tri["auto_handle_if_all"]),
        default_action=tri["default"],
    )


ESCALATION_REASON_IDS = (
    "needs_account_access",
    "money_at_stake",
    "legal_safety_medical",
    "property_irregularity",
    "time_critical",
    "high_distress",
    "unparseable",
)
