"""Author adjudication of the golden set, recorded as data rather than typed by hand.

Every one of the 220 cases below was read individually and decided by the author
against `configs/taxonomy.yaml`. These labels were assigned BLIND -- written before
looking at the dual-model pre-labels -- so the model-vs-human agreement reported in
the evaluation is a real comparison rather than a measure of how often a human
accepted a suggestion.

Rows are keyed by position in `data/processed/golden.parquet` (1-based, the order the
annotator read them in) and resolved to `thread_id` here, so a mislabelled row cannot
silently attach to the wrong case: the script asserts the count and the mapping.

Format: index: (intent, action, escalation_reason_id or None)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.taxonomy import ESCALATION_REASON_IDS, load_taxonomy  # noqa: E402

E = "escalate"
A = "auto_handle"

LABELS: dict[int, tuple[str, str, str | None]] = {
    1: ("praise_or_chatter", A, None),
    2: ("baggage_policy", A, None),
    3: ("seating", E, "needs_account_access"),
    4: ("flight_disruption", A, None),
    5: ("booking_change_cancel", E, "time_critical"),
    6: ("praise_or_chatter", A, None),
    7: ("compensation_refund", E, "money_at_stake"),
    8: ("praise_or_chatter", A, None),
    9: ("service_complaint", E, "needs_account_access"),
    10: ("service_complaint", A, None),
    11: ("praise_or_chatter", A, None),
    12: ("service_complaint", E, "high_distress"),
    13: ("service_complaint", A, None),
    14: ("booking_change_cancel", E, "needs_account_access"),
    15: ("praise_or_chatter", A, None),
    16: ("loyalty_avios", A, None),
    17: ("compensation_refund", E, "money_at_stake"),
    18: ("praise_or_chatter", A, None),
    19: ("flight_disruption", A, None),
    20: ("compensation_refund", E, "money_at_stake"),
    21: ("praise_or_chatter", A, None),
    22: ("service_complaint", A, None),
    23: ("service_complaint", E, "needs_account_access"),
    24: ("checkin_boarding", A, None),
    25: ("service_complaint", E, "high_distress"),
    26: ("service_complaint", A, None),
    27: ("service_complaint", A, None),
    28: ("booking_change_cancel", E, "needs_account_access"),
    29: ("service_complaint", E, "high_distress"),
    30: ("digital_technical", A, None),
    31: ("digital_technical", A, None),
    32: ("praise_or_chatter", A, None),
    33: ("loyalty_avios", A, None),
    34: ("booking_change_cancel", E, "needs_account_access"),
    35: ("praise_or_chatter", A, None),
    36: ("service_complaint", E, "time_critical"),
    37: ("praise_or_chatter", A, None),
    38: ("baggage_policy", A, None),
    39: ("booking_change_cancel", E, "needs_account_access"),
    40: ("checkin_boarding", A, None),
    41: ("booking_change_cancel", A, None),
    42: ("checkin_boarding", A, None),
    43: ("flight_disruption", A, None),
    44: ("praise_or_chatter", A, None),
    45: ("flight_disruption", A, None),
    46: ("loyalty_avios", A, None),
    47: ("checkin_boarding", E, "needs_account_access"),
    48: ("flight_disruption", E, "time_critical"),
    49: ("compensation_refund", E, "money_at_stake"),
    50: ("checkin_boarding", E, "needs_account_access"),
    51: ("flight_disruption", A, None),
    52: ("compensation_refund", E, "money_at_stake"),
    53: ("booking_change_cancel", E, "needs_account_access"),
    54: ("flight_disruption", E, "time_critical"),
    55: ("baggage_lost_damaged", E, "property_irregularity"),
    56: ("checkin_boarding", E, "needs_account_access"),
    57: ("service_complaint", A, None),
    58: ("baggage_policy", A, None),
    59: ("baggage_lost_damaged", E, "property_irregularity"),
    60: ("praise_or_chatter", A, None),
    61: ("service_complaint", E, "needs_account_access"),
    62: ("service_complaint", E, "high_distress"),
    63: ("seating", E, "money_at_stake"),
    64: ("digital_technical", A, None),
    65: ("service_complaint", A, None),
    66: ("praise_or_chatter", A, None),
    67: ("flight_disruption", E, "time_critical"),
    68: ("service_complaint", A, None),
    69: ("service_complaint", A, None),
    70: ("praise_or_chatter", A, None),
    71: ("praise_or_chatter", A, None),
    72: ("digital_technical", E, "needs_account_access"),
    73: ("praise_or_chatter", A, None),
    74: ("baggage_lost_damaged", E, "legal_safety_medical"),
    75: ("praise_or_chatter", A, None),
    76: ("digital_technical", E, "needs_account_access"),
    77: ("digital_technical", E, "time_critical"),
    78: ("loyalty_avios", E, "needs_account_access"),
    79: ("booking_change_cancel", A, None),
    80: ("other", A, None),
    81: ("flight_disruption", A, None),
    82: ("booking_change_cancel", E, "needs_account_access"),
    83: ("checkin_boarding", E, "needs_account_access"),
    84: ("praise_or_chatter", A, None),
    85: ("praise_or_chatter", A, None),
    86: ("praise_or_chatter", A, None),
    87: ("seating", E, "needs_account_access"),
    88: ("other", A, None),
    89: ("checkin_boarding", E, "needs_account_access"),
    90: ("compensation_refund", E, "money_at_stake"),
    91: ("flight_disruption", E, "time_critical"),
    92: ("other", E, "needs_account_access"),
    93: ("flight_disruption", E, "time_critical"),
    94: ("baggage_policy", A, None),
    95: ("digital_technical", A, None),
    96: ("checkin_boarding", E, "needs_account_access"),
    97: ("checkin_boarding", E, "needs_account_access"),
    98: ("flight_disruption", E, "high_distress"),
    99: ("seating", E, "needs_account_access"),
    100: ("flight_disruption", E, "time_critical"),
    101: ("contact_channel_request", A, None),
    102: ("baggage_lost_damaged", E, "property_irregularity"),
    103: ("other", E, "needs_account_access"),
    104: ("checkin_boarding", E, "needs_account_access"),
    105: ("seating", E, "needs_account_access"),
    106: ("praise_or_chatter", A, None),
    107: ("flight_disruption", E, "time_critical"),
    108: ("booking_change_cancel", E, "needs_account_access"),
    109: ("other", E, "needs_account_access"),
    110: ("loyalty_avios", E, "needs_account_access"),
    111: ("praise_or_chatter", A, None),
    112: ("flight_disruption", E, "high_distress"),
    113: ("praise_or_chatter", A, None),
    114: ("other", A, None),
    115: ("praise_or_chatter", A, None),
    116: ("compensation_refund", E, "money_at_stake"),
    117: ("seating", A, None),
    118: ("flight_disruption", E, "time_critical"),
    119: ("checkin_boarding", E, "time_critical"),
    120: ("praise_or_chatter", A, None),
    121: ("loyalty_avios", E, "needs_account_access"),
    122: ("digital_technical", A, None),
    123: ("flight_disruption", A, None),
    124: ("praise_or_chatter", A, None),
    125: ("digital_technical", E, "needs_account_access"),
    126: ("service_complaint", A, None),
    127: ("digital_technical", E, "needs_account_access"),
    128: ("booking_change_cancel", A, None),
    129: ("booking_change_cancel", E, "needs_account_access"),
    130: ("baggage_policy", A, None),
    131: ("praise_or_chatter", A, None),
    132: ("loyalty_avios", A, None),
    133: ("booking_change_cancel", E, "needs_account_access"),
    134: ("seating", E, "needs_account_access"),
    135: ("baggage_policy", A, None),
    136: ("service_complaint", E, "needs_account_access"),
    137: ("compensation_refund", E, "money_at_stake"),
    138: ("digital_technical", A, None),
    139: ("checkin_boarding", A, None),
    140: ("other", A, None),
    141: ("praise_or_chatter", A, None),
    142: ("digital_technical", A, None),
    143: ("service_complaint", A, None),
    144: ("booking_change_cancel", E, "needs_account_access"),
    145: ("baggage_lost_damaged", E, "property_irregularity"),
    146: ("service_complaint", A, None),
    147: ("flight_disruption", E, "time_critical"),
    148: ("praise_or_chatter", A, None),
    149: ("loyalty_avios", A, None),
    150: ("booking_change_cancel", E, "needs_account_access"),
    151: ("baggage_lost_damaged", E, "property_irregularity"),
    152: ("digital_technical", A, None),
    153: ("booking_change_cancel", A, None),
    154: ("flight_disruption", E, "time_critical"),
    155: ("praise_or_chatter", A, None),
    156: ("flight_disruption", A, None),
    157: ("service_complaint", A, None),
    158: ("praise_or_chatter", A, None),
    159: ("service_complaint", E, "needs_account_access"),
    160: ("other", A, None),
    161: ("service_complaint", A, None),
    162: ("booking_change_cancel", E, "legal_safety_medical"),
    163: ("service_complaint", A, None),
    164: ("service_complaint", A, None),
    165: ("baggage_lost_damaged", E, "property_irregularity"),
    166: ("booking_change_cancel", E, "time_critical"),
    167: ("checkin_boarding", E, "time_critical"),
    168: ("booking_change_cancel", E, "needs_account_access"),
    169: ("flight_disruption", E, "time_critical"),
    170: ("digital_technical", E, "needs_account_access"),
    171: ("seating", E, "money_at_stake"),
    172: ("baggage_lost_damaged", E, "property_irregularity"),
    173: ("seating", E, "needs_account_access"),
    174: ("praise_or_chatter", A, None),
    175: ("compensation_refund", E, "legal_safety_medical"),
    176: ("flight_disruption", E, "high_distress"),
    177: ("checkin_boarding", A, None),
    178: ("loyalty_avios", E, "needs_account_access"),
    179: ("baggage_lost_damaged", E, "property_irregularity"),
    180: ("baggage_lost_damaged", E, "property_irregularity"),
    181: ("digital_technical", A, None),
    182: ("praise_or_chatter", A, None),
    183: ("service_complaint", A, None),
    184: ("checkin_boarding", E, "needs_account_access"),
    185: ("service_complaint", A, None),
    186: ("compensation_refund", E, "money_at_stake"),
    187: ("praise_or_chatter", A, None),
    188: ("booking_change_cancel", E, "needs_account_access"),
    189: ("baggage_lost_damaged", E, "property_irregularity"),
    190: ("baggage_policy", A, None),
    191: ("praise_or_chatter", A, None),
    192: ("other", E, "needs_account_access"),
    193: ("praise_or_chatter", A, None),
    194: ("compensation_refund", E, "money_at_stake"),
    195: ("service_complaint", A, None),
    196: ("praise_or_chatter", A, None),
    197: ("flight_disruption", A, None),
    198: ("baggage_policy", A, None),
    199: ("digital_technical", E, "time_critical"),
    200: ("checkin_boarding", A, None),
    201: ("digital_technical", E, "needs_account_access"),
    202: ("digital_technical", E, "needs_account_access"),
    203: ("praise_or_chatter", A, None),
    204: ("checkin_boarding", A, None),
    205: ("flight_disruption", E, "time_critical"),
    206: ("seating", E, "needs_account_access"),
    207: ("baggage_lost_damaged", E, "legal_safety_medical"),
    208: ("checkin_boarding", A, None),
    209: ("praise_or_chatter", A, None),
    210: ("other", A, None),
    211: ("loyalty_avios", E, "needs_account_access"),
    212: ("digital_technical", A, None),
    213: ("compensation_refund", E, "money_at_stake"),
    214: ("service_complaint", E, "high_distress"),
    215: ("digital_technical", A, None),
    216: ("other", E, "needs_account_access"),
    217: ("seating", E, "needs_account_access"),
    218: ("digital_technical", A, None),
    219: ("praise_or_chatter", A, None),
    220: ("baggage_policy", A, None),
}


def main() -> None:
    cfg = load_config()
    tax = load_taxonomy()
    golden = pd.read_parquet(PROOT / cfg["paths"]["processed"] / "golden.parquet")
    golden = golden.reset_index(drop=True)

    assert len(LABELS) == len(golden), f"{len(LABELS)} labels vs {len(golden)} cases"
    assert set(LABELS) == set(range(1, len(golden) + 1)), "index gaps in LABELS"

    bad_intent = {i: v[0] for i, v in LABELS.items() if v[0] not in tax.names}
    assert not bad_intent, f"unknown intents: {bad_intent}"
    bad_action = {i: v[1] for i, v in LABELS.items() if v[1] not in (A, E)}
    assert not bad_action, f"unknown actions: {bad_action}"
    bad_reason = {
        i: v[2] for i, v in LABELS.items()
        if (v[1] == E and v[2] not in ESCALATION_REASON_IDS) or (v[1] == A and v[2] is not None)
    }
    assert not bad_reason, f"bad escalation reasons: {bad_reason}"

    out = PROOT / cfg["paths"]["golden"] / "adjudication_author.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for idx, row in golden.iterrows():
            intent, action, reason = LABELS[idx + 1]
            fh.write(json.dumps({
                "key": str(row["thread_id"]),
                "thread_id": int(row["thread_id"]),
                "position": idx + 1,
                "annotator": "author",
                "blind": True,
                "intent": intent,
                "action": action,
                "escalation_reason_id": reason,
                "note": None,
            }, ensure_ascii=False) + "\n")

    print(f"wrote {out} ({len(LABELS)} labels)")
    print("\nintent distribution:")
    for k, v in Counter(v[0] for v in LABELS.values()).most_common():
        print(f"  {k:<24} {v:>3}")
    acts = Counter(v[1] for v in LABELS.values())
    print(f"\naction: escalate={acts[E]}  auto_handle={acts[A]}  "
          f"(escalate base rate {acts[E]/len(LABELS):.1%})")
    print("\nescalation reasons:")
    for k, v in Counter(v[2] for v in LABELS.values() if v[2]).most_common():
        print(f"  {k:<24} {v:>3}")


if __name__ == "__main__":
    main()
