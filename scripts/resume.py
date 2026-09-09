"""One command that drives the pipeline to completion across daily quota windows.

Run it, and it does whatever is still outstanding, in dependency order, stopping
cleanly when the provider's daily token budget runs out. Every completed LLM call is
already on disk in `llm_cache/`, so re-running costs nothing for work already done and
only the remaining calls hit the API.

    python scripts/resume.py            # do as much as today's quota allows
    python scripts/resume.py --status   # just report what is left

Why this exists: the free tier caps at 200k tokens/day per model, and a full sweep
needs ~1.3M. Rather than a human remembering which stage to re-run, this figures it
out and is safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402

QUOTA_EXIT = 3


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def stages(cfg) -> list[dict]:
    g = PROOT / cfg["paths"]["golden"]
    r = PROOT / cfg["paths"]["results"]
    return [
        {
            "name": "prelabels",
            "done": count(g / "prelabels.jsonl") > 0,
            "cmd": [sys.executable, "scripts/prelabel_golden.py", "--batch-size", "8"],
            "why": "dual-model pre-labels for the golden set",
        },
        {
            "name": "golden_set",
            "done": count(g / "golden_set.jsonl") > 0,
            "cmd": [sys.executable, "scripts/build_golden.py"],
            "why": "merge adjudicated labels into the final golden set",
        },
        {
            "name": "evaluation",
            "done": (r / "eval_results.json").exists(),
            "cmd": [sys.executable, "scripts/run_eval.py", "--ablation", "--judge-n", "120"],
            "why": "agent + baselines + ablation + LLM judge (the expensive stage)",
        },
        {
            "name": "failure_analysis",
            "done": (r / "failures.md").exists(),
            "cmd": [sys.executable, "scripts/failure_analysis.py"],
            "why": "mine predictions for failure modes",
        },
        {
            "name": "judge_bias",
            "done": (r / "judge_bias.json").exists(),
            "cmd": [sys.executable, "scripts/judge_bias.py", "--n", "60"],
            "why": "measure judge self-preference bias",
        },
        {
            "name": "label_agreement",
            "done": (r / "label_agreement.json").exists(),
            "cmd": [sys.executable, "scripts/label_agreement.py"],
            "why": "human vs model label agreement",
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    args = ap.parse_args()

    cfg = load_config()
    todo = stages(cfg)

    print("PIPELINE STAGES")
    for s in todo:
        print(f"  [{'x' if s['done'] else ' '}] {s['name']:<18} {s['why']}")
    remaining = [s for s in todo if not s["done"]]
    if args.status:
        print(f"\n{len(remaining)} stage(s) outstanding")
        return
    if not remaining:
        print("\nEverything is complete. `make repro` replays it offline from the cache.")
        return

    env = {"PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}
    import os

    full_env = {**os.environ, **env}

    for s in remaining:
        print(f"\n=== {s['name']}: {s['why']}")
        rc = subprocess.call(s["cmd"], cwd=str(ROOT), env=full_env)
        if rc == QUOTA_EXIT:
            print(
                f"\nDaily token quota reached during `{s['name']}`.\n"
                "  Nothing is lost -- every completed call is cached on disk.\n"
                "  Re-run `python scripts/resume.py` after the provider's 24h window\n"
                "  refills and it will carry on from exactly here.\n"
                "  `python scripts/run_status.py` shows live quota per model."
            )
            sys.exit(QUOTA_EXIT)
        if rc != 0:
            print(f"\nStage `{s['name']}` failed with exit code {rc}.")
            sys.exit(rc)

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
