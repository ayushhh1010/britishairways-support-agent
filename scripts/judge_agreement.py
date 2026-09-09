"""Does the LLM judge agree with a human?

Without this number, every reply-quality score in the report is unfalsifiable. We
compare the judge's ratings against human ratings of the SAME (message, candidate)
pairs, rated blind to which system produced them.

Reported:
  * Pearson + Spearman on the mean 1-5 score
  * Quadratic-weighted kappa per dimension (ordinal-aware)
  * Cohen's kappa on the binary `acceptable` decision -- the one that would actually
    gate an auto-send in production
  * Judge bias: mean(judge) - mean(human), i.e. whether the judge is systematically
    generous
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import ROOT as PROOT, load_config  # noqa: E402
from support_agent.evaluation.judge import DIMENSIONS  # noqa: E402
from support_agent.evaluation.metrics import agreement, quadratic_weighted_kappa  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    cfg = load_config()
    gdir = PROOT / cfg["paths"]["golden"]
    rdir = PROOT / cfg["paths"]["results"]

    human_files = sorted(gdir.glob("replies_*.jsonl"))
    if not human_files:
        raise SystemExit(
            "No human reply ratings found.\n"
            "Produce them with:\n"
            "  python -m support_agent.labeling.cli replies --annotator <name> --limit 50"
        )

    verdicts = {f"{v['thread_id']}::{v['system']}": v for v in read_jsonl(rdir / "judge_verdicts.jsonl")}
    if not verdicts:
        raise SystemExit("No judge verdicts. Run scripts/run_eval.py first.")

    report: dict = {"annotators": {}, "judge_model": cfg["llm"]["judge_model"]}

    for hf in human_files:
        annot = hf.stem.replace("replies_", "")
        human = read_jsonl(hf)
        paired = [(h, verdicts[h["key"]]) for h in human if h["key"] in verdicts]
        if len(paired) < 5:
            report["annotators"][annot] = {"n_paired": len(paired), "note": "too few to score"}
            continue

        h_mean = np.array([np.mean([h[d] for d in DIMENSIONS]) for h, _ in paired])
        j_mean = np.array([np.mean([j[d] for d in DIMENSIONS]) for _, j in paired])

        per_dim = {}
        for d in DIMENSIONS:
            hv = [h[d] for h, _ in paired]
            jv = [j[d] for _, j in paired]
            per_dim[d] = {
                "human_mean": round(float(np.mean(hv)), 3),
                "judge_mean": round(float(np.mean(jv)), 3),
                "bias_judge_minus_human": round(float(np.mean(jv) - np.mean(hv)), 3),
                "quadratic_weighted_kappa": round(quadratic_weighted_kappa(hv, jv), 4),
                "exact_match": round(float(np.mean(np.array(hv) == np.array(jv))), 4),
                "within_one": round(float(np.mean(np.abs(np.array(hv) - np.array(jv)) <= 1)), 4),
            }

        acc = agreement([h["acceptable"] for h, _ in paired], [j["acceptable"] for _, j in paired])
        pear = pearsonr(h_mean, j_mean) if len(paired) > 2 else (float("nan"),) * 2
        spear = spearmanr(h_mean, j_mean) if len(paired) > 2 else (float("nan"),) * 2

        report["annotators"][annot] = {
            "n_paired": len(paired),
            "mean_score_pearson_r": round(float(pear[0]), 4),
            "mean_score_pearson_p": round(float(pear[1]), 5),
            "mean_score_spearman_rho": round(float(spear[0]), 4),
            "judge_bias_overall": round(float(np.mean(j_mean - h_mean)), 3),
            "mean_absolute_error": round(float(np.mean(np.abs(j_mean - h_mean))), 3),
            "acceptable_agreement": acc,
            "per_dimension": per_dim,
        }

    dest = rdir / "judge_agreement.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
