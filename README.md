# British Airways support agent — Hiver SDE Intern take-home

An AI first-line support agent for **British Airways** built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
corpus (2.8M tweets). For each incoming customer message it:

1. **classifies** it into one of 12 intents induced from the data,
2. **drafts a reply** grounded in how BA historically resolved similar cases,
3. **decides auto-handle vs escalate**, with a stated machine-readable reason.

The interesting part is not the agent — it is the evidence that the agent works.
See [`reports/REPORT.md`](reports/REPORT.md), and in particular the section
*"What is misleading about my headline number?"*.

---

## Reproduce the headline results in under 15 minutes

Every LLM response is cached in `llm_cache/` (committed, keyed by SHA-256 of the
request), and `data/processed/` is committed too. **A fresh clone reproduces the
exact numbers offline, with no API key and no spend.**

```bash
pip install -r requirements.txt
make repro
```

That runs the evaluation, the failure analysis, and the judge-agreement report,
then prints the headline table. Expect ~2 minutes, almost all of it scikit-learn.

Windows note: this repo is developed on Windows. If your console is not UTF-8, set
`PYTHONIOENCODING=utf-8` — the `Makefile` already does.

### Without `make`

```bash
set PYTHONPATH=src                      # Windows;  export PYTHONPATH=src  on POSIX
python scripts/run_eval.py --ablation
python scripts/failure_analysis.py
python scripts/judge_agreement.py
```

### Rebuild everything from raw data

Needs a Kaggle API token in `~/.kaggle/kaggle.json` and a Groq key in `.env`
(copy `.env.example`; free, no card, <https://console.groq.com/keys>).

```bash
make data        # download + unzip the corpus (~169 MB)
make prep        # reconstruct threads, build leak-free splits
make taxonomy    # cluster messages for taxonomy induction
make prelabel    # dual-model pre-labelling of the golden set   (LLM, ~40 min)
make golden      # assemble golden_set.jsonl from adjudications
make repro       # evaluate                                      (LLM, ~2.5 h cold)
```

Cold runs are slow **only** because the Groq free tier caps at 8,000 tokens/minute.
With the shipped cache none of this is needed.

---

## What is in here

| path | what it is |
|---|---|
| `src/support_agent/data/threads.py` | union-find thread reconstruction over the 2.8M-row edge list |
| `src/support_agent/data/dataset.py` | case filtering + hash-based, leak-free splits |
| `configs/taxonomy.yaml` | 12 intents + 7 escalation triggers, with boundary rules |
| `src/support_agent/retrieve.py` | TF-IDF retriever over 12k historical resolved cases |
| `src/support_agent/agent.py` | the agent: retrieve → classify → triage → draft, one LLM call |
| `src/support_agent/baselines.py` | B0 trivial and B1 simple |
| `src/support_agent/evaluation/judge.py` | LLM-as-judge, cross-family, blind to system identity |
| `src/support_agent/evaluation/metrics.py` | metrics with bootstrap CIs, Cohen's κ, quadratic-weighted κ |
| `src/support_agent/labeling/cli.py` | terminal labelling tool (resumable, blind mode) |
| `golden/CODEBOOK.md` | how the golden set was sampled and labelled |
| `reports/REPORT.md` | the report |
| `reports/DECISIONS.md` | 20 non-obvious decisions and why |

---

## Design in one screen

**Brand choice was measured, not assumed.** All 40 top brands were profiled and 6
finalists compared at thread level. `AmazonHelp` has 5x the volume but a **41.3%
link rate** (its Twitter strategy is to route customers to a web form) and is only
76% English. BA has a **6.8% link rate**, is 99.3% English, and its replies carry
real policy content — which is what a grounded reply generator can actually learn
from. Full reasoning: `reports/DECISIONS.md` #1–#2.

**Intents came from clustering, then human merging.** KMeans(k=28) over 3,600
messages gave the structure; centroids are not intents, so clusters were merged by
the rule *"same intent only if an agent would take the same next action"*. One
cluster was a 25% catch-all and another was 11% aviation photography — both had to
be split or absorbed by hand.

**Triage is independent of intent** and is labelled for what *should* happen, not
what BA did. BA deflects answerable questions to DM for its own reasons; copying
that would build an evaluation that rewards deflection.

**The eval is leak-free by construction.** `golden` (220) / `retrieval_pool`
(12,000) / `dev` (400) are disjoint at thread level via SHA-256 of `thread_id`. The
retriever is built only from the pool, so a golden case can never retrieve the reply
it is scored against — asserted in code, not assumed.

**The judge is a different model family from the generator**, and a same-family
judge is run on a subsample purely to quantify self-preference bias. The judge's
agreement with a human is measured and reported; an unvalidated judge is decoration.

---

## Systems compared

| id | intent | triage | reply |
|---|---|---|---|
| `B0_trivial` | majority class | always escalate | one fixed canned reply |
| `B1_simple` | TF-IDF + logistic regression, **out-of-fold CV on the golden labels** | keyword rules | nearest historical reply, copied verbatim |
| `agent` | LLM, zero-shot | LLM against the written policy | LLM, grounded in 4 retrieved cases |
| `agent_no_retrieval` | ablation — same prompt, no evidence | | (no reply drafted) |
| `historical` | — | — | what BA actually replied, scored by the same judge |

`B1_simple` is **trained on the golden labels** while `agent` is zero-shot. The
baseline is deliberately advantaged; see `DECISIONS.md` #12.

---

## Cost and models

Everything runs on the **Groq free tier**. Total spend: **£0**.

| role | model | why |
|---|---|---|
| generator | `openai/gpt-oss-120b` | strongest chat model available on the account |
| judge | `qwen/qwen3.8-27b` | different family → no self-preference |
| bias probe | `openai/gpt-oss-120b` | same family as generator, to *measure* the bias |

The free tier caps at 8,000 tokens/minute per model, which shaped real engineering:
a token-reserving rolling-window limiter, a compact taxonomy rendering (~530 tokens
vs ~2,000), and `reasoning_effort: low`. See `DECISIONS.md` #16.

Swap providers with `LLM_PROVIDER` / `GEN_MODEL` / `JUDGE_MODEL` in `.env`; the
client also speaks Ollama and any OpenAI-compatible endpoint.

---

## Attribution

- Dataset: `thoughtvector/customer-support-on-twitter` (Kaggle).
- Cohen's κ, quadratic-weighted κ, TF-IDF, KMeans, logistic regression, stratified
  k-fold: **scikit-learn**. Percentile bootstrap CIs: implemented here
  (`evaluation/metrics.py`), standard method.
- κ interpretation bands: Landis & Koch (1977), used as coarse labels only.
- Escalation-cost asymmetry, the rubric dimensions, the taxonomy, all prompts, and
  the evaluation design are mine.
- Built with AI coding assistance; every design decision is recorded with its
  reasoning in `reports/DECISIONS.md`.
