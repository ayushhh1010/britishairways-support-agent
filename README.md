# British Airways support agent — Hiver SDE Intern take-home

An AI first-line support agent for **British Airways**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
corpus (2.8M tweets). For each incoming customer message it:

1. **classifies** it into one of **13 intents** induced from the data,
2. **drafts a reply** grounded in how BA historically resolved similar cases,
3. **decides auto-handle vs escalate**, with a machine-readable reason.

The agent is the easy part. The evidence that it works is the point — start with
[`reports/REPORT.md`](reports/REPORT.md), especially §6 *"What is misleading about my
headline number?"*.

---

## Reproduce the headline results — no API key needed

Every LLM response is cached in `llm_cache/` (committed, keyed by SHA-256 of the
request), and `data/processed/` is committed too.

```bash
pip install -r requirements.txt
make repro
```

**Verified: 213 seconds with the `.env` file removed entirely**, producing numbers
identical to those below. You do not need a key, an account, or the 500 MB raw
dataset to check any claim in the report.

Windows: if your console is not UTF-8, set `PYTHONIOENCODING=utf-8`. The `Makefile`
already does.

<details>
<summary>Without <code>make</code></summary>

```bash
set PYTHONPATH=src                      # Windows;  export PYTHONPATH=src  on POSIX
python scripts/run_eval.py --ablation --judge-n 60
python scripts/failure_analysis.py
```
</details>

---

## Headline results (220 golden cases, zero failed)

| system | intent acc | macro-F1 | triage cost/case ↓ | false auto-handle | judge mean |
|---|---|---|---|---|---|
| `B0_trivial` | 0.182 | 0.024 | 0.518 | 0 | 3.58 |
| `B1_simple` *(trained on the labels)* | 0.468 | 0.395 | 0.886 | 58 | 3.23 |
| `agent` *(zero-shot)* | 0.768 | 0.726 | 0.304 | 14 | **3.99** |
| **`agent_no_retrieval`** *(ablation)* | **0.814** | **0.773** | **0.168** | **4** | – |
| `historical` *(real BA agents)* | – | – | – | – | **4.32** |

Three findings:

- **Retrieval makes the agent worse.** The no-retrieval ablation wins on all four
  automated measures. Median retrieval similarity is 0.247 — the retriever returns
  *topically adjacent* cases, not applicable precedents. Shipping on "the agent beats
  both baselines" would have shipped the worse of two systems.
- **`B1_simple` is worse than doing nothing**, and only the asymmetric cost metric
  shows it: higher triage accuracy than always-escalate (0.641 vs 0.482) but a worse
  cost per case (0.886 vs 0.518), because it auto-handles 58 cases needing a human.
- **The agent does not beat human agents** (3.99 vs 4.32), though it approaches them
  on send-unedited acceptability (60.0% vs 62.2%).

---

## Known gap

**Judge-vs-human agreement is not measured.** The assignment asks for it; the harness
is built but needs a human rater:

```bash
python -m support_agent.labeling.cli replies --annotator <name> --limit 40
python scripts/judge_agreement.py
```

Until that is run, the reply-quality column rests on an unvalidated instrument. That
caveat is not hypothetical: two earlier judge configurations produced confidently
wrong output, one of which is documented in `DECISIONS.md` #26.

---

## What is in here

| path | what it is |
|---|---|
| `src/support_agent/data/threads.py` | union-find thread reconstruction over the 2.8M-row edge list |
| `src/support_agent/data/dataset.py` | case filtering + hash-based, leak-free splits |
| `configs/taxonomy.yaml` | 13 intents + 7 escalation triggers, with boundary rules |
| `src/support_agent/retrieve.py` | TF-IDF retriever over 12k historical resolved cases |
| `src/support_agent/agent.py` | the agent: retrieve → classify → triage → draft |
| `src/support_agent/baselines.py` | B0 trivial and B1 simple |
| `src/support_agent/evaluation/judge.py` | LLM-as-judge: cross-provider, blind to system identity |
| `src/support_agent/evaluation/metrics.py` | bootstrap CIs, Cohen's κ, cost-weighted triage, answer rate |
| `src/support_agent/labeling/cli.py` | terminal labelling tool (resumable, blind mode) |
| `golden/CODEBOOK.md` | how the golden set was sampled and labelled |
| `golden/golden_set.jsonl` | **220 cases, 100% author-adjudicated, labelled blind** |
| `reports/REPORT.md` | the report |
| `reports/DECISIONS.md` | 28 non-obvious decisions and why |
| `reports/results/failures.md` | mined failure material behind report §8 |
| `tests/` | 26 tests, focused on where a bug produces a plausible number |

---

## Design in one screen

**Brand choice was measured.** All 40 top brands profiled, 6 finalists compared at
thread level. `AmazonHelp` has 5x the volume but a **41.3% link rate** (its Twitter
strategy is to route customers to a web form) and is only 76% English. BA has a
**6.8% link rate**, is 99.3% English, and its replies carry real policy content.

**Intents came from clustering, then human merging.** KMeans(k=28) over 3,600
messages; clusters merged by *"same intent only if an agent would take the same next
action"*. One cluster was a 25% catch-all, another 11% aviation photography. A 13th
intent, `other`, was added *during* annotation after ~5% of cases proved
unclassifiable.

**Triage is independent of intent** and labelled for what *should* happen, not what BA
did — BA deflects answerable questions, and copying that would reward deflection.

**The eval cannot leak.** `golden` (220) / `retrieval_pool` (12,000) / `dev` (400) are
disjoint by SHA-256 of `thread_id`; the retriever indexes the pool only. Asserted in
code and covered by a test. Prompts were iterated on `dev` only.

**`B1_simple` is deliberately advantaged** — trained on the golden labels (scored
out-of-fold), while the agent is zero-shot and never sees one.

---

## Models and cost

Total spend: **£0**. Everything runs on free tiers.

| role | model | provider | why |
|---|---|---|---|
| generator | `gemini-3.5-flash-lite` | Google AI Studio | free tier that could complete the run |
| judge | `openai/gpt-oss-120b` | **Groq** | different family *and* different provider → no shared lineage or stack |
| bias probe | `gemini-3.5-flash` | Google AI Studio | same family as generator, to *measure* self-preference |

Free tiers cap at **20 requests/day/model**, which shaped real engineering: the agent
processes **25 tickets per request** (220 cases in 9 requests), responses are cached,
and a tokens-per-day 429 fails fast rather than retrying for hours. See `DECISIONS.md`
#22 and #27 — including that batching the system under test is a disclosed compromise.

Swap providers via `.env` (`LLM_PROVIDER`, `GEN_MODEL`, `JUDGE_MODEL`); the client also
speaks Groq, Cerebras, Ollama, and any OpenAI-compatible endpoint.

<details>
<summary>Rebuild everything from raw data (needs keys)</summary>

```bash
make data        # download + unzip the corpus (~169 MB), needs ~/.kaggle/kaggle.json
make prep        # reconstruct threads, build leak-free splits
make taxonomy    # cluster messages for taxonomy induction
make prelabel    # dual-model pre-labelling of the golden set
make repro       # evaluate
make status      # pipeline progress + live per-model quota
make resume      # continue across daily quota windows
```
</details>

---

## Attribution

- Dataset: `thoughtvector/customer-support-on-twitter` (Kaggle).
- **scikit-learn**: TF-IDF, KMeans, logistic regression, stratified k-fold, Cohen's κ,
  quadratic-weighted κ. Percentile bootstrap CIs implemented here
  (`evaluation/metrics.py`), standard method.
- κ interpretation bands: Landis & Koch (1977), used as coarse labels only.
- The escalation-cost asymmetry, rubric dimensions, taxonomy, prompts, answer-rate
  metric, and evaluation design are mine.
- Built with AI coding assistance; every design decision is recorded with its
  reasoning in `reports/DECISIONS.md`, including the ones I got wrong and reversed.
