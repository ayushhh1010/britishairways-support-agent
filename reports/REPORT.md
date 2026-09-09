# An AI first-line support agent for British Airways

*Hiver SDE Intern take-home. Companion documents: [`DECISIONS.md`](DECISIONS.md)
(24 non-obvious decisions), [`../golden/CODEBOOK.md`](../golden/CODEBOOK.md)
(how the evaluation set was built).*

> **Status.** Sections 1-6 are complete and measured on all 220 golden cases, with
> zero failed cases in the agent and ablation runs. The LLM-judge reply-quality table
> (§7) is the one thing outstanding: the free-tier provider allows 20 requests/day per
> model and I spent that budget getting the main run right. It needs ~12 requests on
> an untouched model and one command; see §7.

---

## 1. Problem framing

### The brand, chosen by measurement

I profiled all 40 highest-volume brands and compared six finalists at thread level
before committing. `AmazonHelp` is the obvious pick on volume - 82,534 reconstructed
threads against BA's 16,450 - and my first automated metric said its replies were
almost never deflections (0.7%).

Both signals were wrong in the way that matters:

| | AmazonHelp | British_Airways |
|---|---|---|
| threads | 82,534 | 16,450 |
| **link rate** (deflection proxy) | **41.3%** | **6.8%** |
| **English** | 76% | **99.3%** |
| knowledge-bearing replies | 76.5% | 75.7% |
| customer-thanks rate | 10.3% | **16.0%** |

Amazon's Twitter operation exists to route customers *off* Twitter. An agent grounded
in that history learns to answer "please contact us here" to everything, and a
220-case evaluation would conflate multilingual failure with intent failure. BA's
replies contain actual answers - *"flights under 8 hours 30 have one main meal and a
snack in World Traveller"* - which is the substance a grounded generator can learn
from, and why BA wins despite being 5x smaller.

### What "good" means for this brand

1. **Never invent a policy.** A wrong baggage allowance or an implied compensation
   promise is a commercial and regulatory liability. "That depends on your ticket
   type" is a *good* answer.
2. **Never auto-handle something that needed a human.** Escalating unnecessarily
   costs an agent a minute; auto-handling a stranded passenger or a compensation
   claim reaches a real customer with a wrong answer. Priced 3:1, with `escalate` as
   the default under uncertainty.
3. **Actually answer the answerable.** BA's own replies deflect heavily. A system
   that mimics that scores well and delivers nothing, which is why answer rate is
   reported beside the quality scores.

(1) and (3) pull against each other. That tension is the product problem.

### What I chose not to build

- **Multi-turn dialogue.** The unit is the first customer message and BA's first
  reply. Cost: follow-up turns, where the hardest support work happens, are never
  evaluated.
- **A live tool-calling agent.** No booking lookup, no flight-status API. Roughly
  half of what BA does needs account access - which is precisely why those cases are
  *escalated* rather than answered.
- **Fine-tuning.** With 220 labels, a zero-shot model with a written, auditable
  policy beats a model fine-tuned on a set this size.
- **A silver-label intent model**, which would bake the LLM's errors into the
  "independent" baseline.
- **Banking77.** Its label space has nothing to do with an airline's.

---

## 2. What was built

```
customer message
  -> TF-IDF retrieval over 12,000 historical resolved BA cases
  -> one LLM call: intent (13 classes) + triage (auto/escalate + reason id) + reply
  -> automated metrics . LLM judge on reply quality . judge validated against a human
```

**Intents were induced, not invented.** KMeans(k=28) over 3,600 messages gave the raw
structure; clusters were merged by hand under one rule - *two messages share an intent
only if an agent would take the same next action*. Centroids are not intents: one
cluster held **25%** of all messages as a generic "flight/help" blob and another
**11%** was aviation-enthusiast photography.

Two findings came out of that:

- **~36% of BA's inbound is not a support request.** A taxonomy covering only problems
  would manufacture a fake escalation rate. Hence `praise_or_chatter`.
- **`other` had to be added during annotation.** Reading all 220 golden messages
  surfaced ~5% that are real support requests no intent covered - onboard amenities,
  lounge access, invoices, codeshare admin. Clustering missed a class hand-reading
  found immediately.

**Triage is a separate decision from intent**, driven by 7 escalation triggers. Labels
record what *should* happen, not what BA did - BA deflects answerable questions for
its own reasons, and copying that would build an evaluation that rewards deflection.

**The evaluation cannot leak.** `golden` (220) / `retrieval_pool` (12,000) / `dev`
(400) are disjoint at thread level by SHA-256 of `thread_id`. The retriever indexes
the pool only, so a golden case can never retrieve the reply it is scored against -
asserted in code and covered by a test. Prompts were iterated on `dev` only.

---

## 3. The golden set

220 first-contact cases, **100% author-adjudicated, every label written blind** -
before any model pre-label was read. Protocol in [`CODEBOOK.md`](../golden/CODEBOOK.md).

| intent | n | | intent | n |
|---|---|---|---|---|
| praise_or_chatter | 40 | | baggage_lost_damaged | 12 |
| service_complaint | 30 | | seating | 11 |
| flight_disruption | 24 | | other | 11 |
| digital_technical | 21 | | loyalty_avios | 10 |
| booking_change_cancel | 20 | | baggage_policy | 9 |
| checkin_boarding | 19 | | contact_channel_request | 1 |
| compensation_refund | 12 | | | |

**Escalate base rate 48.2%** (106/220). Sampling is **uniform, not stratified**, so
headline accuracy reflects BA's real traffic mix. The cost is visible above:
`contact_channel_request` has one example and its per-class metrics are meaningless.

---

## 4. Annotation quality

Blind human labels vs `gpt-oss-120b` pre-labels, n=219:

| | Cohen's kappa | raw agreement |
|---|---|---|
| intent (all) | **0.735** - substantial | 76.3% |
| intent (excl. `other`) | 0.779 | 80.3% |
| triage action | **0.482** - moderate | 74.4% |

Of 56 triage disagreements, **48 are model-says-auto-handle where I say escalate** - a
6:1 asymmetry entirely in the unsafe direction, visible before the agent ran.

**Retrieval quality:** every case retrieves evidence, but median top-similarity is
only **0.247** and just **5.5%** exceed 0.40. Grounding is frequently *topically
adjacent* rather than case-specific. This turns out to matter enormously (§5).

---

## 5. Results vs baselines

All 220 cases, every system scored identically, **zero failed cases**.

| id | intent | triage | reply |
|---|---|---|---|
| `B0_trivial` | majority class | always escalate | one fixed canned reply |
| `B1_simple` | TF-IDF + logistic regression, **out-of-fold CV on golden labels** | keyword rules | nearest historical reply, verbatim |
| `agent` | LLM, zero-shot | LLM against the written policy | LLM, grounded in 3 retrieved cases |
| `agent_no_retrieval` | ablation - same prompt, no evidence | same | (drafts no reply) |

`B1_simple` is **trained on the golden labels**; `agent` is **zero-shot and never sees
one**. The baseline is deliberately advantaged.

### Intent classification (13 classes)

| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| `B0_trivial` | 0.182 | [0.132, 0.236] | 0.024 | [0.018, 0.030] |
| `B1_simple` | 0.468 | [0.405, 0.532] | 0.395 | [0.323, 0.479] |
| `agent` | 0.768 | [0.714, 0.827] | 0.726 | [0.647, 0.790] |
| **`agent_no_retrieval`** | **0.814** | [0.759, 0.864] | **0.773** | [0.689, 0.833] |

### Triage (escalate vs auto-handle)

Cost weights false-auto-handle 3x false-escalate. **Lower cost/case is better.**

| system | accuracy | 95% CI | esc. precision | esc. recall | false auto-handle | cost/case |
|---|---|---|---|---|---|---|
| `B0_trivial` | 0.482 | [0.414, 0.550] | 0.482 | 1.000 | 0 | 0.518 |
| `B1_simple` | 0.641 | [0.577, 0.705] | 0.696 | 0.453 | **58** | **0.886** |
| `agent` | 0.823 | [0.773, 0.868] | 0.786 | 0.868 | 14 | 0.304 |
| **`agent_no_retrieval`** | **0.868** | [0.823, 0.909] | 0.803 | 0.962 | **4** | **0.168** |

### Answer rate (of cases the system chose to auto-handle)

| system | auto-handled | answered | deflected |
|---|---|---|---|
| `B0_trivial` | 0 | - | - |
| `B1_simple` | 151 | 82.8% | 17.2% |
| `agent` | 103 | **99.0%** | 1.0% |

### Three things worth reading twice

**1. Retrieval makes the agent WORSE, and that is the headline finding.** The
no-retrieval ablation beats the full RAG agent on intent accuracy (0.814 vs 0.768),
macro-F1 (0.773 vs 0.726), and triage cost per case (0.168 vs 0.304 - a 45%
reduction), and makes 4 unsafe errors against 14. The confidence intervals overlap,
so this is suggestive rather than conclusive, but the direction is consistent across
four independent metrics.

The explanation is in §4: median retrieval similarity is 0.247. The retriever supplies
*topically adjacent* cases, not applicable precedents, and irrelevant evidence
actively distracts the model. The single most valuable thing this evaluation produced
is the discovery that the most architecturally elaborate component is net-negative.
Had I shipped the RAG agent on the strength of it beating the baselines - which it
does, comfortably - I would have shipped a worse system than the simpler one.

**2. The simple baseline is worse than doing nothing, and only the cost metric shows
it.** `B1_simple` has higher triage *accuracy* than always-escalate (0.641 vs 0.482)
and would look like an improvement on any symmetric metric. But it auto-handles **58
cases that needed a human**, giving it a cost per case of 0.886 - worse than the
trivial system that escalates everything (0.518). A keyword rule that fires on
"refund" but not on "can't check in for a flight in 8 hrs" is exactly the failure the
3:1 weighting exists to catch.

**3. The agent nearly always answers when it says it will** (99.0%), against 82.8% for
the nearest-neighbour baseline, which frequently copies a historical BA reply that
deflects to DM.

---

## 6. What is misleading about my headline number?

**I published a misleading number in this project's first hour.** My deflection regex
scored AmazonHelp at 0.7%. Reading twelve non-matching replies showed they were *all*
deflections in phrasings the regex never anticipated - *"give us a shout here"*,
*"report this here"*. A keyword detector over an unbounded phrasing space has an
unbounded false-negative rate. The same failure modes apply below.

**1. "The agent scores 0.768" is the wrong headline; 0.814 is.** The best configuration
I measured is the one *without* the retrieval system I spent the most effort on. Any
report that led with the RAG number would be selecting the architecture it wanted to
justify. The CIs overlap, so the honest statement is: *retrieval did not help, and
probably hurt.*

**2. Intent accuracy is inflated by the easiest class.** `praise_or_chatter` is 18% of
the set and near-trivial. A system that only learned "complaint or photo" scores well.
Macro-F1 is the honest read, and it is dragged by classes with under 12 examples where
it is barely estimable. `contact_channel_request` has exactly 1 example.

**3. Triage labels are policy, not ground truth.** Intent has a defensible right
answer. `action` encodes *my* escalation policy; another airline could draw the line
differently and the agent would score worse without changing. The 3:1 cost ratio is a
judgement, not a measured quantity - though note the ranking between `B1_simple` and
`B0_trivial` flips at roughly 1.6:1, so that specific conclusion is not knife-edge.

**4. Single-annotator labels.** All 220 were adjudicated by one person. Blind labelling
removes anchoring but not systematic bias: where I misread the taxonomy I misread it
consistently, and the agent is scored against that. A second annotator would bound
this properly; the labelling CLI supports it and it was not run.

**5. n=220, 13 classes.** ~17 cases per class, as few as 1 in the tail. Differences
smaller than the quoted CIs are not differences.

**6. Batched inference is a real methodological compromise.** The provider allows 20
requests/day/model, so the agent processes 25 tickets per request. The prompt
instructs independent handling and each ticket carries its own evidence, but the model
*can* see other tickets. I did not get to run the batched-vs-single agreement check
that would bound this; it is the first thing I would do next.

**7. 2017 data.** Policies and fees have changed, so groundedness is measured against
what BA said then, not today's published policy.

---

## 7. Reply quality (outstanding)

The harness is built, tested, and wired: `ReplyJudge` grades four candidates per case
(both baselines, the agent, and BA's real historical reply) on groundedness,
helpfulness, tone, and safety, blind to which system wrote each. The judge is
**Gemma**, a different model family from the **Gemini** generator, so it never grades
its own family's output; `scripts/judge_bias.py` re-grades a subsample with a
same-family judge to *measure* self-preference rather than assume it.

It needs ~12 requests on a model whose daily budget is untouched:

```bash
python scripts/run_eval.py --ablation --judge-n 60
```

One early observation already recorded: in a smoke run, `B0_trivial` - a fixed
*"sorry, please DM us"* - scored **4.50/5 with 100% "acceptable"**. Nothing in it can
be wrong, so groundedness and safety score full marks. That is a defect in the rubric,
and it is exactly why answer rate is reported alongside it.

---

## 8. What I'd do next with one more week

1. **Kill or fix retrieval.** The ablation says it hurts. Either measure retrieval
   directly (label whether top-k contains an applicable precedent) and fix it with
   hybrid BM25 + embeddings, or drop it from the classification path and use it only
   for reply grounding, where it may still earn its place.
2. **Batched-vs-single agreement check.** Bounds the §6.6 compromise. ~15 requests.
3. **A second annotator on a 50-case blind slice.** The largest remaining weakness.
4. **Fix the judge's non-answer blind spot** with an explicit "does this resolve it?"
   dimension, re-validated against human ratings.
5. **Confidence-gated auto-send.** The agent emits `intent_confidence` and nothing
   consumes it. Calibrate it, then auto-send only above a threshold chosen so
   false-auto-handle stays inside a stated budget - turning the cost metric into an
   operating policy.
6. **A prompt-injection test set.** The prompt says never to obey instructions inside
   customer messages; that claim is untested, and public tweets are an obvious
   injection surface.
