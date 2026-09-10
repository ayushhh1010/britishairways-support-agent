# An AI first-line support agent for British Airways

*Hiver SDE Intern take-home. Companion documents: [`DECISIONS.md`](DECISIONS.md)
(24 non-obvious decisions), [`../golden/CODEBOOK.md`](../golden/CODEBOOK.md)
(how the evaluation set was built).*

> **Status.** Complete, except judge-vs-human agreement (§7), which needs a human
> rater; the harness is built and the command is in the README. All numbers below are
> measured on the 220-case golden set.

---

## 1. Problem framing

### The brand, chosen by measurement

I profiled all 40 highest-volume brands and compared six finalists at thread level.
`AmazonHelp` is the obvious pick on volume, and my first automated metric said its
replies were almost never deflections (0.7%). Both signals were wrong:

| | AmazonHelp | British_Airways |
|---|---|---|
| threads | 82,534 | 16,450 |
| **link rate** (deflection proxy) | **41.3%** | **6.8%** |
| **English** | 76% | **99.3%** |
| knowledge-bearing replies | 76.5% | 75.7% |
| customer-thanks rate | 10.3% | **16.0%** |

Amazon's Twitter operation exists to route customers *off* Twitter; an agent grounded in
that history learns to answer "please contact us here" to everything, and a 220-case
evaluation would conflate multilingual failure with intent failure. BA's replies contain
actual answers - *"flights under 8h30 have one main meal and a snack in World
Traveller"* - which is what a grounded generator can learn from. Hence BA, despite 5x
less volume.

### What "good" means here

1. **Never invent a policy.** A wrong baggage allowance or implied compensation promise
   is a commercial and regulatory liability. "That depends on your ticket type" is a
   *good* answer.
2. **Never auto-handle something that needed a human.** Escalating wastes an agent's
   minute; auto-handling a stranded passenger reaches a customer with a wrong answer.
   Priced 3:1, with `escalate` as the default under uncertainty.
3. **Actually answer the answerable.** BA's own replies deflect heavily; a system that
   mimics that scores well and delivers nothing.

(1) and (3) pull against each other. That tension is the product problem.

### What I chose not to build

- **Multi-turn dialogue.** The unit is first message -> first reply. Cost: follow-up
  turns, where the hardest support work happens, are never evaluated.
- **A live tool-calling agent.** No booking lookup - roughly half of what BA does needs
  account access, which is precisely why those cases are *escalated*.
- **Fine-tuning.** With 220 labels, a zero-shot model with an auditable written policy
  beats weights.
- **A silver-label intent model**, which would bake the LLM's errors into the
  "independent" baseline.
- **Banking77.** Its label space has nothing to do with an airline's.

---

## 2. What was built

```
customer message
  -> TF-IDF retrieval over 12,000 historical resolved BA cases
  -> one LLM call: intent (13 classes) + triage (auto/escalate + reason id) + reply
  -> automated metrics . LLM judge on reply quality . judge reliability probe
```

**Intents were induced, not invented.** KMeans(k=28) over 3,600 messages gave the raw
structure; clusters were merged by hand under one rule - *two messages share an intent
only if an agent would take the same next action*. Centroids are not intents: one cluster
held **25%** of all messages as a generic "flight/help" blob, another **11%** was aviation
photography. Two findings fell out:

- **~36% of BA's inbound is not a support request.** A taxonomy covering only problems
  would manufacture a fake escalation rate. Hence `praise_or_chatter`.
- **`other` had to be added during annotation**, after ~5% of golden cases proved
  unclassifiable - onboard amenities, lounge access, invoices, codeshare admin.
  Clustering missed a class hand-reading found immediately.

**Triage is a separate decision from intent**, driven by 7 escalation triggers, and
labelled for what *should* happen rather than what BA did - BA deflects answerable
questions, and copying that would reward deflection.

**The evaluation cannot leak.** `golden` (220) / `retrieval_pool` (12,000) / `dev` (400)
are disjoint by SHA-256 of `thread_id`; the retriever indexes the pool only, so a golden
case can never retrieve the reply it is scored against - asserted in code and covered by a
test. Prompts were iterated on `dev` only.

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
`B1_simple` is **trained on the golden labels** (scored out-of-fold); `agent` is
**zero-shot and never sees one**. The baseline is deliberately advantaged.

| id | intent | triage | reply |
|---|---|---|---|
| `B0_trivial` | majority class | always escalate | one fixed canned reply |
| `B1_simple` | TF-IDF + logistic regression, out-of-fold CV | keyword rules | nearest historical reply, verbatim |
| `agent` | LLM, zero-shot | LLM against the written policy | LLM, grounded in 3 retrieved cases |
| `agent_no_retrieval` | ablation - same prompt, no evidence | same | (drafts no reply) |

### Intent classification (13 classes)

| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| `B0_trivial` | 0.182 | [0.132, 0.236] | 0.024 | [0.018, 0.030] |
| `B1_simple` | 0.468 | [0.405, 0.532] | 0.395 | [0.323, 0.479] |
| `agent` | 0.768 | [0.714, 0.827] | 0.726 | [0.647, 0.790] |
| **`agent_no_retrieval`** | **0.814** | [0.759, 0.864] | **0.773** | [0.689, 0.833] |

### Triage. Cost weights false-auto-handle 3x; **lower is better**

| system | accuracy | 95% CI | esc. prec | esc. recall | false auto-handle | cost/case |
|---|---|---|---|---|---|---|
| `B0_trivial` | 0.482 | [0.414, 0.550] | 0.482 | 1.000 | 0 | 0.518 |
| `B1_simple` | 0.641 | [0.577, 0.705] | 0.696 | 0.453 | **58** | **0.886** |
| `agent` | 0.823 | [0.773, 0.868] | 0.786 | 0.868 | 14 | 0.304 |
| **`agent_no_retrieval`** | **0.868** | [0.823, 0.909] | 0.803 | 0.962 | **4** | **0.168** |

### Answer rate (of cases the system chose to auto-handle)

`B0_trivial` 0 auto-handled | `B1_simple` 151, **82.8%** answered | `agent` 103, **99.0%** answered.

### Three things worth reading twice

**1. Retrieval makes the agent WORSE - the headline finding.** The no-retrieval ablation
beats the full RAG agent on intent accuracy (0.814 vs 0.768), macro-F1 (0.773 vs 0.726),
and triage cost (0.168 vs 0.304, a 45% reduction), with 4 unsafe errors against 14. CIs
overlap, so this is suggestive, not conclusive - but the direction is consistent across
four independent metrics. The cause is in §4: median retrieval similarity is 0.247, so
the retriever supplies topically adjacent cases, not applicable precedents, and
irrelevant evidence distracts the model. Had I shipped on "the agent beats both
baselines" - which it does, comfortably - I would have shipped the worse of two systems.

**2. The simple baseline is worse than doing nothing, and only the cost metric shows
it.** `B1_simple` has higher triage *accuracy* than always-escalate (0.641 vs 0.482) and
would look like progress on any symmetric metric. But it auto-handles **58 cases that
needed a human**, giving cost/case 0.886 against the trivial system's 0.518. A keyword
rule that fires on "refund" but not on "can't check in for a flight in 8 hrs" is exactly
what the 3:1 weighting exists to catch.

**3. The agent nearly always answers when it says it will** - 99.0% against 82.8% for the
nearest-neighbour baseline, which often copies a historical BA reply that deflects to DM.

---

## 6. What is misleading about my headline number?

**I published a misleading number in this project's first hour.** My deflection regex
scored AmazonHelp at 0.7%. Reading twelve non-matching replies showed they were *all*
deflections in phrasings it never anticipated - *"give us a shout here"*, *"report this
here"*. A keyword detector over an unbounded phrasing space has an unbounded
false-negative rate. The same failure modes apply below.

**1. "The agent scores 0.768" is the wrong headline; 0.814 is.** The best configuration
is the one *without* the retrieval system I spent most effort on. Leading with the RAG
number would be selecting the architecture I wanted to justify. CIs overlap, so the
honest statement is: *retrieval did not help, and probably hurt.*

**2. Intent accuracy is inflated by the easiest class.** `praise_or_chatter` is 18% of
the set and near-trivial; a system that only learned "complaint or photo" scores well.
Macro-F1 is the honest read, and it is dragged by classes under 12 examples where it is
barely estimable. `contact_channel_request` has exactly 1.

**3. Triage labels are policy, not ground truth.** Intent has a defensible right answer.
`action` encodes *my* escalation policy; another airline could draw the line differently
and the agent would score worse without changing. The 3:1 ratio is a judgement - though
the `B1_simple`/`B0_trivial` ranking only flips at ~1.6:1, so that conclusion is not
knife-edge.

**4. Single-annotator labels.** All 220 adjudicated by one person. Blind labelling
removes anchoring but not systematic bias: where I misread the taxonomy I misread it
consistently, and the agent is scored against that.

**5. n=220, 13 classes.** ~17 cases per class, as few as 1. Differences smaller than the
quoted CIs are not differences.

**6. Batched inference is a real compromise.** The provider allows 20 requests/day/model,
so the agent processes 25 tickets per request. Each carries its own evidence and the
prompt demands independent handling, but the model *can* see other tickets. The
batched-vs-single check that would bound this was not run.

**7. The reply-quality scores rest on an unreliable instrument.** Two independent judges
agree on `acceptable` at kappa 0.08 - chance. Aggregate ordering is stable; no individual
verdict in §7 is measured. A same-family judge would have inflated the agent by +0.63,
roughly the entire gap to human agents.

**8. 2017 data.** Policies and fees have changed, so groundedness is measured against
what BA said then, not today's published policy.

---

## 7. Reply quality (LLM judge)

Judge: `openai/gpt-oss-120b` on **Groq** - a different model family *and* provider from
the Gemini generator, sharing neither training lineage nor inference stack. Blind:
candidates from all four systems are shuffled together, unlabelled. n = 45-51 per system;
failed judge calls are excluded as missing data, never scored as 1.

| system | grounded | helpful | tone | safety | mean | acceptable | beats historical |
|---|---|---|---|---|---|---|---|
| `B0_trivial` | 3.63 | 2.28 | 3.59 | **4.80** | 3.58 | 30.4% | 17.4% |
| `B1_simple` | 2.57 | 2.02 | 3.33 | 5.00 | 3.23 | 25.5% | 9.8% |
| `agent` | 4.20 | 3.06 | 3.94 | 4.76 | **3.99** | **60.0%** | 24.0% |
| `historical` (real BA agents) | 4.69 | 3.38 | 4.20 | 5.00 | **4.32** | 62.2% | - |

**The agent does not beat human agents**: 3.99 vs 4.32, preferred in only 24% of
head-to-heads. The gap narrows on the operational decision - *acceptable to send
unedited* - at 60.0% vs 62.2%.

**The trivial baseline exposes the rubric.** One fixed sentence for all 220 cases scores
**4.80 on safety** and 3.59 on tone: a reply that commits to nothing cannot be unsafe or
off-tone. Only `helpfulness` (2.28) and answer rate catch it. A headline built on the
mean of these four dimensions would rate a system that answers nothing at 3.58/5.

`historical` scoring highest is the control that validates the judge. An earlier
configuration returned 1.13/5 for real human replies; that was a broken run (93% of calls
failing, averaged in as 1s), discarded rather than reported.

### How much should you trust that table? Less than it looks.

Human ratings were not obtained (below), so I measured **judge-vs-judge reliability**: a
second judge, `gemini-3.5-flash` on Google's stack, re-graded 30 cases.

**Self-preference bias is real and now quantified.** The probe shares a family with the
generator; the primary does not.

| | primary judge (Groq) | probe (generator's family) | gap |
|---|---|---|---|
| `agent` replies | 4.058 | 4.567 | **+0.51** |
| `historical` (control) | 4.421 | 4.303 | -0.12 |

The probe is not a softer marker - it grades the *historical* control **lower**. It
favours its own family by **+0.628 points on a 5-point scale**, roughly the entire gap
between the agent and human agents. A same-family judge - the convenient default - would
have shown false parity with humans.

**Per-case reliability is poor.** On `acceptable`, the decision that would gate an
auto-send: kappa **0.08** (agent, n=26) and **-0.02** (historical, n=19) - chance. Per
dimension QWK: groundedness 0.42/0.61, helpfulness 0.38/0.62, tone 0.30/0.29, safety
**-0.04**/undefined. Mean absolute difference per case is 0.72 points.

**What survives:** aggregate *ordering* is stable - both judges rank `historical` highest
and separate the agent from the baselines - so the ranking above is probably sound. No
per-case claim survives: "this reply is acceptable" is not a verdict this judge can
support, and 60.0% should be read as indicative, not measured.

**This is reliability, not validity.** Two judges agreeing would not make them right. The
real validation is judge-vs-human agreement, which needs a human rater
(`python label.py replies`, then `scripts/judge_agreement.py`). The harness is built and
tested; ratings are not collected. A first attempt produced nine ratings with internal
contradictions (all-1 scores marked sendable) and was discarded. I did not supply the
ratings myself: I wrote both the judge rubric and the agent prompts, so they would have
validated my own instrument against itself.

---

## 8. Failure analysis: the top 5 modes

**1. Asks for personal data in public - a contradiction in my own prompt.**
Hit **3 of the 5 worst-scoring replies** (safety = 1):

> *"going to Dubai for our honeymoon & seat allocation has separated our booking!"*
> -> *"...Please DM your booking reference and email address..."* -> judge: *"asks for personal data publicly"*

I checked, and it is my bug. The escalate rule says *"ask them to DM the details a
colleague needs (booking reference, flight number)"*; hard constraint #5 says *"never ask
for a booking reference in the public reply"*. Both are in the same prompt. The fix is
removing the conflict, not adding instruction: the escalate rule should say *"invite them
to DM us"* and never enumerate what to send. One line, not re-run.

**2. Check-in failures get auto-handled - 6 of 14 unsafe triage errors.**

> *"can't check in online for Dubai Heathrow club world flight in 8 hrs"*
> *"app says 'still waiting for authorisation from the US gov'"*

A broken check-in reads like a generic app fault with a plausible generic answer, but it
is account-specific *and* time-critical - the customer flies within hours. The agent sees
a tech-support surface and misses the clock. Encoding "mentions an imminent flight" as a
hard escalation trigger, rather than leaving it to model judgement, should fix most.

**3. `flight_disruption` misread as `service_complaint` - largest intent confusion (5).**
An angry tweet about a delay is textually a complaint; the taxonomy assigns by *subject*
(the disruption), not tone. The pre-label models got the same boundary wrong. If two
strong models and a human all disagree on a boundary, the boundary is the problem - this
is a taxonomy-design cost, not purely model error.

**4. `other` is unlearnable - the ablation's top confusion (8 cases).**
`other` -> `service_complaint` (4), `other` -> `booking_change_cancel` (4). It was added
*during* annotation (DECISIONS #23) and is defined negatively - "a real support request
nothing else covers". A model cannot match a positive prototype that does not exist, and
at 11 examples it is the thinnest real class. It needs positive sub-definitions (onboard
product, lounge, invoices, codeshare admin) or those should become their own intents.

**5. Confident filler where the retrieval pool has no precedent.**

> *"staff travel tickets issued through MyIDTravel"* -> groundedness **1**
> *"UK agent here - system issue pre-booking seats with AA codeshare?"* -> *"generic advice, not specific"*

B2B and edge queries - trade agents, staff travel, invoices - have no analogue in 12,000
consumer tweets. Retrieval returns loosely-related consumer cases and the agent produces
confident-sounding filler. Same root cause as §5: at similarity 0.247, evidence is
decoration. A retriever that could **abstain** below a similarity floor, telling the model
it has no precedent, would beat one that always returns three rows.

---

## 9. What I'd do next with one more week

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
