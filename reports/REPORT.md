# An AI first-line support agent for British Airways

*Hiver SDE Intern take-home. Companion documents: [`DECISIONS.md`](DECISIONS.md)
(24 non-obvious decisions), [`../golden/CODEBOOK.md`](../golden/CODEBOOK.md)
(how the evaluation set was built).*

> **Status.** Complete. Every number below is measured on the 220-case golden set
> (agent and ablation: zero failed cases; judge: n=45-51 per system, sampled). The one
> deliverable I could not produce alone is judge-vs-human agreement (§7), which needs
> a human rater; the tool is built and the command is in the README.

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

**7. The reply-quality scores rest on an unreliable instrument.** Two independent
judges agree on `acceptable` at kappa 0.08 - chance. The aggregate ordering is stable,
but no individual verdict in §7 should be treated as measured. And a same-family judge
would have inflated the agent by +0.63 points, which is roughly the entire gap between
the agent and human agents.

**8. 2017 data.** Policies and fees have changed, so groundedness is measured against
what BA said then, not today's published policy.

---

## 7. Reply quality (LLM judge)

Judge: `openai/gpt-oss-120b` on **Groq** - a different model family *and* a different
provider from the Gemini generator, so it shares neither training lineage nor
inference stack. It grades blind: candidates from all four systems are shuffled
together and carry no system label. n = 45-51 per system (60 cases sampled; failed
judge calls are excluded as missing data, never scored as 1).

| system | grounded | helpful | tone | safety | mean | acceptable | beats historical |
|---|---|---|---|---|---|---|---|
| `B0_trivial` | 3.63 | 2.28 | 3.59 | **4.80** | 3.58 | 30.4% | 17.4% |
| `B1_simple` | 2.57 | 2.02 | 3.33 | 5.00 | 3.23 | 25.5% | 9.8% |
| `agent` | 4.20 | 3.06 | 3.94 | 4.76 | **3.99** | **60.0%** | 24.0% |
| `historical` (real BA agents) | 4.69 | 3.38 | 4.20 | 5.00 | **4.32** | 62.2% | - |

**The agent does not beat human agents, and that is the honest headline.** It scores
3.99 against BA's real replies at 4.32, and the judge preferred the agent in only 24%
of head-to-heads. The gap is smaller on the decision that matters operationally -
*acceptable to send unedited* - at 60.0% vs 62.2%.

**The trivial baseline exposes the rubric.** `B0_trivial` is a single fixed sentence,
identical for all 220 cases. It scores **4.80 on safety** and 3.59 on tone, because a
reply that commits to nothing cannot be unsafe and cannot be off-tone. Only
`helpfulness` (2.28) and the separate answer-rate metric catch it. Any evaluation
using a mean of these four dimensions as its headline would rate a system that
answers nothing at 3.58/5 - which is why answer rate is reported alongside, and why
the mean is not the headline.

`historical` scoring highest is also the sanity check that validates the judge. An
earlier judge configuration returned 1.13/5 for real human BA replies; that was not a
finding, it was a broken run (93% of calls failing and being averaged in as 1s), and
it was discarded rather than reported.

### How much should you trust that table? Less than it looks.

I could not obtain independent human ratings (see below), so I measured what I could:
**judge-vs-judge reliability**. A second judge - `gemini-3.5-flash`, on Google's stack -
re-graded 30 cases already graded by the primary `gpt-oss-120b` judge on Groq.

**Self-preference bias is real and now quantified.** The probe judge shares a model
family with the generator; the primary does not.

| | primary judge (Groq) | probe judge (generator's family) | gap |
|---|---|---|---|
| `agent` replies | 4.058 | 4.567 | **+0.51** |
| `historical` (control) | 4.421 | 4.303 | -0.12 |

The probe is not simply a softer marker: it grades the *historical* control slightly
*lower*. It is specifically generous to output from its own family, by
**+0.628 points on a 5-point scale**. Had I judged with a same-family model - the
convenient default - the agent's 3.99 would have read closer to 4.6, and the gap to
human replies would have vanished as an artefact.

**Per-case reliability is poor.** On the binary decision that would actually gate an
auto-send:

| | agent (n=26) | historical (n=19) |
|---|---|---|
| `acceptable` raw agreement | 61.5% | 52.6% |
| `acceptable` Cohen's kappa | **0.08** (slight) | **-0.02** (worse than chance) |
| groundedness QWK | 0.42 | 0.61 |
| helpfulness QWK | 0.38 | 0.62 |
| tone QWK | 0.30 | 0.29 |
| safety QWK | **-0.04** | undefined (near-constant) |

Two independent judges agree on *whether a reply is sendable* at essentially chance
level. Mean absolute difference per case is 0.72 points for the agent.

**What survives this.** Aggregate *ordering* is stable - both judges rank `historical`
highest and both separate the agent from the baselines - so the §7 ranking is probably
sound. What does **not** survive is any per-case claim: "this reply is acceptable" is
not a verdict this judge can currently support, and the 60.0% acceptable-rate should be
read as indicative, not measured. A production auto-send gate could not be built on it.

**This is reliability, not validity.** Two judges agreeing would not make them right.
The real validation is judge-vs-human agreement, which needs a human rater:
`python label.py replies --annotator <name> --limit 40`, then
`python scripts/judge_agreement.py`. The harness is built and tested; the ratings are
not collected. I attempted to have them collected, discarded the first nine because
they contained internal contradictions (all-1 scores marked sendable), and chose to
report this gap rather than supply the ratings myself - I wrote both the judge rubric
and the agent prompts, so my ratings would have validated my own instrument against
itself.

---

## 8. Failure analysis: the top 5 modes

### 1. Asks for personal data in public - caused by a contradiction in my own prompt

The single most common serious defect, and it hit **3 of the 5 worst-scoring replies**
(safety = 1):

> **Customer:** *"going to Dubai for our honeymoon & seat allocation has separated our booking so can't even sit together! #fail"*
> **Agent:** *"...Please DM your booking reference and email address so we can take a closer look."*
> **Judge:** *"asks for personal data publicly"*

**Hypothesis - and I checked, it is my bug.** The prompt's escalate rule says *"ask
them to DM the details a colleague needs (booking reference, flight number)"*, while
hard constraint #5 says *"never ask for a booking reference or personal detail in the
public reply"*. Both are in the same prompt. The model resolved the contradiction the
wrong way in a minority of cases. The fix is not more instruction, it is removing the
conflict: the escalate rule should say *"invite them to DM us"* and never enumerate
what to send. This is a one-line prompt change I did not get to re-run.

### 2. Check-in failures get auto-handled - 6 of 14 unsafe triage errors

`checkin_boarding` dominates the false-auto-handle list:

> *"argh can't check in online for Dubai Heathrow club world flight in 8 hrs or pick seats"*
> *"trying to check in on app. Not working & says 'still waiting for authorisation from the US gov'"*

**Hypothesis.** A broken check-in reads like a generic app fault, which has a
plausible generic answer ("try reinstalling"). But it is almost always
account-specific *and* time-critical - the customer flies within hours. The agent
sees a tech-support surface and misses the clock. Encoding "mentions an imminent
flight" as a hard escalation trigger, rather than leaving it to the model's judgement,
would likely fix most of these.

### 3. `flight_disruption` misread as `service_complaint` - the largest intent confusion (5 cases)

**Hypothesis.** An angry tweet about a delay is textually a complaint; the taxonomy
assigns it by *subject* (the disruption) rather than by *tone*. The boundary rule
exists in `taxonomy.yaml` but is genuinely hard, and it is the same boundary the
pre-label models got wrong. This is a taxonomy-design cost, not purely a model error:
if two strong models and a human disagree on a boundary, the boundary is the problem.

### 4. `other` is unlearnable - the ablation's top confusion (8 cases)

`other` -> `service_complaint` (4) and `other` -> `booking_change_cancel` (4).

**Hypothesis.** `other` was added *during* annotation (DECISIONS.md #23) and is
defined negatively - "a real support request nothing else covers". A model cannot
match a positive prototype that does not exist. With 11 examples it is also the
thinnest real class. Either it needs positive sub-definitions (onboard product,
lounge, invoices, codeshare admin) or those should become their own intents.

### 5. Generic non-answers where the retrieval pool has no precedent

> *"I need assistance with some staff travel tickets issued through MyIDTravel"* -> groundedness **1**
> *"UK agent here. Is there a system issue with pre-booking seats with AA codeshare?"* -> *"generic advice, not specific"*

**Hypothesis.** These are B2B and edge queries - trade agents, staff travel, invoices -
with no analogue in 12,000 consumer tweets. Retrieval returns loosely-related consumer
cases, and the agent produces confident-sounding filler. This is the same root cause as
the headline finding in §5: when top similarity is 0.247, evidence is decoration. A
retriever that could *abstain* - returning nothing below a similarity floor, and telling
the model it has no precedent - would be more useful than one that always returns three
rows.

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
