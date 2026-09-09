# An AI first-line support agent for British Airways

*Hiver SDE Intern take-home. Companion documents: [`DECISIONS.md`](DECISIONS.md)
(24 non-obvious decisions), [`../golden/CODEBOOK.md`](../golden/CODEBOOK.md)
(how the evaluation set was built).*

> **Status.** Everything in §1–§4 and §7 is measured and final. §5 (agent vs
> baselines) is the one section still running: the free-tier provider caps at
> 200,000 tokens/day/model and the agent sweep needs ~386,000. Progress is cached to
> disk, and `python scripts/resume.py` continues across quota windows. The exact
> state is printed by `make status`. I have deliberately left the table empty rather
> than fill it from a partial run — see §6, which is about exactly this kind of
> temptation.

---

## 1. Problem framing

### The brand, chosen by measurement

I profiled all 40 highest-volume brands and compared six finalists at thread level
before committing. `AmazonHelp` is the obvious pick on volume — 82,534 reconstructed
threads against BA's 16,450 — and my first automated metric said its replies were
almost never deflections (0.7%).

Both signals were wrong in the way that matters:

| | AmazonHelp | British_Airways |
|---|---|---|
| threads | 82,534 | 16,450 |
| **link rate** (deflection proxy) | **41.3%** | **6.8%** |
| **English** | 76% | **99.3%** |
| knowledge-bearing replies | 76.5% | 75.7% |
| customer-thanks rate | 10.3% | **16.0%** |

Amazon's Twitter operation exists to route customers *off* Twitter. An agent
grounded in that history learns to answer "please contact us here" to everything,
and a 220-case evaluation would conflate multilingual failure with intent failure.
BA's replies contain actual answers — *"flights under 8 hours 30 have one main meal
and a snack in World Traveller"*, *"one-way tickets are normally more expensive as
they are more flexible"*. That is the substance a grounded generator can learn from,
and it is why BA wins despite being 5x smaller.

### What "good" means for this brand

In priority order:

1. **Never invent a policy.** A wrong baggage allowance or an implied compensation
   promise is a commercial and regulatory liability, not merely a bad experience.
   "That depends on your ticket type" is a *good* answer.
2. **Never auto-handle something that needed a human.** Escalating unnecessarily
   costs an agent a minute; auto-handling a stranded passenger, a lost bag, or a
   compensation claim reaches a real customer with a wrong answer. Priced 3:1 in the
   headline triage metric, with `escalate` as the default under uncertainty.
3. **Actually answer the answerable.** BA's own replies deflect heavily. A system
   that mimics that scores well and delivers nothing, which is why `answer_rate` is
   reported beside the quality scores.

(1) and (3) pull against each other. That tension is the product problem, and most
of the failure analysis lives on that boundary.

### What I chose not to build

- **Multi-turn dialogue.** The unit is the first customer message and BA's first
  reply. First-contact triage is where the escalate/auto-handle decision has value.
  Cost: follow-up turns, where the hardest support work happens, are never evaluated.
- **A live tool-calling agent.** No booking lookup, no flight-status API. Roughly
  half of what BA does needs account access — which is precisely why those cases are
  *escalated* rather than answered. This is a triage-and-draft layer, not a
  resolution engine.
- **Fine-tuning.** With 220 labels, a zero-shot model with a written policy beats a
  model fine-tuned on a set this size, and the policy is auditable and editable by a
  support manager. Weights are not.
- **A silver-label intent model.** Cheap and tempting, but it would bake the LLM's
  own errors into the "independent" baseline.
- **Banking77.** Its label space has nothing to do with an airline's; transferring it
  would mean optimising for a taxonomy I am not evaluating on.

---

## 2. What was built

```
customer message
  → TF-IDF retrieval over 12,000 historical resolved BA cases
  → one LLM call: intent (13 classes) + triage (auto/escalate + reason id) + reply
  → automated metrics · LLM-judge on reply quality · judge validated against a human
```

**Intents were induced, not invented.** KMeans(k=28) over 3,600 messages gave the raw
structure; clusters were then merged by hand under one rule — *two messages share an
intent only if an agent would take the same next action*. Centroids are not intents:
one cluster held **25%** of all messages as a generic "flight/help" blob and another
**11%** was aviation-enthusiast photography.

Two findings came straight out of that:

- **~36% of BA's inbound is not a support request.** A taxonomy covering only
  problems would force these into problem classes and manufacture a fake escalation
  rate. Hence `praise_or_chatter` as a first-class intent.
- **`other` had to be added during annotation.** Reading all 220 golden messages
  surfaced ~5% that are real support requests no intent covered — onboard amenities,
  lounge access, invoices, codeshare admin. A taxonomy induced purely from clustering
  missed a class that hand-reading found immediately.

**Triage is a separate decision from intent**, driven by 7 escalation triggers
(account access, money, legal/safety/medical, property irregularity, time-critical,
high distress, unparseable). Labels record what *should* happen, not what BA did —
BA deflects answerable questions for its own reasons, and copying that would build an
evaluation that rewards deflection.

**The evaluation cannot leak.** `golden` (220) / `retrieval_pool` (12,000) / `dev`
(400) are disjoint at thread level by SHA-256 of `thread_id`. The retriever indexes
the pool only, so a golden case can never retrieve the reply it is scored against —
asserted in code and covered by a test. Prompts were iterated on `dev` only; my first
smoke test accidentally used golden cases and I moved off them rather than tune on
the eval set.

---

## 3. The golden set

220 first-contact cases, **100% author-adjudicated, every label written blind** —
before any model pre-label was read. Full protocol in
[`CODEBOOK.md`](../golden/CODEBOOK.md).

| intent | n | | intent | n |
|---|---|---|---|---|
| praise_or_chatter | 40 | | baggage_lost_damaged | 12 |
| service_complaint | 30 | | seating | 11 |
| flight_disruption | 24 | | other | 11 |
| digital_technical | 21 | | loyalty_avios | 10 |
| booking_change_cancel | 20 | | baggage_policy | 9 |
| checkin_boarding | 19 | | contact_channel_request | 1 |
| compensation_refund | 12 | | | |

**Escalate base rate 48.2%** (106 / 220). Escalation reasons: account access 52,
time-critical 19, money 13, property 10, distress 8, legal/safety/medical 4.

Sampling is **uniform, not stratified** — deliberately, so headline accuracy reflects
BA's real traffic mix. The cost is visible above: `contact_channel_request` has a
single example and its per-class metrics will be meaningless. I report support counts
rather than rebalancing to hide it.

---

## 4. Annotation quality (measured)

My blind human labels vs `gpt-oss-120b` pre-labels, n=219:

| | Cohen's κ | raw agreement |
|---|---|---|
| intent (all) | **0.735** — substantial | 76.3% |
| intent (excl. `other`) | 0.779 | 80.3% |
| triage action | **0.482** — moderate | 74.4% |

`other` is excluded in the second row because it was added to the taxonomy *after*
these pre-labels were generated, so the model could not have chosen it.

**The finding that matters.** Of 56 triage disagreements, **48 are model-says-
auto-handle where I say escalate** — a 6:1 asymmetry, entirely in the unsafe
direction. The model systematically wants to answer things that need a human. This is
visible before the agent has run, and it is exactly the failure the 3:1 cost weighting
exists to catch.

Top intent confusions: `flight_disruption`→`service_complaint` (7),
`checkin_boarding`→`digital_technical` (4), `praise_or_chatter`→`service_complaint`
(4), `other`→`checkin_boarding` (4).

**Retrieval quality** (measured on all 220 golden cases): every case retrieves
evidence, but median top-similarity is only **0.247** and just **5.5%** exceed 0.40.
The grounding is frequently *topically adjacent* rather than case-specific. This
matters for §6.

---

## 5. Results vs baselines

| id | intent | triage | reply |
|---|---|---|---|
| `B0_trivial` | majority class | always escalate | one fixed canned reply |
| `B1_simple` | TF-IDF + logistic regression, **out-of-fold CV on golden labels** | keyword rules | nearest historical reply, verbatim |
| `agent` | LLM, zero-shot | LLM against the written policy | LLM, grounded in 3 retrieved cases |
| `agent_no_retrieval` | ablation — same prompt, no evidence | | (no reply) |
| `historical` | — | — | what BA actually replied |

`B1_simple` is **trained on the golden labels** while `agent` is **zero-shot and never
sees one**. The baseline is deliberately advantaged; if the LLM merely ties it, that
is the honest result.

> **Pending.** Numbers land in `reports/results/RESULTS.md` when the run completes.
> Filling this table from a partial sweep would be the exact error §6 is about.

---

## 6. What is misleading about my headline number?

**I have already published one misleading number in this project, and it took reading
the data to catch it.** My deflection regex scored AmazonHelp at 0.7%. Reading twelve
random non-matching replies showed they were *all* deflections in phrasings the regex
never anticipated — *"give us a shout here"*, *"report this here"*, *"provide your info
w/ us here"*. A keyword detector over an unbounded phrasing space has an unbounded
false-negative rate. I replaced it with link rate, which does not depend on phrasing.
That is why this section exists, and the same failure modes apply to everything below.

**1. Intent accuracy is inflated by the easiest class.** `praise_or_chatter` is 18% of
the set and near-trivial — a photo with hashtags. A system that only learned "is this
a complaint or a photo" already scores well. **Macro-F1 and per-class support are the
honest read**, and macro-F1 will be dragged down by classes with under 12 examples,
where it is also barely estimable.

**2. The judge rubric can be gamed by a safe non-answer.** In an early smoke run,
`B0_trivial` — a fixed *"sorry, please DM us"* — scored **4.50/5 mean with 100%
"acceptable"**. Nothing in it can be wrong, so groundedness and safety score full
marks. That is a defect in the rubric, not a property of the baseline, and it is why
`answer_rate` (share of auto-handled cases that actually answer) is reported beside it.
Read the two together or not at all.

**3. Triage labels are policy, not ground truth.** Intent has a defensible right
answer. `action` encodes *my* stated escalation policy. A different airline could
rationally draw the line elsewhere and my agent would score worse against their labels
without changing at all. The 3:1 cost ratio is a judgement, not a measured quantity.

**4. Groundedness is graded against weak evidence.** Median retrieval similarity is
0.247. The judge sees the same weak evidence the generator did, so a reply that is
vaguely consistent with loosely-related cases can score well on groundedness without
being grounded in anything specific to *this* customer. The judge cannot detect a
fact that is absent from evidence it also never had.

**5. Single-annotator labels.** All 220 were adjudicated by one person (me). Blind
labelling removes anchoring but not systematic bias: where I misread the taxonomy, I
misread it consistently, and the agent is scored against that. The human-vs-model κ in
§4 bounds this, but a second independent annotator would bound it properly. The
labelling CLI supports exactly that (`--blind`); it was not run for want of a second
human, and that is a real gap, not an oversight.

**6. n=220, 13 classes.** Roughly 17 cases per class on average and as few as 1 in the
tail. Every headline carries a bootstrap CI for this reason; differences smaller than
those intervals are not differences.

**7. 2017 data.** Policies, fees, and routes have changed. Groundedness is measured
against what BA said then, not against today's published policy — so a factually
"correct" 2026 answer could be scored as ungrounded, and vice versa.

---

## 7. Engineering the free tier (an unplanned finding)

The binding constraints were not the documented ones, and this consumed real time:

| provider | claim | measured reality |
|---|---|---|
| Groq | 1000 req/day, 30 RPM | **200k tokens/day per model**, absent from rate-limit headers, visible only in a 429 body |
| Cerebras | 1M tokens/day free | fresh key returns **HTTP 402 payment_required** |
| local Ollama | unlimited | llama3.1:8b does not fit 4GB VRAM → CPU at **18 tok/s** → ~2 min/call |
| Mistral | 1B tokens/month | genuinely generous; needs phone verification |

Three things in the code exist because of this: the rate limiter reserves *estimated
tokens* rather than counting requests; a tokens-per-day 429 raises `QuotaExhausted`
immediately instead of retrying (retrying a daily limit turns a clear failure into a
silent multi-hour stall — it did, twice); and `resume.py` drives the pipeline across
quota windows without rework.

Diagnosing the stall needed `py-spy dump` on the live process — the stack showed every
worker parked in `time.sleep` inside the retry path, which is what distinguished
"throttled" from "hung" after two wrong guesses. Both guesses (thread-unsafe
`requests.Session`, uncapped `retry-after`) were real latent bugs and were fixed, but
neither was the cause.

---

## 8. What I'd do next with one more week

1. **A second annotator on a 50-case blind slice.** The single largest weakness in
   §6. Inter-annotator κ would bound the ceiling that no classifier can exceed, and it
   is one afternoon of work.
2. **Fix the judge's non-answer blind spot.** Add an explicit *"does this resolve the
   customer's question?"* dimension and re-validate against human ratings. The current
   rubric rewards silence.
3. **Retrieval that actually retrieves.** Median similarity 0.247 is weak. Hybrid
   BM25 + embeddings, and — more importantly — *measure retrieval directly* by
   labelling whether the top-k contains a genuinely applicable precedent, instead of
   inferring it from downstream reply scores.
4. **Confidence-gated auto-send.** The agent emits `intent_confidence` but nothing
   consumes it. Calibrate it, then auto-send only above a threshold chosen so that
   false-auto-handle stays under a stated budget. That converts the cost metric into
   an operating policy.
5. **Multi-turn.** Extend to the second customer turn, where the interesting failures
   (the customer says "that didn't work") actually live.
6. **A real prompt-injection test set.** The prompt says never to obey instructions
   inside customer messages, and that claim is currently untested. Public tweets are
   an obvious injection surface.
