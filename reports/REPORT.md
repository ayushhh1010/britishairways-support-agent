# An AI first-line support agent for British Airways

*Hiver SDE Intern take-home. Companion documents: [`DECISIONS.md`](DECISIONS.md) (20
non-obvious decisions), [`../golden/CODEBOOK.md`](../golden/CODEBOOK.md) (how the
evaluation set was built).*

---

## 1. Problem framing

### The brand, and why it was chosen with numbers

I profiled all 40 highest-volume brands in the corpus and compared six finalists at
thread level before committing. `AmazonHelp` is the obvious pick on volume — 82,534
reconstructed threads against BA's 16,450 — and my first automated metric said its
replies were almost never deflections (0.7%).

Both of those signals were wrong in the way that matters. Amazon's Twitter operation
exists to route customers *off* Twitter: **41.3% of its replies contain a link**,
against **6.8%** for BA, and only **76%** of its inbound is English against BA's
**99.3%**. An agent grounded in Amazon's history learns to answer "please contact us
here" to everything, and a small evaluation set would conflate multilingual failure
with intent failure.

British Airways replies contain actual answers — *"flights under 8 hours 30 have one
main meal and a snack in World Traveller"*, *"one-way tickets are normally more
expensive as they are more flexible"*. That is the substance a grounded reply
generator can learn from, and it is why BA was chosen despite being 5x smaller.

### What "good" means for this brand

Three things, in priority order:

1. **Never invent a policy.** An airline reply that states a wrong baggage allowance
   or implies compensation is a commercial and regulatory liability, not a bad
   customer experience. The agent should say "that depends on your ticket type" and
   route, rather than guess.
2. **Never auto-handle something that needed a human.** Escalating unnecessarily
   costs an agent a minute. Auto-handling a stranded passenger, a lost bag, or a
   compensation claim reaches a real customer with a wrong answer. These errors are
   priced 3:1 in the headline triage metric, and the policy default on uncertainty
   is escalate.
3. **Actually answer the answerable.** BA's real replies deflect a great deal. A
   system that mimics that would score well and deliver nothing, so answer rate is
   reported next to quality scores specifically to catch it.

Note that (1) and (3) pull against each other. That tension is the substance of the
product problem, and most of the failure analysis in §4 lives on that boundary.

### What I chose not to build

- **Multi-turn dialogue.** The unit is the first customer message and BA's first
  reply. First-contact triage is where the escalate/auto-handle decision has value.
  The cost is real: follow-up turns, where most hard support work happens, are never
  evaluated.
- **A live tool-calling agent.** No booking lookup, no flight-status API. Roughly
  half of what BA does needs account access — which is exactly why those cases are
  *escalated* rather than answered. The agent is a triage-and-draft layer, not a
  resolution engine.
- **Fine-tuning.** With 220 labelled examples, a zero-shot model with a written
  policy beats a model fine-tuned on a set this size, and the policy is auditable
  and editable by a support manager, which a set of weights is not.
- **A retrained intent model on silver labels.** Tempting and cheap, but it would
  bake the LLM's own errors into the "simple" baseline and destroy its independence.
- **Banking77.** The optional secondary dataset would have improved intent modelling
  in the abstract, but its label space has nothing to do with an airline's, and
  transferring it would have meant optimising for a taxonomy I was not evaluating on.

---

## 2. What was built

```
customer message
   → TF-IDF retrieval over 12,000 historical resolved BA cases
   → one LLM call: intent (12 classes) + triage (auto/escalate + reason id) + drafted reply
   → automated metrics, LLM-judge on reply quality, judge validated against a human
```

**Intents were induced, not invented.** KMeans(k=28) over 3,600 messages gave the
raw structure; the clusters were then merged by hand under one rule: *two messages
share an intent only if an agent would take the same next action*. Centroids are not
intents — one cluster held 25% of all messages as a generic "flight/help" blob and
another 11% was aviation-enthusiast photography. The result is 12 intents, of which
two (`praise_or_chatter`, `contact_channel_request`) exist because **~36% of BA's
inbound is not a support request at all**. A taxonomy covering only problems would
have forced those into problem classes and produced a fake escalation rate.

**Triage is a separate decision from intent**, driven by 7 escalation triggers
(needs account access, money at stake, legal/safety/medical, property irregularity,
time critical, high distress, unparseable). Labels record what *should* happen, not
what BA did — BA deflects answerable questions for its own operational reasons, and
copying that would build an evaluation that rewards deflection.

**The evaluation cannot leak.** `golden` (220) / `retrieval_pool` (12,000) / `dev`
(400) are disjoint at thread level by SHA-256 of `thread_id`. The retriever is built
only from the pool, so a golden case can never retrieve the reply it is scored
against — asserted in code. Prompts were iterated on `dev` only; the first smoke test
accidentally used golden cases, and I moved off them rather than tune on the eval set.

*(Results, failure analysis, and the misleading-number section follow below.)*
