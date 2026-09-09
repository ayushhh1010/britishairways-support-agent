# Decision log

Non-obvious calls made while building this, and why. Ordered roughly by when they
were made. Where a decision was made *against* my initial instinct, I say so.

---

**1. Brand: `British_Airways`, not the highest-volume brand.**
I profiled all 40 top brands (`scripts/profile_brands.py`) and thread-level
structure for 6 finalists (`scripts/compare_candidates.py`). `AmazonHelp` wins on
raw volume (82.5k threads vs 16.4k) and on my first automated deflection metric.
I picked BA anyway, on two numbers that matter more: **link rate 6.8% vs 41.3%**
and **English 99.3% vs 76%**. Amazon's Twitter strategy is to route customers to a
web form; grounding a reply generator in that history teaches it to say "click
here" for everything. BA's replies contain actual answers — *"flights under 8
hours 30 have one main meal and a snack in World Traveller"* — which is the thing
a grounded reply generator can learn from.

**2. My own deflection metric was wrong, and I only caught it by reading the data.**
My regex put Amazon's deflection rate at **0.7%**, which appeared to refute my
prior that Amazon is boilerplate-heavy. Reading 12 random non-matching replies
showed they were *all* deflections in phrasings the regex never anticipated:
*"give us a shout here"*, *"report this here"*, *"provide your info w/ us here"*,
*"get in touch with our support team from here"*. The phrasing space is unbounded,
so a keyword detector has an unbounded false-negative rate. I replaced it with
link-rate, which is not phrasing-dependent. This is the concrete reason the report
has a mandatory "what's misleading about my headline number" section: I produced a
misleading headline number within the first hour of this project.

**3. Unit of analysis: first customer message → BA's first reply.**
Not full multi-turn dialogue. First-contact triage is the decision the product
actually has to make, it is the point where escalate-vs-auto-handle has value, and
it gives one clean supervision pair per thread. The cost: the agent is never
evaluated on follow-up turns, where most genuinely hard support work happens. This
is a scope choice, and it is listed in "what I chose not to build".

**4. Threads reconstructed by union-find, not by following reply pointers.**
The corpus is an edge list with branching replies and missing parents. Union-find
over `(tweet_id, in_response_to_tweet_id)` recovers conversations as connected
components in one pass over 2.8M rows, which is robust to branches and to
mid-thread gaps that a naive parent-walk drops.

**5. 12 intents, merged by "would an agent do the same next thing?"**
KMeans(k=28) over 3,600 messages produced the raw structure, but centroids are not
intents: one cluster held 25% of everything as a generic "flight/help" blob, and
another 11% was aviation-enthusiast photography. I hand-merged clusters where the
next action is identical and split the catch-alls. Clusters were input to taxonomy
design, never the taxonomy itself.

**6. ~36% of BA's inbound is not a support request.**
Between the marketing/photo cluster and the praise clusters, a large minority of
messages need no support action at all. An intent taxonomy that only covers
problems would force these into a problem class and produce a fake escalation
rate. Hence `praise_or_chatter` exists as a first-class intent.

**7. `praise` and `chatter` merged into one intent.**
They are semantically different but operationally identical: friendly
acknowledgement, never escalate. Splitting them would add a fine boundary that
depresses annotator agreement while changing no downstream behaviour. Interpretability
loss is real and noted.

**8. Triage is predicted independently of intent, and labelled for what *should*
happen — not what BA did.**
BA routinely deflects answerable questions to DM for its own operational reasons.
Labelling `action` by copying BA's behaviour would build an evaluation that rewards
deflection, which is the opposite of the product goal. So the labelling CLI hides
the historical reply by default. The consequence, stated plainly: `action` labels
encode *my stated policy*, and a different airline could rationally choose
differently. Intent labels have a defensible right answer; triage labels are a
policy artefact.

**9. Retrieval is TF-IDF, not embeddings.**
The corpus is 12k short jargon-dense tweets where the discriminative tokens are
exact strings — `BA2553`, `Avios`, `T5`, `LHR`, `World Traveller`. Lexical overlap
is the signal. Embeddings would add an API dependency, cost, and latency for a gain
I could not have measured within the eval budget. Retrieval is judged on downstream
reply groundedness, not in isolation — which is also a limitation, since I therefore
cannot report retrieval recall.

**10. Golden labels: dual-model proposal → disagreement routing → human adjudication.**
Two *different model families* independently propose intent and action. Disagreement
or an explicit ambiguity flag routes a case to mandatory human adjudication. No
label is ever set by a single model, and `build_golden.py` records per row whether
the final label was human-decided or unadjudicated dual-model consensus, so the
report can state the split rather than implying it was all hand-done. Model–model
agreement is itself reported: a boundary two strong models cannot agree on is one a
human will not apply consistently either.

**11. Splits are hash-based, not shuffle-based, and the retriever sees only the pool.**
`golden` / `retrieval_pool` / `dev` are assigned by SHA-256 of `thread_id`, so the
split is byte-identical on any machine without shipping an index. The retriever is
built **only** from `retrieval_pool`, so a golden case can never retrieve the reply
it is being scored against. `run_eval.py` asserts disjointness rather than trusting it.

**12. The simple baseline is given an advantage the system under test does not get.**
`B1_simple` trains a TF-IDF + logistic-regression classifier **on the golden labels**
(scored out-of-fold via stratified 5-fold CV). The LLM agent is zero-shot and never
sees a golden label. This is deliberate: a baseline you have handicapped proves
nothing. If the LLM merely ties a supervised model that saw the labels, that is the
honest result.

**13. Judge and generator are different model families.**
`openai/gpt-oss-120b` drafts; `qwen/qwen3.8-27b` judges. A judge from the
generator's own family inflates scores through self-preference. I additionally run
`gpt-oss-120b` as a *second* judge on a subsample purely to **quantify** that bias,
rather than asserting it exists.

*Revision:* the original plan was Llama-3.3-70b (generator) vs gpt-oss-120b (judge).
Groq no longer serves Llama chat models on this account, so the split became
OpenAI-OSS vs Qwen. The cross-family principle survived; the specific models did not.

**14. Prompts were iterated on `dev`, never on `golden`.**
My first smoke test ran on golden cases and immediately exposed two failures
(a fabricated baggage allowance, and a signature the prompt forbade). Fixing the
prompt against those cases would have contaminated the eval set, so I moved all
iteration to `dev` and left `golden` untouched. Cheap to get right, and invisible
in the final numbers if you get it wrong.

**15. One LLM call per ticket, not three.**
Intent + triage + reply are produced in a single structured call. It is what a
real deployment would do on latency and cost, and it lets the reply be conditioned
on the agent's own triage decision. Cost: I cannot attribute an error to the
classification stage vs the drafting stage from the main run alone, which is
partly why the no-retrieval ablation exists.

**16. Engineering forced by a hard 8,000 tokens/minute cap.**
The free tier's binding constraint is tokens-per-minute, not requests-per-day. Three
consequences, all of which changed the code: a rolling-window limiter that reserves
*estimated tokens* (a request-only limiter collects a wall of 429s); a compact
taxonomy rendering (~530 tokens vs ~2,000) that keeps the boundary rules and drops
the prose; and `reasoning_effort: low` on gpt-oss, which cut completion tokens ~4x
with no accuracy change on my probes. Limiters are per-model because quotas are
per-model.

**17. Every LLM response is cached to disk and committed to the repo.**
`llm_cache/` is keyed by SHA-256 of the full request. This is what makes the
README's "reproduce in under 15 minutes" claim true rather than aspirational: a
fresh clone replays the exact run offline, with no key and no spend. `data/processed/`
(4 MB) is committed for the same reason. Only `data/raw/` is gitignored.

**18. Escalation asymmetry is priced into the headline triage metric.**
Wrongly escalating costs an agent a minute. Wrongly auto-handling sends a customer a
wrong or harmful answer. Reporting plain accuracy would treat these as equal, so the
headline is a cost-weighted error rate with false-auto-handle weighted 3x, and the
policy default on genuine uncertainty is `escalate`. The 3x is a stated judgement, not
a measured quantity — an airline with real complaint-handling costs would fit it.

**19. Golden set sampled uniformly, not stratified by intent.**
Stratifying would give tighter per-class estimates but would misrepresent headline
accuracy, because accuracy would no longer reflect BA's real traffic mix. I chose
traffic realism and report per-class support counts so thin classes are visibly thin.

**20. The agent does not sign its replies.**
Real BA agents sign with initials (`^JR`). Copying that would imply a named human
stands behind an automated message. The prompt forbids signatures, and a
lower tone score against the historical baseline is an accepted cost.

**21. Pre-labelling is batched; the agent under test never is.**
The taxonomy + policy block is ~1,300 tokens and identical for every case, so sending
it once per message spends ~90% of a daily token budget on repeated boilerplate.
Batching 8 messages per call cut pre-labelling cost ~8x. The agent itself is
deliberately **not** batched: batching would let the model see other customers'
tickets while deciding one, which is both unrealistic and a contamination risk.

**22. Provider selection was decided by measurement, not by documentation.**
The free-tier limits that actually bind are not the advertised ones, and three of
the four options failed for different reasons:

| provider | claim | what actually happened |
|---|---|---|
| Groq | 1000 req/day, 30 RPM | real cap is **200k tokens/day per model**, absent from rate-limit headers and visible only in a 429 body. Exhausted in ~2 hours. |
| Cerebras | 1M tokens/day free | a fresh key returns **HTTP 402 payment_required**; the published free limits need billing credits. |
| Local Ollama | unlimited, free | llama3.1:8b (4.9GB) does not fit a 4GB-VRAM RTX 3050, so it ran on CPU at **18 tok/s prompt eval** -- ~2 min/call, 20+ hours for the sweep. |
| Mistral | 1B tokens/month | genuinely generous, but needs phone verification and opting into data training. |

Groq was kept because it works and needs no new signup, at the cost of spreading the
sweep across a rolling 24h window. Two engineering consequences worth keeping: the
rate limiter reserves *estimated tokens* rather than counting requests, and a
tokens-per-day 429 raises `QuotaExhausted` immediately instead of being retried --
retrying a daily limit turns a clear failure into a silent multi-hour stall, which is
exactly what it did before the fix.

Diagnosing that stall needed `py-spy dump` on the live process: the stack showed every
worker parked in `time.sleep` inside the retry path, which is what distinguished
"throttled" from "hung" after two wrong guesses (thread-unsafe `requests.Session`,
uncapped `retry-after`). Both were fixed anyway -- they were real latent bugs -- but
neither was the cause.

**23. `other` was added to the taxonomy during annotation, not before it.**
Reading all 220 golden messages surfaced ~5% that are real support requests no intent
covered: onboard amenities ("are there USB ports on long haul?"), lounge access,
invoices, codeshare admin. Forcing them into `service_complaint` would have polluted
the class that the classifier already confuses most. Adding a 13th `other` intent is
the honest fix, and its size is itself a finding: a taxonomy induced from clustering
missed a fifth of a class that hand-reading found immediately.

**24. Golden labels were written blind, before any pre-label was read.**
The author labelled all 220 cases from the raw messages alone, then compared against
the model pre-labels. This turns pre-labels into an independent second opinion rather
than an anchor, and makes the reported human-vs-model agreement a real comparison
instead of a measure of how often a human clicked "accept". It also cost more time,
which is why the option existed to do it the other way.
