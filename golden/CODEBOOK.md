# Golden set: sampling and labelling protocol

## What the golden set is

220 first-contact British Airways customer messages, each labelled with:

| field | values |
|---|---|
| `intent` | one of 12 (see `configs/taxonomy.yaml`) |
| `action` | `auto_handle` or `escalate` |
| `escalation_reason_id` | one of 7 reason ids, only when `action = escalate` |
| `label_source` | `human:<name>` / `author_adjudicated` / `dual_model_consensus` |

## How cases were sampled

1. Start from all 16,450 threads where `British_Airways` posted at least one tweet,
   reconstructed by union-find over the `in_response_to_tweet_id` edge list
   (`src/support_agent/data/threads.py`).
2. Reduce each thread to one case: the **first customer message** and BA's
   **first reply after it**. First-contact triage is the decision the product
   actually has to make.
3. Filter to usable supervision (`src/support_agent/data/dataset.py`):
   - customer message 30–600 characters
   - BA reply of at least 5 words
   - drop retweets, URL-only messages, punctuation-only messages
   - drop near-duplicate customer messages (normalised lowercase alphanumeric)

   16,450 → **16,142** cases.
4. Split by a **stable SHA-256 hash of `thread_id`**, not by a random shuffle, so
   the split is identical on every machine and across reruns:
   - `golden` — first 220
   - `retrieval_pool` — next 12,000
   - `dev` — next 400

   The three are disjoint at thread level. `run_eval.py` asserts this, and the
   retriever is built **only** from `retrieval_pool`, so a golden case can never
   retrieve the reply it is scored against.

**Sampling is uniform, not stratified by intent.** This is deliberate: the golden
set should reproduce the real inbound mix, so that headline accuracy means
"accuracy on BA's actual traffic". The cost is that rare intents get few examples
and their per-class F1 is noisy — this is called out in the report rather than
hidden by rebalancing.

## How labels were produced

A three-stage protocol, designed so that no label is ever set by a single model:

**Stage 1 — dual-model proposal.** Two different model families independently
propose `intent` + `action` for every case (`scripts/prelabel_golden.py`):
`llama-3.3-70b-versatile` (Meta) and `gpt-oss-120b` (OpenAI OSS). Each is also
asked for a second choice and an explicit `ambiguous` flag.

**Stage 2 — disagreement routing.** A case is marked `needs_adjudication` if the
two models disagree on intent or action, **or** if either flags ambiguity. Model–
model agreement is reported in `reports/results/prelabel_stats.json`; it is a
useful signal about the taxonomy itself, since a boundary two strong models cannot
agree on is a boundary a human will not apply consistently either.

**Stage 3 — human adjudication.** Every case is read and decided by a human
annotator, who sees the message and may see the model proposals. The proposals are
suggestions with no authority: `build_golden.py` records, per row, whether the
final label came from a human or from unadjudicated dual-model consensus, and the
report states the split explicitly.

Cases still marked `needs_adjudication` at build time are **dropped, not guessed**.
An ambiguous case that nobody adjudicated is not ground truth.

### Anchoring bias

Showing a model's suggestion before a human decides biases the human toward
accepting it. Two mitigations:

- `python -m support_agent.labeling.cli intents --annotator <name> --blind` hides
  all suggestions, and every row records whether it was labelled blind.
- A blind slice labelled by a second annotator gives inter-annotator agreement
  (Cohen's κ). This measures the ceiling on the task: no classifier can
  meaningfully exceed the rate at which two humans agree with each other.

## Labelling rules

**Intent — pick by next action, not by topic.** Two messages share an intent only
if a support agent would do the same next thing. When two intents both fit, apply
the `do NOT use when` clauses in `configs/taxonomy.yaml`; they exist to settle the
recurring boundaries:

| boundary | rule |
|---|---|
| delay complaint vs money claim | asks for money → `compensation_refund`; otherwise `flight_disruption` |
| baggage rules vs missing bag | a *specific* bag is missing/damaged → `baggage_lost_damaged`; a general rule → `baggage_policy` |
| booking change vs broken site | the site erroring is the subject → `digital_technical`; the customer's goal is the subject → the goal's intent |
| complaint vs specific request | a specific actionable ask → that intent; a diffuse grievance → `service_complaint` |
| praise with a question | any real question → the question's intent, not `praise_or_chatter` |
| problem + "can you DM me" | a described problem → that problem's intent; **only** a channel request → `contact_channel_request` |

**Action — label what *should* happen, not what BA did.** BA's historical reply is
hidden by default in the labelling CLI. BA routinely deflects answerable questions
to DM for operational reasons of its own; copying that behaviour would train the
evaluation to reward deflection. Apply `configs/taxonomy.yaml → triage` and default
to `escalate` when genuinely unsure.

**One label per message.** Multi-intent messages get the intent carrying the
customer's primary ask — normally the one they would be angriest to see ignored.

## Known limitations

- **Single-annotator core.** Most labels have one adjudicator, so most of the set
  carries no inter-annotator agreement measurement. Only the blind overlap slice
  does.
- **Uniform sampling ⇒ thin tails.** Rare intents may have <10 examples; their
  per-class metrics are close to meaningless and are reported with support counts.
- **`action` is a judgement, not a fact.** Unlike intent, no ground truth exists in
  the corpus for what *should* have been escalated. It encodes a stated policy, and
  a different airline could rationally choose differently.
- **2017 data.** Policies, fees, and routes have changed. Groundedness is measured
  against what BA said then, not against today's published policy.
