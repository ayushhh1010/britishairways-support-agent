## 5. Results vs baselines

All 220 golden cases, every system scored identically. `agent` and
`agent_no_retrieval` completed with **zero failed cases**.

| id | intent | triage | reply |
|---|---|---|---|
| `B0_trivial` | majority class | always escalate | one fixed canned reply |
| `B1_simple` | TF-IDF + logistic regression, **out-of-fold CV on golden labels** | keyword rules | nearest historical reply, verbatim |
| `agent` | LLM, zero-shot | LLM against the written policy | LLM, grounded in 3 retrieved cases |
| `agent_no_retrieval` | ablation — same prompt, no evidence | same | (drafts no reply) |

`B1_simple` is **trained on the golden labels**; `agent` is **zero-shot and never
sees one**. The baseline is deliberately advantaged.

### Intent classification (13 classes)

| system | accuracy | 95% CI | macro-F1 | 95% CI |
|---|---|---|---|---|
| `B0_trivial` | 0.182 | [0.132, 0.236] | 0.024 | [0.018, 0.030] |
| `B1_simple` | 0.468 | [0.405, 0.532] | 0.395 | [0.323, 0.479] |
| `agent` | 0.768 | [0.714, 0.827] | 0.726 | [0.647, 0.790] |
| `agent_no_retrieval` | 0.814 | [0.759, 0.864] | 0.773 | [0.689, 0.833] |

### Triage (escalate vs auto-handle)

Cost weights false-auto-handle 3x false-escalate. **Lower cost/case is better.**

| system | accuracy | 95% CI | esc. precision | esc. recall | false auto-handle | cost/case |
|---|---|---|---|---|---|---|
| `B0_trivial` | 0.482 | [0.414, 0.550] | 0.482 | 1.000 | 0 | 0.518 |
| `B1_simple` | 0.641 | [0.577, 0.705] | 0.696 | 0.453 | 58 | 0.886 |
| `agent` | 0.823 | [0.773, 0.868] | 0.786 | 0.868 | 14 | 0.304 |
| `agent_no_retrieval` | 0.868 | [0.823, 0.909] | 0.803 | 0.962 | 4 | 0.168 |

### Answer rate (of cases the system chose to auto-handle)

| system | auto-handled | answered | deflected |
|---|---|---|---|
| `B0_trivial` | 0 | – | – |
| `B1_simple` | 151 | 82.8% | 17.2% |
| `agent` | 103 | 99.0% | 1.0% |

*(`agent_no_retrieval` drafts no replies, so answer rate does not apply to it.)*
