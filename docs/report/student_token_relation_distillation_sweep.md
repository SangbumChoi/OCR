# Token-relation distillation sweep

## Question

Selective weight transfer initializes compatible blocks, while same-tokenizer online distillation
can transfer behavior from a larger native document teacher. The existing runner supports top-k
logit KL and pointwise hidden-state cosine matching. Pointwise matching does not directly preserve
which document tokens the teacher represents as related.

[`configs/sub1b_token_relation_distillation_sweep.yaml`](../../configs/sub1b_token_relation_distillation_sweep.yaml)
tests whether a bounded token-relation target transfers document structure more effectively than
pointwise hidden anchors. This is a hidden-representation relation objective, not a claim to
reproduce MiniLM's exact query-key-value attention relations.

## Objective

For each configured language-layer pair, the student feature is projected to the teacher width.
For every sample, at most `relation_max_tokens` valid text positions are selected at deterministic,
evenly spaced indices. After L2 normalization, the teacher and student relation logits are

\[
R_{ij} = \frac{\hat h_i^\top \hat h_j}{\tau_r}.
\]

Self-relations are masked. Each row becomes a distribution over the other sampled tokens, and the
loss is

\[
\mathcal L_{\text{relation}}
= \tau_r^2\,\operatorname{KL}
\left(
\operatorname{softmax}(R^{T})
\;\|\;
\operatorname{softmax}(R^{S})
\right).
\]

The implementation averages over valid samples and configured layer pairs. Sequences with fewer
than two valid tokens contribute no relation term. The quadratic work is bounded by the configured
token cap; training logs `train/distillation_relation_pairs` so the realized dose is visible.

## Matched design

The six cells cross two arms with three paired stochastic blocks:

| Arm | Logit KL | Representation target | Weight | Relation cap |
| --- | ---: | --- | ---: | ---: |
| `hidden_anchors` | 0.15 | pointwise hidden cosine | 0.10 | 0 |
| `token_relations` | 0.15 | token-relation KL | 0.10 | 128 |

Both arms use the same native teacher at `artifacts/native_teacher/student`, the same tokenizer,
teacher identity, layer pairs, relation temperature, data, curriculum boundaries, student
architecture, optimizer budget, post-training, and evaluation documents. The representation-loss
weight is exchanged rather than added, preventing a larger total teacher-loss coefficient from
masquerading as a better target.

The teacher artifact must be a native `DocumentVLMStudent` checkpoint with the same tokenizer
fingerprint. Missing or incompatible teacher evidence fails before training. Active relation loss
also fails closed when the relation cap is below two or no language-layer pair is configured.

## Run

Compile all six experiment DAGs without executing them:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_token_relation_distillation_sweep.yaml \
  --dry-run
```

Run one paired cell:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_token_relation_distillation_sweep.yaml \
  --variant token_relations \
  --replicate seed_0
```

Aggregate after all cells finish:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_token_relation_distillation_sweep.yaml
```

Promotion requires a positive multiplicity-corrected heldout-score effect and non-regression on
grounding, multilingual recognition, full-page OCR, and reading order. Parameter, generalization,
grounding, reasoning, multilingual, and reliability gates must all pass. Until those GPU results
exist, pointwise hidden anchors remain the control and token-relation distillation remains an
ablation.
