# Native pretraining loss sweep

## Why this suite exists

The native student has four losses that are active in the default end-to-end experiment:

- autoregressive token cross-entropy, whose selected labels may come from gold or quality-gated
  offline LFM sequence targets;
- region-text contrastive alignment;
- normalized box regression plus generalized IoU;
- four-way page orientation classification.

The blueprint also implements same-tokenizer online `teacher_kl` and
`hidden_feature_distillation`, but the default experiment has no native teacher checkpoint. Those
weights are therefore explicitly zero. Treating them as active would create a silent no-op
ablation and invalidate conclusions.

[`configs/sub1b_pretraining_loss_sweep.yaml`](../../configs/sub1b_pretraining_loss_sweep.yaml)
removes each active objective one at a time:

| Arm | Removed loss | Primary causal question |
| --- | --- | --- |
| `full_objective` | none | reference recipe |
| `no_autoregressive` | token cross-entropy | can dense spatial objectives bootstrap transferable document features? |
| `no_region_text_contrastive` | region-text contrastive | does alignment improve dense recognition and grounding? |
| `no_box_regression` | box regression/GIoU | does coordinate supervision improve extraction and relational grounding? |
| `no_orientation` | orientation classification | does explicit rotation supervision improve reading order and robustness? |

Every arm preserves architecture, data, offline teacher generation, initialization, token budgets,
post-training, evaluation documents, and the three paired stochastic blocks. Curriculum overrides
for the removed loss are also zeroed, preventing a later stage from silently re-enabling it.

## Supervision contract

Before training, the runner resolves the effective losses for every curriculum stage. It rejects:

- a stage with no active loss;
- positive `teacher_kl` or `hidden_feature_distillation` without a native online teacher;
- a native teacher checkpoint when both online-teacher losses are inactive.

Teacher inference is skipped in stages where both online losses are zero. Every checkpoint stores
the resolved stage-level active losses, online-teacher status, and counts of gold versus offline
teacher-selected targets. Resume rejects a changed supervision contract.

This separates three mechanisms that must not be conflated:

1. **Selective weight transfer** initializes compatible vision/language parameters.
2. **Offline sequence distillation** uses cross-tokenizer LFM text after quality gating.
3. **Online native distillation** uses same-tokenizer top-k KL and hidden anchors only when an
   explicit native teacher checkpoint is supplied.

## Run

Audit all 15 DAGs without executing them:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_pretraining_loss_sweep.yaml \
  --dry-run
```

Run one paired cell:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_pretraining_loss_sweep.yaml \
  --variant no_box_regression \
  --replicate seed_0
```

Resume and aggregate the complete suite:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_pretraining_loss_sweep.yaml
```

The standard sweep aggregator reports heldout quality, train-heldout gaps, paired bootstrap
intervals, capability-axis deltas, and deployment gates. W&B runs share the
`docvlm-pretraining-loss-ablation` group and receive `loss-ablation`, variant, and replicate tags.

## Interpretation boundary

The suite identifies the marginal value of active native objectives under the current offline LFM
label mixture. It does not estimate the value of online native distillation. That requires a
matched suite with a pinned same-tokenizer teacher, positive online loss weights, and the same
supervision contract across arms.
