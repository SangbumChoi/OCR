# Connector family sweep

The blueprint now treats `student.connector.family` as an executable architecture choice instead
of ignored YAML metadata.

## Compared families

| Family | Mechanism | Connector parameters |
|---|---|---:|
| `gated_resampler` | two learned-latent cross-attention and SwiGLU layers | 33,158,146 |
| `average_pool_projector` | ordered adaptive average pooling plus one linear projection | 1,181,184 |

Both emit exactly 64 visual prefix tokens at language width 1,536. The pooled connector removes
31,976,962 parameters, reducing the full deployment model from 799,919,884 to 767,942,922
parameters. It pools only valid visual tokens, supports dense and packed inputs, and preserves
sequence order. It does not use a learned query or content-dependent token selection, so tiny or
rare evidence may be averaged away.

The comparison is motivated by
[MM1](https://arxiv.org/abs/2403.09611), which ablated average pooling, attention pooling, and
C-Abstractor and found connector choice less influential than image resolution and image-token
count at its larger scales. That conclusion is not assumed to transfer to dense sub-1B documents.

## Matched experiment

[`configs/sub1b_connector_family_sweep.yaml`](../../configs/sub1b_connector_family_sweep.yaml)
defines two families by three paired seeds. It fixes:

- the vision tower, language decoder, 64-token connector output, task heads, and tokenizer;
- initialization, authored/public data, teacher targets, augmentations, and calibration partition;
- all losses, curricula, post-training stages, and the total student-FLOP budget.

The families intentionally have different parameter counts and per-step FLOPs. The sweep is
compute-matched, not parameter-matched. Report heldout quality jointly with actual parameters,
measured visual latency, peak memory, and completed optimizer updates. A pooled connector wins only
if its efficiency gain does not cross the OCR legibility cliff on rare scripts, small text,
evidence localization, table cells, chart legends, formulas, and cross-region reasoning.

## Commands

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_connector_family_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_connector_family_sweep.yaml \
  --smoke-max-steps 2

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_connector_family_sweep.yaml
```

Keep `gated_resampler` when its paired heldout gain justifies roughly 32M extra parameters and the
measured latency. Promote `average_pool_projector` when it remains within the predeclared quality
margin and improves the deployment Pareto frontier.

The machine-readable Pareto contract uses heldout score and train-minus-heldout gap as quality
constraints, actual parameter count and heldout milliseconds per sample as efficiency objectives,
and the full deployment-gate set. The pooled projector must save parameters, show a simultaneous
latency non-regression, pass every capability guardrail, and remain within the 0.005 heldout margin.
Missing target-GPU visual or full-training feasibility evidence is insufficient evidence, not a
promotion.
