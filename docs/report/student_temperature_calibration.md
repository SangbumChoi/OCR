# Native-student temperature calibration

The native evaluator fits and applies scalar confidence temperature scaling after model training.
This implements the adopted O05 method from the frontier catalog without changing generated tokens
or task scores.

## Leakage contract

The production experiment deterministically partitions the configured source split by hashing
`seed:sample_id`. With the default settings, 20% of `heldout` and at least 20 samples become a
separate `calibration` split. Those rows fit one scalar temperature by minimizing binary negative
log likelihood. The remaining 80% retain the `heldout` name and are the only rows used for
heldout quality, generalization, reliability gates, and model selection.

The partition is stable under input reordering, disjoint by construction, and fingerprinted in
`calibration.json`. Fitting fails closed as `insufficient_evidence` when confidence is unavailable,
there are too few rows, or all calibration outcomes have the same correctness label.

## Confidence contract

Correctness is `score >= correct_threshold`, with `0.5` as the default across the mixed document
metrics. The evaluator maps the raw geometric-mean token probability `p` through:

```text
calibrated_probability = sigmoid(logit(p) / temperature)
```

A bounded deterministic search fits the temperature in `[0.05, 20.0]`. Every per-sample artifact
keeps the original `confidence` and adds `calibrated_confidence`; the prediction and score are
unchanged. Each split summary records raw ECE, calibrated ECE, fit status, fit split, sample count,
threshold, and temperature.

## Adjustable configuration

```yaml
evaluation:
  temperature_calibration:
    enabled: true
    source_split: heldout
    fraction: 0.2
    min_samples: 20
    correct_threshold: 0.5
    min_temperature: 0.05
    max_temperature: 20.0
    seed: 47
```

The tiny one-sample smoke experiment disables fitting because it cannot provide both correct and
incorrect outcomes. Production runs keep it enabled.

## W&B and gates

The native evaluator emits paired metric names suitable for one-panel train-versus-heldout views:

```text
eval_by_axis/ece_raw/<split>
eval_by_axis/ece_calibrated/<split>
eval/<split>_calibration_temperature
```

The reliability gate requires calibrated heldout ECE to stay below the configured ceiling, not
worsen relative to raw ECE, and improve relative to the baseline checkpoint. Selective-risk ranking
prefers `calibrated_confidence` when present, while retaining compatibility with older raw-only
artifacts.

Temperature scaling follows
[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). The current method is
global scalar calibration; language- or task-conditional calibration remains an ablation only when
each slice has enough independent calibration samples.
