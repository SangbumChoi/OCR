# Validation-adaptive pretraining mixture

## Research question

The static balanced sampler prevents large corpora from dominating, but it cannot react when one
document capability remains systematically harder during training. This ablation asks whether
heldout-from-optimization group loss should change the next epoch's sampling probabilities.

The implementation is related to the domain-reweighting motivation in
[DoReMi](https://arxiv.org/abs/2305.10429), but it is deliberately smaller and more conservative:
it uses the native student's validation losses directly, updates only at epoch boundaries, and
does not claim to reproduce DoReMi's proxy-reference excess-loss objective.

## Split contract

Adaptive sampling must not inspect the final reported heldout set. The experiment DAG optionally
generates three semantically disjoint synthetic roots:

1. `train` supplies optimization examples;
2. `validation` supplies group losses to the mixture controller;
3. `heldout` remains untouched until final generation evaluation.

`validate_synth_splits.py` checks all enabled roots together before either UDD dataset is built.
The pretraining CLI accepts the validation UDD through `--eval-src`; every row in this explicit
dataset is evaluated regardless of its internal UDD fold label.

Sampler and validation group names must match exactly under the same `balance_by` dimension. This
fail-closed rule prevents a missing validation domain from silently retaining an arbitrary
probability. The dedicated sweep uses synthetic-only training and `task` grouping so train and
validation have the same controlled capability vocabulary.

## Update rule

For group \(g\), periodic evaluation produces weighted validation loss \(L_{g,t}\). The controller
maintains

\[
\bar L_{g,t} = \beta \bar L_{g,t-1} + (1-\beta)L_{g,t}
\]

after the first direct observation. At the next epoch boundary, after the configured warmup, it
updates

\[
\tilde p_{g,t+1} \propto p_{g,t}\,
\exp\left(\eta\left(\frac{\bar L_{g,t}}{\operatorname{mean}_j \bar L_{j,t}}-1\right)\right)
\]

where \(\eta\) is `step_size`. A uniform per-group floor is then mixed in:

\[
p_{g,t+1} = p_{\min} + (1-Gp_{\min})\tilde p_{g,t+1}.
\]

All probabilities remain normalized and no group can fall below `min_probability`.

## Determinism and resume

Evaluation updates the EMA and marks one update as pending; it does not mutate a prefetched
sampler. Pending state is applied only when `batch_in_epoch == 0`. Consequently, every batch within
an epoch uses one fixed probability vector.

Checkpoints store the configuration, normalized weights, EMA losses, pending flag, evaluation
count, and update count. Resume rejects changed configuration, groups, non-finite state, invalid
counters, or a missing controller payload. A mid-epoch interrupted run therefore reconstructs the
same remaining batches and applies the same pending update at the same next epoch boundary.

The metric stream records:

- `train/group_weight/<group>`;
- `adaptive/group_weight/<group>`;
- `adaptive/heldout_loss/<group>` for the optimizer-heldout validation split;
- `adaptive/heldout_loss_ema/<group>`;
- `adaptive/evaluations`, `adaptive/updates`, and `adaptive/pending_update`.

## Matched experiment

[`configs/sub1b_adaptive_mixture_sweep.yaml`](../../configs/sub1b_adaptive_mixture_sweep.yaml)
defines three arms by three paired seeds:

| Arm | Feedback | Step size |
| --- | --- | ---: |
| `fixed_uniform` | none | n/a |
| `adaptive_eta025` | validation task loss | 0.25 |
| `adaptive_eta050` | validation task loss | 0.50 |

All arms use the same synthetic-only training corpus, validation and final-heldout artifacts,
token budget, model, optimizer, post-training, and evaluation seed within each replicate. This is
a causal method comparison, not evidence that adaptive mixing improves quality until the nine GPU
runs and paired heldout analysis are complete.

Compile without running:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_adaptive_mixture_sweep.yaml \
  --dry-run
```

Run one smoke arm first:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_adaptive_mixture_sweep.yaml \
  --variant adaptive_eta025 \
  --replicate seed_0 \
  --to-stage pretrain
```

## Failure modes and decision rule

- Small validation groups can produce noisy losses; the EMA and paired seeds reduce but do not
  remove this risk.
- Intrinsically noisy or unlearnable tasks can absorb probability without improving final
  capability.
- Loss scales can differ across objectives even after using the same active loss weights.
- Reusing final heldout feedback would invalidate the reported generalization estimate; the
  three-way split is mandatory.
- A mean heldout gain is insufficient. Adoption requires a positive paired heldout effect without
  a wider train-minus-heldout gap, probability collapse, or a regression on grounding and
  multilingual slices.
