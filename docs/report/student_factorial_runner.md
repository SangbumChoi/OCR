# Initialization-by-data-scale factorial runner

## Research question

The fixed-scale initialization sweep can show whether transferred weights improve final quality,
but it cannot show whether they reduce the amount of unique document supervision required.
[`scripts/run_student_factorial.py`](../../scripts/run_student_factorial.py) crosses the five
initialization arms with three training-data scales while preserving paired stochastic blocks.

The shipped design compiles 45 complete experiment DAGs:

| Scale | Synthetic train documents | Public UDD cap | Heldout documents |
| --- | ---: | ---: | ---: |
| `low` | 10 per case, 40 total | 256 | 100 per case, 400 total |
| `medium` | 32 per case, 128 total | 2,048 | 100 per case, 400 total |
| `full` | 100 per case, 400 total | all eligible rows | 100 per case, 400 total |

Each scale contains random, vision-only, language-only, dual-tower, and half-depth selective
initialization, each repeated under three paired seed blocks. Public row selection is deterministic
at the pinned dataset revision.

## Isolation contract

Only unique training-data diversity changes across scales. The resolved model, initialization
sources, mixture weights, teacher, 20B effective-token pretraining budget, SFT and RLVR budgets,
evaluation samples, and all non-data-scale controls remain fixed. A smaller corpus is replayed until
the same token budget is reached. The experiment therefore estimates a data-diversity effect at
fixed optimization compute; it is not a joint data-and-compute scaling law.

The experiment schema supports independent `synthetic.train_count` and
`synthetic.heldout_count`. This prevents a training-scale patch from silently shrinking the
benchmark. Before factorial statistics are emitted, normalized heldout sample fingerprints must
match across every scale and replicate.

Run a configuration audit without downloading checkpoints or data:

```bash
python scripts/run_student_factorial.py \
  --factorial configs/sub1b_initialization_data_scale.yaml \
  --dry-run
```

Run one bounded cell of the design:

```bash
python scripts/run_student_factorial.py \
  --factorial configs/sub1b_initialization_data_scale.yaml \
  --scale low \
  --variant selective \
  --replicate seed_0
```

A later unfiltered invocation resumes completed cells and aggregates only after the full
scale-by-arm-by-replicate rectangle reaches evaluation.

## Statistical estimand

Every scale first uses the existing paired sweep estimator:

```text
effect(arm, scale, replicate)
  = metric(arm, scale, replicate)
  - metric(random, scale, replicate)
```

The factorial interaction then subtracts the same arm effect at the `full` reference scale:

```text
interaction(arm, scale, replicate)
  = effect(arm, scale, replicate)
  - effect(arm, full, replicate)
```

A positive heldout interaction means the transferred initialization helps more at that reduced
data scale than it does at full scale. A strictly positive paired 95% bootstrap interval is labeled
`improved`; a strictly negative interval is `degraded`; an interval crossing zero is
`inconclusive`. Capability-axis interactions are computed with the same paired
difference-in-differences estimator.

Three replicates are a screening design, not high-powered confirmatory evidence. A promising
interaction should be repeated with more seed blocks and family/language-stratified confidence
intervals before changing the deployment recipe.

## Provenance and outputs

The runner records the configured cap and the actual mixed component rows for every replicate.
All initialization arms inside a scale must have identical row, weight, fold, and pinned
public-selection signatures. The ordinary sweep gate independently requires content-normalized
train and heldout evaluation fingerprints to match across arms. Together these checks catch
incomplete Hub selection, changed synthetic output, or accidental mixture drift before reporting
sample-efficiency claims without treating run-specific cache paths as data differences.

The factorial root contains:

- `factorial_plan.json` and `factorial_spec.json`;
- `scales/<scale>/` with an ordinary matched-sweep plan, runs, gates, and comparison;
- `factorial_run_summary.json`, updated after each scale completes or fails;
- `factorial_comparison.json` with scale effects, paired interactions, capability-axis
  interactions, heldout fingerprints, and actual row counts;
- `factorial_comparison.md` with the heldout quality and interaction table.

W&B runs retain variant and replicate tags and add `data-scale:<id>`. No GPU result is claimed by
the configuration alone; the comparison artifacts become evidence only after all 45 runs finish.
