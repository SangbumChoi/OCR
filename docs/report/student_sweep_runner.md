# Matched native-student sweep runner

## Purpose

[`scripts/run_student_sweep.py`](../../scripts/run_student_sweep.py) turns adjustable model and
training controls into fair, reproducible comparisons. It compiles every variant into an independent
[`student_experiment_runner.md`](student_experiment_runner.md) DAG, verifies declared controls are
identical within each paired replicate, runs each DAG with its existing exact-resume behavior, and
aggregates train/heldout generation metrics and paired confidence intervals against one baseline.

This is an experiment execution contract, not evidence that the full GPU sweep has completed. The
comparison files become evidence only after every selected variant has finished on the declared data
and token budgets.

## Configurations

Inspect the six-arm by three-replicate full suite without creating an output root:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sweep.yaml \
  --dry-run
```

The full suite compares:

- the complete random-initialized recipe;
- no cross-tokenizer sequence distillation;
- no spatial auxiliary pretraining losses;
- answer-only instead of evidence-linked SFT;
- answer-correctness-only instead of decomposed grounded RLVR rewards;
- no supervised replay anchor during RLVR.

Run the two-arm by two-replicate CPU contract:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sweep_tiny.yaml
```

Use repeated `--variant ID` or `--replicate ID` flags to run a subset. `--from-stage`, `--to-stage`,
and `--no-resume` are forwarded to every selected experiment. A later full invocation resumes
completed runs and produces the suite comparison once the complete arm-by-replicate rectangle has
evaluation artifacts.

## Variant contract

Variants use the `add`, `replace`, and `remove` operations from RFC 6902 JSON Patch. Experiment and
blueprint patches are separate, so a change cannot silently target the wrong document:

```yaml
variants:
  - id: baseline
    hypothesis: "Complete recipe."

  - id: answer_only
    hypothesis: "Tests evidence-linked SFT against answer-only supervision."
    experiment_patches:
      - op: replace
        path: /posttraining/sft/target_mode
        value: answer_only

  - id: no_box_loss
    hypothesis: "Tests normalized box supervision."
    blueprint_patches:
      - op: replace
        path: /training/pretraining/losses/box_regression
        value: 0.0
```

`/name`, `/output_root`, and `/blueprint` are compiler-owned and cannot be patched. Every
non-baseline variant must declare at least one variant-specific operation. Missing pointers,
out-of-range list indices, unsupported operations, duplicate IDs, and invalid compiled experiment
or architecture schemas fail before training.

## Fairness gate

`matched_controls` is an explicit list of JSON pointers that must resolve to exactly equal values in
every compiled variant:

```yaml
matched_controls:
  - document: experiment
    path: /synthetic/train_seed
  - document: experiment
    path: /data/components
  - document: experiment
    path: /pretraining/max_steps
  - document: blueprint
    path: /training/pretraining/optimizer/total_tokens
```

The default suite fixes train/heldout synthesis seeds, data mixture, phase step limits, evaluation
sampling and seed, and pretraining/SFT/RLVR budgets. A patch that changes any matched value is
rejected with the variant ID and offending control paths. The resolved values are copied into the
suite plan and final comparison, so “matched budget” is inspectable rather than implied.

## Paired replicates

`replicate_controls` declares the only experiment or blueprint pointers that replicate blocks may
change. Every replicate must set every declared pointer; undeclared changes fail compilation. Each
replicate patch is applied before the arm patch, and every replicate control must also be a matched
control. This gives each arm the same stochastic conditions inside one block while allowing
independent conditions between blocks.

The shipped suites vary all material stochastic sources:

- random model initialization;
- synthetic train and heldout generation;
- public-data deterministic subsampling in the full suite;
- sequence-target selection;
- image augmentation;
- pretraining, SFT, and RLVR optimization;
- evaluation subsampling.

`initialization.seed` is passed into model construction before any parameter is allocated and is
stored as `initialization_seed` in the initial checkpoint metadata. The resolved blueprint records
the remaining stage seeds. W&B run names are
`<sweep>--<variant>--<replicate>`, with separate `variant:<id>` and `replicate:<id>` tags.

After every run finishes, the aggregator independently normalizes each train/heldout evaluation
JSONL, replaces its run-specific image path with the SHA-256 of the referenced image bytes, and
fingerprints the sorted samples. Artifact equality is required across arms inside each replicate,
not across independent replicates. A mismatch stops aggregation before any delta is reported.

For every metric, the report contains the replicate mean, sample standard deviation, range, and a
deterministic 95% percentile-bootstrap interval. Arm effects use paired deltas
`arm(replicate) - baseline(replicate)` before bootstrapping, which removes shared block variation.
The heldout conclusion is `improved`, `degraded`, or `inconclusive` when the paired interval is
strictly above zero, strictly below zero, or crosses zero. With one replicate the interval is
deliberately unavailable and the conclusion is `insufficient_replicates`.
The baseline arm is labeled `reference`.

The aggregator also re-evaluates every candidate against the baseline from the same replicate.
It joins split-level per-sample outputs, uses the actual parameter count recorded by native
evaluation, and writes one gate report per run plus one replicate-consensus report per arm. An arm
gate passes only when every replicate passes; any replicate failure fails the arm, while missing
confidence, control pairs, or monolingual controls remains `insufficient_evidence`. The baseline
arm remains a reference and therefore does not claim improvement over itself.

Architecture changes also record the validated vision, language, connector, task-head, and total
parameter estimates per variant. Equal steps are not automatically equal FLOPs when resolution,
latent-token count, or depth changes; such architecture sweeps must add a measured-compute control
before making fixed-compute claims.

## Outputs and W&B

Each suite root contains:

- `compiled/<variant>--<replicate>/{experiment,blueprint}.yaml`;
- `runs/<variant>--<replicate>/` with complete manifests, states, logs, and artifacts;
- `sweep_plan.json` and `sweep_spec.json`;
- `sweep_run_summary.json`, updated after each completed or failed run;
- `gates/<variant>--<replicate>.json` and `gates/<variant>.json` with matched baseline decisions;
- `comparison.json` with run metrics, arm distributions, paired baseline deltas, answer-type
  deltas, confidence intervals, gate status, and ranking;
- `comparison.md` with heldout mean and standard deviation, paired 95% interval, parameter count,
  generalization gap, evidence conclusion, and deployment-gate status.

Every compiled evaluator receives the same W&B group and a unique arm-replicate run name. Native
evaluation already logs paired axis-first keys such as `eval_by_axis/H-count/train` and
`eval_by_axis/H-count/heldout`, allowing train and heldout curves for one suffix to share a panel.
Tags include `native-student-sweep`, `variant:<id>`, and `replicate:<id>`.

Ranking sorts mean heldout score descending, then prefers a smaller mean train-minus-heldout score.
Ranking remains a navigation aid: paired intervals, capability slices, multiple-comparison
discipline, and failure inspection are required before selecting the deployment recipe.

Initialization sample efficiency requires a baseline inside every data scale rather than one global
baseline. [`student_factorial_runner.md`](student_factorial_runner.md) composes independent matched
sweeps and estimates paired difference-in-differences while requiring one unchanged heldout set.
