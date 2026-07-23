# Matched native-student sweep runner

## Purpose

[`scripts/run_student_sweep.py`](../../scripts/run_student_sweep.py) turns adjustable model and
training controls into fair, reproducible comparisons. It compiles every variant into an independent
[`student_experiment_runner.md`](student_experiment_runner.md) DAG, verifies declared controls are
identical, runs each DAG with its existing exact-resume behavior, and aggregates train/heldout
generation metrics against one baseline.

This is an experiment execution contract, not evidence that the full GPU sweep has completed. The
comparison files become evidence only after every selected variant has finished on the declared data
and token budgets.

## Configurations

Inspect the five-arm full suite without creating an output root:

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
- answer-correctness-only instead of decomposed grounded RLVR rewards.

Run the two-arm CPU contract:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sweep_tiny.yaml
```

Use repeated `--variant ID` flags to run a subset. `--from-stage`, `--to-stage`, and `--no-resume`
are forwarded to every selected experiment. A later full invocation resumes completed variants and
produces the suite comparison once all evaluation artifacts exist.

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

After every run finishes, the aggregator independently normalizes each train/heldout evaluation
JSONL, replaces its variant-specific image path with the SHA-256 of the referenced image bytes, and
fingerprints the sorted samples. Any cross-variant evaluation-content mismatch stops aggregation.
The two resulting artifact fingerprints are recorded beside the declared matched controls.

Architecture changes also record the validated vision, language, connector, task-head, and total
parameter estimates per variant. Equal steps are not automatically equal FLOPs when resolution,
latent-token count, or depth changes; such architecture sweeps must add a measured-compute control
before making fixed-compute claims.

## Outputs and W&B

Each suite root contains:

- `compiled/<variant>/{experiment,blueprint}.yaml`;
- `runs/<variant>/` with the complete experiment manifests, states, logs, and artifacts;
- `sweep_plan.json` and `sweep_spec.json`;
- `sweep_run_summary.json`, updated after each completed or failed variant;
- `comparison.json` with raw metrics, baseline deltas, answer-type deltas, and ranking;
- `comparison.md` with heldout score, parameter count, generalization gap, and latency.

Every compiled evaluator receives the same W&B group and a unique run name. Native evaluation
already logs paired axis-first keys such as `eval_by_axis/H-count/train` and
`eval_by_axis/H-count/heldout`, allowing train and heldout curves for one suffix to share a panel.
Variant tags are `native-student-sweep` and `variant:<id>`.

Ranking sorts heldout score descending, then prefers a smaller train-minus-heldout score. Ranking is
a navigation aid, not a statistical claim: repeated seeds, confidence intervals, capability slices,
and failure inspection remain required before selecting the deployment recipe.
