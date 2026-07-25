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

Dry-run output is compact by default. Shared stage topology, arm patches, replicate patches, and
run fingerprints appear once instead of repeating every long command for every arm-replicate
pair. This keeps large factorial and structured-output sweeps inspectable without producing
table-sized JSON dominated by duplicated paths and arguments. Add `--full-dry-run` only when a
targeted audit needs every resolved command.

The full suite compares:

- the complete random-initialized recipe;
- no cross-tokenizer sequence distillation;
- no spatial auxiliary pretraining losses;
- answer-only instead of evidence-linked SFT;
- answer-correctness-only instead of decomposed grounded RLVR rewards;
- no supervised replay anchor during RLVR.

The diagnostic-only gradient-conflict suite uses the same compiler and paired-replicate contract:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_gradient_conflict_audit.yaml \
  --dry-run
```

Its loss-pair telemetry is aggregated separately by
[`student_gradient_conflict_audit.md`](student_gradient_conflict_audit.md), because ordinary sweep
comparison files contain held-out generation quality rather than optimizer-gradient diagnostics.

Run the two-arm by two-replicate CPU contract:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sweep_tiny.yaml
```

Run the three-seed selective-transfer evidence proxy:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_selective_transfer_fixture_sweep.yaml \
  --no-resume
```

This suite compares random initialization with deterministic cross-architecture fixture transfer.
It exercises exact and structured-MLP transfer, paired evaluation, feedback planning, run
attestation, and aggregation. The fixtures contain random weights, so the suite deliberately has
no promotion contract and cannot establish a pretrained-transfer quality benefit.

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

Before a completed arm is added to the comparison, the runner writes a full-hash experiment
attestation. It seals the exact completed stage prefix, current source signatures, declared
artifacts, checkpoint lineage, optimization progress, and final evaluation. Aggregation loads the
stored attestation and recomputes it from the current worktree and run root. A missing, failed,
stale, or byte-semantically different attestation stops the whole sweep before confidence
intervals, ranking, or promotion are produced. Recipe promotion performs the same recomputation.

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
before making fixed-compute claims. The executable resolution/latent specialization is
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md).

## Outputs and W&B

Each suite root contains:

- `compiled/<variant>--<replicate>/{experiment,blueprint}.yaml`;
- `runs/<variant>--<replicate>/` with complete manifests, states, logs, and artifacts;
- `sweep_plan.json` and `sweep_spec.json`;
- `sweep_run_summary.json`, updated after each completed or failed run;
- `runs/<variant>--<replicate>/evidence_attestation.json`, sealed immediately after each complete
  evaluation path;
- `gates/<variant>--<replicate>.json` and `gates/<variant>.json` with matched baseline decisions;
- `comparison.json` with run metrics, arm distributions, paired baseline deltas, answer-type
  deltas, document-family/language/evidence-count/degradation deltas, confidence intervals,
  final-checkpoint pretraining efficiency statistics, the verified per-run attestation set, gate
  status, ranking, and promotion evidence;
- `comparison.md` with heldout mean and standard deviation, paired 95% interval, parameter count,
  generalization gap, evidence conclusion, deployment-gate status, and promotion decision.

Every compiled training stage and evaluator receives the same W&B group and a unique
arm-replicate-stage run name. Training tags additionally include `stage:pretrain`, `stage:sft`,
`stage:preference`, or `stage:rlvr`; tracking-only fields are excluded from matched-control
comparisons. Native evaluation already logs paired axis-first keys such as
`eval_by_axis/H-count/train` and
`eval_by_axis/H-count/heldout`, allowing train and heldout curves for one suffix to share a panel.
Canonical robustness panels use `eval_by_slice/<axis>/<value>/train` and
`eval_by_slice/<axis>/<value>/heldout`.
Tags include `native-student-sweep`, `variant:<id>`, and `replicate:<id>`.

Ranking sorts mean heldout score descending, then prefers a smaller mean train-minus-heldout score.
Ranking remains a navigation aid: paired intervals, capability slices, multiple-comparison
discipline, and failure inspection are required before selecting the deployment recipe.

## Fail-closed promotion contract

Quality sweeps may declare a machine-readable `promotion` block:

```yaml
promotion:
  primary_metric: heldout_score
  direction: maximize
  minimum_effect: 0.005
  minimum_replicates: 3
  familywise_alpha: 0.05
  max_promotions: 1
  eligible_variants:
    - treatment
  required_gates:
    - parameter_budget
    - generalization
    - grounding
    - reasoning
    - multilingual
    - reliability
  required_axis_deltas:
    L1-locate: 0.0
    L1-region: 0.0
```

`primary_metric` may name an aggregate field from `delta_vs_baseline`, such as
`heldout_score`, or a capability endpoint as `axis.<name>`. For example,
`axis.L1-region` uses the paired heldout `L1-region` delta as the primary effect.
The named primary axis should not also appear in `required_axis_deltas`; those
entries are simultaneous non-regression guardrails for other capabilities.
Missing target-axis evidence produces `insufficient_evidence`.
When `eligible_variants` is present, only those prespecified non-baseline arms enter promotion
testing and multiplicity correction. Other arms remain in the descriptive ranking and declared
linear contrasts. This supports factorials where architecture controls must be reported but only
a treatment arm is eligible for deployment. Omitting the field tests every non-baseline arm.

The aggregator converts every paired primary-metric delta into a positive-is-better benefit,
including metrics configured with `direction: minimize`. It then computes a deterministic
one-sided percentile-bootstrap lower bound using a Bonferroni alpha divided across every candidate
primary effect and required axis guardrail. This controls the declared family-wise error rate
instead of promoting whichever arm happens to rank first.

An arm is eligible only when:

- every required deployment gate is `pass`;
- the complete replicate count reaches `minimum_replicates`;
- the simultaneous lower bound is strictly above `minimum_effect`;
- every required capability-axis lower bound meets its declared non-regression threshold.

A failed gate or regressed axis rejects the arm. Missing metrics, axes, confidence, or gate evidence
produces `insufficient_evidence`; it never becomes a pass. When several arms are eligible, the
selector orders them by simultaneous lower bound, mean benefit, parameter count, and stable ID,
then promotes at most `max_promotions`. `comparison.json` records the full contract, multiplicity
calculation, per-arm evidence, selected variants, and whether the baseline was retained.
`comparison.md` renders the same decision immediately below the descriptive ranking.

The generic quality sweeps for adaptive mixture, composition curriculum, box IoU, contrastive
memory and objective, pretraining loss, sequence teacher, SFT target, preference method and
objective, and RLVR reward and advantage all declare this contract with three paired replicates.
Box IoU and SFT target use `axis.L1-region` as the primary endpoint; the other quality sweeps use
aggregate heldout score.
Connector family, visual canvas, architecture compute, and language-mixer compute remain outside
this scalar path because their decision rule is quality-versus-efficiency Pareto selection. They
declare `mode: pareto` instead of substituting an arbitrary weighted score.

## Pareto promotion contract

A Pareto contract declares at least two positive-is-better objectives after applying each
objective's direction:

```yaml
promotion:
  mode: pareto
  objectives:
    - metric: heldout_score
      direction: maximize
      minimum_effect: -0.005
      required_improvement: false
    - metric: parameters.total
      direction: minimize
      minimum_effect: 1.0
      required_improvement: true
  selection_order: [heldout_score, parameters.total]
  minimum_replicates: 3
  familywise_alpha: 0.05
  max_promotions: 1
```

Metric namespaces are explicit: unprefixed names resolve to paired evaluation deltas,
`axis.<name>` to paired heldout capability deltas, `parameters.<component>` to parameter-count
deltas, `efficiency.<name>` to final-checkpoint pretraining-efficiency deltas, and
`decision.<name>` to compiler-pinned analytical metrics. Every value is compared with the matched
baseline from the same replicate.

`minimum_effect` is a simultaneous lower-bound constraint and may be negative to encode a
predeclared non-inferiority margin. At least one objective must set `required_improvement: true`,
and at least one such objective must have a lower bound strictly above zero. The aggregator applies
the same Bonferroni-corrected one-sided bootstrap across every objective and capability guardrail.
It rejects constraint violations, fails closed on missing evidence, removes candidates dominated
on every objective mean, and applies `selection_order` lexicographically to simultaneous lower
bounds on the remaining frontier. This preserves the declared trade-off instead of hiding it in a
post-hoc weighted average.

Architecture profiles can attach finite `decision_metrics` to each arm. The architecture compiler
uses this mechanism for training FLOPs per sample, total forward FLOPs, and peak bf16 RLVR KV-cache
bytes. It explicitly removes the scalar promotion inherited from the base sweep when the
architecture specification does not provide its own contract. After aggregation, the realized
three-phase compute-budget report is attached as an external gate; a failed gate revokes any
selection in both `comparison.json` and `comparison.md`.

## Materialize the promoted recipe

After a sweep authorizes exactly one arm, materialize a canonical next-stage recipe:

```bash
python scripts/promote_student_recipe.py \
  --sweep configs/sub1b_sweep.yaml \
  --comparison outputs/sweeps/docvlm-core-ablation/comparison.json \
  --output outputs/promoted/docvlm-core
```

The same command accepts an architecture meta-sweep:

```bash
python scripts/promote_student_recipe.py \
  --sweep configs/sub1b_architecture_compute_sweep.yaml \
  --comparison outputs/sweeps/docvlm-visual-architecture-compute/comparison.json \
  --output outputs/promoted/docvlm-visual-architecture
```

The command recompiles the current sweep, re-aggregates the stored run artifacts, and requires the
supplied comparison, sweep fingerprint, and promotion contract to match that recomputed evidence.
It then applies only the shared and selected-arm patches to the original base experiment and
blueprint. Replicate patches are deliberately excluded, so a favorable seed block cannot leak into
the promoted recipe.
For an architecture meta-sweep, it reconstructs the generated child sweep, reapplies the realized
compute-budget external gate, and materializes the selected profile's compiler-generated patches.

The output contains `experiment.yaml`, `blueprint.yaml`, and `promotion_manifest.json`. The
manifest pins the source sweep and comparison hashes, complete promotion evidence, applied patches,
artifact hashes, validated experiment fingerprint, and parameter estimates. Repeating the command
is idempotent only while every fingerprint and artifact hash remains identical. A stale comparison,
changed contract, tampered artifact, non-promoted result, multiple selected arms, or unrelated
non-empty output directory fails closed.

The materialized experiment can become `base_experiment` for the next matched sweep. Winners from
independent sweeps are not merged automatically: their interactions must be measured by a new
sweep based on the previously promoted recipe before another arm is promoted.

Initialization sample efficiency requires a baseline inside every data scale rather than one global
baseline. [`student_factorial_runner.md`](student_factorial_runner.md) composes independent matched
sweeps and estimates paired difference-in-differences while requiring one unchanged heldout set.
