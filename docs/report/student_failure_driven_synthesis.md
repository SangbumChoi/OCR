# Failure-driven synthetic document curriculum

The native student closes the loop between structured validation failures and the next exact
synthetic training batch. The model does not generate its own labels. Validation evidence changes
the allocation over deterministic programmatic generators whose answers, evidence boxes, semantic
graphs, and rendering provenance remain exactly verifiable.

## Leakage and authorization contract

[`plan_student_synthesis.py`](../../scripts/plan_student_synthesis.py) accepts one structured
`per_sample.jsonl` artifact. A plan is authorized for training only when every row has
`split: validation`. Mixed splits and train rows fail. Heldout rows require the explicit
`--allow-heldout-analysis` flag and produce `training_authorized: false` with
`claim_scope: heldout_analysis_only`.

[`generate_from_synthesis_policy.py`](../../scripts/generate_from_synthesis_policy.py) always calls
the plan validator with training authorization required. A heldout-derived analysis therefore
cannot be reused for training by changing a CLI flag. The content-addressed plan also fails if any
allocation, arm, budget, or provenance field changes after fingerprinting.

The final heldout split remains reserved for generalization and promotion evidence. The production
experiment uses a distinct validation root for temperature calibration, adaptive pretraining
feedback, and synthesis-policy learning.

## Reward and factor model

For validation example \(i\), the bounded failure signal is:

```text
failure_i =
  0.50 * (1 - task_score_i)
  + 0.30 * (1 - verifier_reward_i)
  + 0.20 * structure_failure_i
```

All weights are configurable. Each row is attributed to an exact arm using generator case,
language, difficulty level, visual layout family, and composition tier. Composition is classified
as single document, multi-page, or multi-document.

The generator writes `generator_case` into every `gt.json`. The benchmark converter separately
preserves that stable identity and the path-derived `case`, so fanned-out directory names cannot
silently become generator labels.

Sparse Cartesian arms are estimated with empirical-Bayes factor shrinkage. For factor value \(v\):

```text
posterior_failure(v) =
  (sum_failure(v) + prior_strength * global_failure)
  / (count(v) + prior_strength)
```

An arm combines the configured factor posteriors and adds a bounded uncertainty bonus. A
temperature softmax supplies exploitation; an explicit uniform mixture supplies exploration.
Largest-remainder rounding makes integer allocations sum exactly to the requested document budget.
Arm ordering, tie-breaking, seeds, output directories, and the plan fingerprint are deterministic.

## Exact generation

The policy space is declared in
[`sub1b_synthesis_policy.yaml`](../../configs/sub1b_synthesis_policy.yaml). It covers the five hard
single-document families, the multi-page audit packet, the multi-document investment dossier, five
languages where supported, difficulty levels 3-5, and three exact hard-layout families.
Case-specific language and difficulty restrictions prevent unsupported requests.

Each allocated document is generated in an isolated directory with an independent deterministic
seed. The executor verifies the persisted GT against all five arm dimensions before accepting it.
It writes aggregate `index.json` and `gen_config.json` artifacts, so the standard split-leakage,
UDD-building, and benchmark-building stages consume policy output without a separate data path.

## Experiment loop

The production DAG evaluates train, validation, and heldout splits. Its final
`plan_next_synthetic_batch` stage reads:

```text
artifacts/evaluation/validation/per_sample.jsonl
```

and writes:

```text
artifacts/synthetic/next_train_plan.json
```

To execute that allocation in the next run, set:

```yaml
synthetic:
  training_policy_plan: outputs/previous-run/artifacts/synthetic/next_train_plan.json
```

The compiler validates and content-addresses the plan, then replaces only `synthetic_train` with
the exact policy executor. Validation and heldout generation retain independent seeds and fixed
distributions.

Standalone planning is also available:

```bash
python scripts/plan_student_synthesis.py \
  --per-sample outputs/run/artifacts/evaluation/validation/per_sample.jsonl \
  --config configs/sub1b_synthesis_policy.yaml \
  --output outputs/run/artifacts/synthetic/next_train_plan.json
```

This is a cross-run curriculum: run \(t\) produces the authorized allocation for run \(t+1\).
[`student_curriculum_runner.md`](student_curriculum_runner.md) executes the complete attested loop
without rebuilding the tokenizer, reinitializing the model, or repeating pretraining. It does not
adapt against the final heldout set and does not allow student predictions to replace exact
generator ground truth.
