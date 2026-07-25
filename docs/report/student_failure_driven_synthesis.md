# Failure-driven synthetic document curriculum

The native student closes the loop between structured validation failures and the next exact
synthetic training batch. The model does not generate its own labels. Validation evidence changes
the allocation over deterministic programmatic generators whose answers, evidence boxes, semantic
graphs, and rendering provenance remain exactly verifiable.

## Leakage and authorization contract

[`plan_student_synthesis.py`](../../scripts/plan_student_synthesis.py) accepts the final and
matched-baseline structured `per_sample.jsonl` artifacts. A plan is authorized for training only
when every row has `split: validation`. The two inputs must contain exactly the same unique sample
IDs and immutable sample identities, including task, ground truth, image, metadata, language, and
synthesis arm. Mixed splits, train rows, duplicate IDs, or a changed benchmark identity fail.
Heldout rows require the explicit `--allow-heldout-analysis` flag and produce
`training_authorized: false` with `claim_scope: heldout_analysis_only`.

[`generate_from_synthesis_policy.py`](../../scripts/generate_from_synthesis_policy.py) always calls
the plan validator with training authorization required. A heldout-derived analysis therefore
cannot be reused for training by changing a CLI flag. The content-addressed plan also fails if any
allocation, arm, budget, current source, matched baseline source, sample-ID set, or provenance
field changes after fingerprinting.

The final heldout split remains reserved for generalization and promotion evidence. The production
experiment uses a distinct validation root for temperature calibration, adaptive pretraining
feedback, and synthesis-policy learning.

## Residual failure and learning progress

For validation example \(i\), the bounded failure signal is:

```text
failure_i =
  0.50 * (1 - task_score_i)
  + 0.30 * (1 - verifier_reward_i)
  + 0.20 * structure_failure_i
```

The matched learning-progress signal is:

```text
progress_i =
  0.50 * (final_task_score_i - baseline_task_score_i)
  + 0.30 * (final_verifier_reward_i - baseline_verifier_reward_i)
  + 0.20 * (final_structure_valid_i - baseline_structure_valid_i)

utility_i = max(0, failure_i + 0.25 * progress_i)
```

All weights and the progress coefficient are configurable. Residual failure remains the dominant
signal, so difficult or regressed arms are still represented. Positive matched progress adds a
bounded learnability reward; negative progress lowers the utility of repeatedly sampling an arm
whose observed update was harmful. This is a contextual allocation heuristic, not a causal claim
about any individual training example.

Each row is attributed to an exact arm using generator case, language, difficulty level, visual
layout family, and composition tier. Composition is classified as single document, multi-page, or
multi-document.

The generator writes `generator_case` into every `gt.json`. The benchmark converter separately
preserves that stable identity and the path-derived `case`, so fanned-out directory names cannot
silently become generator labels.

Sparse Cartesian arms are estimated with empirical-Bayes factor shrinkage. For factor value \(v\):

```text
posterior_signal(v) =
  (sum_signal(v) + prior_strength * global_signal)
  / (count(v) + prior_strength)
```

The planner estimates failure, learning progress, and combined utility separately. An arm combines
the configured factor utility posteriors and adds a bounded uncertainty bonus. A temperature
softmax supplies exploitation; an explicit uniform mixture supplies exploration. Largest-remainder
rounding makes integer allocations sum exactly to the requested document budget. Arm ordering,
tie-breaking, seeds, output directories, and the plan fingerprint are deterministic.

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
artifacts/evaluation_baseline/validation/per_sample.jsonl
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
  --baseline-per-sample \
    outputs/run/artifacts/evaluation_baseline/validation/per_sample.jsonl \
  --config configs/sub1b_synthesis_policy.yaml \
  --output outputs/run/artifacts/synthetic/next_train_plan.json
```

This is a cross-run curriculum: run \(t\) produces the authorized allocation for run \(t+1\).
[`student_curriculum_runner.md`](student_curriculum_runner.md) executes the complete attested loop
without rebuilding the tokenizer, reinitializing the model, or repeating pretraining. It does not
adapt against the final heldout set and does not allow student predictions to replace exact
generator ground truth.

## Reward-component routing

Schema version 3 preserves the decomposed verifier signal instead of reducing every document to
one reward before choosing a generator family. For each applicable reward component \(c\), the
planner computes:

```text
component_deficit_c = 1 - final_component_score_c
component_progress_c = final_component_score_c - baseline_component_score_c
routed_utility_c =
  max(0, posterior_deficit_c + learning_progress_coefficient * posterior_progress_c)
```

The posterior uses the configured empirical-Bayes prior. Applicability must match exactly between
the final and baseline row for a sample; a changed verifier mask fails rather than creating a
spurious gain. Components are routed only to generator families that can author their ground
truth. Table-tree deficits target table-bearing families, chart tolerance targets chart-bearing
families, and formula equivalence targets scientific pages. Answer, text, box, rationale, and
abstention components can route across all families. Boolean structural validity is always
applicable, so an early checkpoint whose every response is malformed still produces a residual
deficit and a valid next-batch plan instead of losing every task-specific verifier signal.

The candidate priority becomes:

```text
priority_arm =
  predicted_utility_arm
  + uncertainty_coefficient * uncertainty_arm
  + reward_routing_coefficient * routed_reward_utility_arm
```

Only routed utility, dominant component, and evidence count are repeated per generation job.
Component statistics are stored once at plan level, avoiding large repeated tables. Routing
changes document allocation, not labels: every selected generator still creates executable
answers, boxes, programs, and provenance independently of the student prediction.
