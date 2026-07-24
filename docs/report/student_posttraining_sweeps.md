# Paired post-training ablations

## Purpose

Five matched sweeps isolate the required post-training questions without conflating SFT targets,
RLVR rewards, advantage estimators, and preference-learning methods:

1. [`configs/sub1b_sft_target_sweep.yaml`](../../configs/sub1b_sft_target_sweep.yaml)
   compares answer-only, free-rationale, and evidence-linked SFT before any RLVR update.
2. [`configs/sub1b_rlvr_reward_sweep.yaml`](../../configs/sub1b_rlvr_reward_sweep.yaml)
   fixes evidence-linked SFT and compares SFT-only, answer-correctness-only RLVR, and the full
   decomposed grounded reward.
3. [`configs/sub1b_rlvr_advantage_sweep.yaml`](../../configs/sub1b_rlvr_advantage_sweep.yaml)
   holds the full reward fixed and compares standardized GRPO advantages with RLOO.
4. [`configs/sub1b_preference_method_sweep.yaml`](../../configs/sub1b_preference_method_sweep.yaml)
   compares on-policy GRPO with verifier-ranked DPO under a fixed algorithmic student-FLOP budget.
5. [`configs/sub1b_preference_objective_sweep.yaml`](../../configs/sub1b_preference_objective_sweep.yaml)
   compares DPO with IPO on identical verifier-ranked pairs and fixed compute.

Each design has three paired stochastic replicates. These configurations are executable experiment
contracts, not evidence that one target, reward, or estimator is better. Heldout claims require
completed GPU runs and the generated paired confidence intervals.

## SFT target estimand

The nine-run SFT sweep sets `posttraining.rlvr.enabled: false` for every arm. Evaluation therefore
loads the SFT checkpoint directly. The arms preserve one strict JSON response schema while changing
only target content:

| Arm | Answer | Evidence | Rationale | Baseline role |
| --- | --- | --- | --- | --- |
| `evidence_linked` | gold | authored boxes | authored concise rationale | reference |
| `free_rationale` | gold | empty | authored concise rationale | removes boxes |
| `answer_only` | gold | empty | empty | removes boxes and rationale |

This design measures the effect at the end of SFT. The `sft_answer_only` arm in the broader core
sweep instead measures whether that intervention propagates through a subsequent full RLVR phase;
it is a different estimand.

Inspect or run the design:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sft_target_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sft_target_sweep.yaml
```

## RLVR reward estimand

The nine-run RLVR sweep fixes evidence-linked SFT for all arms:

| Arm | RLVR | Reward |
| --- | --- | --- |
| `full_reward` | enabled | all configured verifiable components |
| `correctness_only` | enabled | answer correctness only |
| `sft_only` | disabled | no rollout or policy update |

The `sft_only` arm estimates the incremental effect of running RLVR. The paired
`correctness_only - full_reward` delta isolates the contribution of task-specific, grounding,
rationale, and abstention rewards under the same rollout and update budget. It does not isolate
individual reward components; a later leave-one-reward-out sweep is warranted only if the
decomposed mixture beats correctness-only.

Inspect or run the design:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_rlvr_reward_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_rlvr_reward_sweep.yaml
```

## Fail-closed stage contract

`posttraining.rlvr.enabled: false` removes the RLVR stage from the compiled DAG, points evaluation
to `@student:sft`, and makes evaluation depend directly on SFT. A disabled RLVR block cannot retain
non-null `max_steps`, replay interval, replay coefficient, or replay dataset overrides. This
prevents an apparently configured RLVR treatment from being silently ignored.

SFT remains mandatory. The experiment compiler also validates the three supported target modes
before data generation or GPU allocation.

## Advantage-estimator estimand

The six-run estimator sweep keeps rollout sampling controls, rewards, replay, optimizer controls,
and the algorithmic student-compute budget matched. `group_standardized` removes within-group
reward scale, while `leave_one_out` uses every other completion as a critic-free baseline. See
[`student_rlvr_advantage_sweep.md`](student_rlvr_advantage_sweep.md) for the equations, exact-resume
objective contract, and promotion rule.

## Preference-method estimand

The six-run method sweep starts every arm from the same evidence-linked SFT checkpoint and matches
the verifier, group size, rollout controls, optimizer controls, and `192e15` algorithmic
student-FLOP budget. GRPO samples from the evolving policy; DPO samples from the frozen SFT
reference and updates on the highest-versus-lowest verifier pair. This is therefore a method-level
comparison rather than a pure objective ablation. See
[`student_preference_method_sweep.md`](student_preference_method_sweep.md) for the exact objective,
compute convention, skipped-pair accounting, and promotion rule.

## Preference-objective estimand

The six-run objective sweep holds the frozen-reference candidate stream, verifier, rollout,
optimizer, and compute budget fixed. It changes only the preference loss between DPO's
log-sigmoid objective and IPO's finite log-ratio target. See
[`student_preference_objective_sweep.md`](student_preference_objective_sweep.md) for equations,
exact-resume guards, and the interpretation boundary.

## Paired controls and outputs

Within each replicate, arms share model initialization, authored train and heldout documents,
public-data selection, offline LFM target selection, image augmentation, pretraining optimization,
SFT optimization, evaluation sampling, and all applicable token or step budgets. The RLVR reward
and advantage arms additionally share RLVR seed, update budget, rollout contract, and replay
contract. Independent replicates change every material stochastic seed together across arms.

The standard sweep aggregator verifies identical evaluation artifacts within each paired block and
writes arm distributions, paired deltas, deterministic 95% bootstrap intervals, capability slices,
and deployment gates. W&B runs use separate groups:

- `docvlm-sft-target-ablation`, tagged `sft-target-ablation`;
- `docvlm-rlvr-reward-ablation`, tagged `rlvr-reward-ablation`;
- `docvlm-rlvr-advantage-ablation`, tagged `rlvr-advantage-ablation`;
- `docvlm-preference-method-ablation`, tagged `preference-method-ablation`.
- `docvlm-preference-objective-ablation`, tagged `preference-objective-ablation`.

The baselines are `evidence_linked`, `full_reward`, `group_standardized`, `grpo`, and `dpo`,
respectively.
Consequently, an arm-minus-baseline interval supports one policy only after checking per-family,
per-language, grounding, reliability, and train-minus-heldout results.

## Interpretation boundary

Do not select a recipe from mean headline score alone. Evidence-linked supervision is useful only
if grounding improves without unacceptable extraction loss or hallucination. Full RLVR is useful
only if gains survive heldout templates and graphs, preserve multilingual controls, and are not
explained by invalid response rates or answer abstention. A confidence interval crossing zero is
inconclusive, not proof of equivalence.
