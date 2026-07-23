# Paired post-training ablations

## Purpose

Two matched sweeps isolate the required post-training questions without conflating SFT targets with
RLVR rewards:

1. [`configs/sub1b_sft_target_sweep.yaml`](../../configs/sub1b_sft_target_sweep.yaml)
   compares answer-only, free-rationale, and evidence-linked SFT before any RLVR update.
2. [`configs/sub1b_rlvr_reward_sweep.yaml`](../../configs/sub1b_rlvr_reward_sweep.yaml)
   fixes evidence-linked SFT and compares SFT-only, answer-correctness-only RLVR, and the full
   decomposed grounded reward.

Each design has three paired stochastic replicates. These configurations are executable experiment
contracts, not evidence that one target or reward is better. Heldout claims require completed GPU
runs and the generated paired confidence intervals.

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

## Paired controls and outputs

Within each replicate, arms share model initialization, authored train and heldout documents,
public-data selection, offline LFM target selection, image augmentation, pretraining optimization,
SFT optimization, evaluation sampling, and all applicable token or step budgets. The RLVR reward
arms additionally share RLVR seed and update budget. Independent replicates change every material
stochastic seed together across arms.

The standard sweep aggregator verifies identical evaluation artifacts within each paired block and
writes arm distributions, paired deltas, deterministic 95% bootstrap intervals, capability slices,
and deployment gates. W&B runs use separate groups:

- `docvlm-sft-target-ablation`, tagged `sft-target-ablation`;
- `docvlm-rlvr-reward-ablation`, tagged `rlvr-reward-ablation`.

The global baseline is `evidence_linked` for the SFT sweep and `full_reward` for the RLVR sweep.
Consequently, a negative arm-minus-baseline interval supports the corresponding baseline only after
checking per-family, per-language, grounding, reliability, and train-minus-heldout results.

## Interpretation boundary

Do not select a recipe from mean headline score alone. Evidence-linked supervision is useful only
if grounding improves without unacceptable extraction loss or hallucination. Full RLVR is useful
only if gains survive heldout templates and graphs, preserve multilingual controls, and are not
explained by invalid response rates or answer abstention. A confidence interval crossing zero is
inconclusive, not proof of equivalence.
