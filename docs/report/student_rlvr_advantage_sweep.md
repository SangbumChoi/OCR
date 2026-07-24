# RLVR advantage-estimator sweep

## Question

The native RLVR runner samples several completions for one document prompt, scores them with exact
verifiers, and performs one on-policy update. Its original GRPO estimator centered rewards and
divided by the within-group standard deviation. This is scale-invariant, but it can give a tiny
reward difference on an almost-solved prompt the same normalized magnitude as a large verifier
difference on a hard chart, table, or multilingual extraction prompt.

[REINFORCE Leave-One-Out](https://arxiv.org/abs/2402.14740) provides a matched alternative without a
learned critic. The experiment asks whether preserving verifier reward scale improves held-out
document capability at the same rollout and student-compute dose.

## Estimators

For rewards \(r_1,\ldots,r_G\), `group_standardized` uses:

\[
A_i = \frac{r_i-\bar r}{\max(\sigma_r,\epsilon)}.
\]

When reward variance is below `advantage_epsilon`, every advantage is zero.

`leave_one_out` uses the mean reward of the other group members as its baseline:

\[
A_i = r_i-\frac{1}{G-1}\sum_{j\ne i}r_j.
\]

It also produces zero advantage when all rewards tie, but it does not divide away the verifier
scale. Both estimators use the same sampling procedure and paired initial RNG schedule,
sequence-level log-probability reduction, frozen-reference KL, one-update policy, and supervised
replay anchor. Their sampled completions may diverge after the first policy update.

Configure the estimator under the RLVR block:

```yaml
training:
  posttraining:
    rlvr:
      advantage_estimator: group_standardized
```

Every training record includes `advantage_estimator`, `rlvr/advantage_abs_mean`, and
`rlvr/advantage_std`.

## Paired experiment

[`configs/sub1b_rlvr_advantage_sweep.yaml`](../../configs/sub1b_rlvr_advantage_sweep.yaml) compiles
two estimators by three paired stochastic replicates. It holds constant:

- authored train and heldout documents plus public-data selection;
- initialization, teacher-target, augmentation, pretraining, SFT, RLVR, and evaluation seeds;
- evidence-linked SFT checkpoint construction;
- group size, rollout temperature, top-p, completion horizon, and KV-cache policy;
- decomposed reward weights, malformed reward, KL coefficient, and advantage epsilon;
- supervised replay, optimizer learning rate, and a `192e15` algorithmic student-FLOP stop.

This budget approximates the original 1,000-update production-shaped dose. Each arm stops after
crossing the same cumulative student-FLOP target, with at most one rollout-group overshoot.
Sampling seeds and controls are paired, but completions need not remain identical after policy
updates cause the arms to diverge.

Inspect or execute the six runs:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_rlvr_advantage_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_rlvr_advantage_sweep.yaml
```

The standard sweep report provides paired heldout deltas and intervals. Compare answer quality,
grounding, language slices, structural validity, reward variance, KL, advantage scale, and
train-minus-heldout gaps. Update counts may differ because completion lengths and realized step
costs differ; cumulative `rlvr/student_flops_seen` remains the compute estimand.

## Resume contract

RLVR checkpoints store an objective contract containing the estimator, advantage epsilon, KL
coefficient, reward weights, and malformed reward. Resume rejects any change to this contract.
This prevents a run from silently switching estimator or verifier policy after a checkpoint.

## Promotion rule

RLOO is promoted only when its paired heldout interval is positive on the target capability without
failing grounding, multilingual, reliability, or generalization gates at matched student FLOPs.
A larger `advantage_std` or faster training-reward increase alone is not evidence of better document
understanding. If the interval crosses zero, `group_standardized` remains the conservative default.
