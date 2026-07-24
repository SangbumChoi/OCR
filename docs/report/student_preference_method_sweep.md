# Verifier-ranked DPO versus GRPO

## Question

[`configs/sub1b_preference_method_sweep.yaml`](../../configs/sub1b_preference_method_sweep.yaml)
tests whether the native sub-1B document VLM benefits more from offline verifier-ranked
preferences or from on-policy group-relative reinforcement learning after the same
evidence-linked SFT checkpoint.

This is an executable six-run design with two arms and three paired stochastic replicates. It is
not evidence that either method improves heldout capability until the GPU runs and paired
evaluation artifacts exist.

## DPO data and objective

For each prompt, the frozen SFT reference samples eight completions with the same top-p,
temperature, completion horizon, and visual-prefix cache used by RLVR. The structured verifier
scores every completion with the configured decomposed reward. The highest-reward completion is
chosen and the lowest-reward completion is rejected only when their reward margin is at least
`minimum_reward_margin`. Ties and insufficient margins consume rollout compute but do not perform
an optimizer step.

For chosen response \(y_w\), rejected response \(y_l\), policy \(\pi_\theta\), and frozen SFT
reference \(\pi_{\mathrm{ref}}\), the implemented loss is:

\[
\mathcal{L}_{DPO} =
-\log \sigma\left(
\beta \left[
\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}
-
\log \frac{\pi_{\mathrm{ref}}(y_w|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]\right).
\]

The default uses summed completion-token log probabilities, `beta: 0.10`, and a minimum verifier
margin of `0.05`. The policy and reference each score the accepted pair in one teacher-forced
pass. One image encoding is reused across both sequences.

Run one DPO job directly with `training.posttraining.preference.objective: dpo`:

```bash
python scripts/posttrain_student.py preference \
  --samples data/posttraining/train.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/checkpoints/step-00002000/student \
  --output outputs/student_preference/verifier_ranked
```

## Matched method comparison

The paired arms are:

| Arm | Candidate policy | Update |
| --- | --- | --- |
| `grpo` | evolving trainable policy | standardized within-group advantage plus frozen-reference KL |
| `dpo` | frozen SFT reference | best-versus-worst direct preference loss |

Both arms share the model initialization, data, teacher targets, pretraining, SFT checkpoint,
reward functions, candidate group size, rollout controls, optimizer hyperparameters, evaluation,
and a `192e15` algorithmic student-FLOP stopping budget. DPO and GRPO optimizer seeds are paired
within each replicate.

Algorithmic FLOPs count one visual encoding per prompt. Repeated language scoring scales with the
candidate or pair batch, while the visual tower and connector do not. Activation-checkpoint
recomputation is reported separately from the compute-matching counter. A skipped DPO pair counts
the frozen-reference rollout and no policy or reference-scoring update.

Inspect or execute the sweep:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_method_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_method_sweep.yaml
```

## Exact continuation

DPO checkpoints include policy weights, optimizer and AMP state, Python/Torch/CUDA RNG state,
preference and optimizer cursors, accepted and skipped pair counts, algorithmic and recomputation
FLOPs, tokenizer fingerprint, and frozen-reference identifier. Resume rejects changes to:

- candidate sampling and cache controls;
- preference source, reward margin, beta, or sequence reduction;
- verifier weights or malformed-response reward;
- activation checkpointing or the student-FLOP budget;
- tokenizer or frozen SFT reference.

This preserves the candidate stream and optimizer trajectory across interruption. Tests compare
resumed and uninterrupted policy weights exactly.

## Interpretation boundary

This is a method-level estimand, not a pure loss-function ablation. GRPO samples from the evolving
policy; DPO builds preferences from the frozen SFT reference. A DPO loss advantage could therefore
come from the offline candidate distribution, lower-variance pair updates, or the objective itself.
The separate
[`student_preference_objective_sweep.md`](student_preference_objective_sweep.md) holds candidate
source fixed and directly compares DPO with IPO.

Promotion requires a paired heldout-score interval above zero without a worse train-minus-heldout
gap, plus non-regressive grounding, multilingual, reliability, and structural-validity slices.
Accepted-pair rate and skipped rollout compute must be reported beside quality; a DPO arm that
rarely finds a verifier margin has not received the same number of parameter updates even when its
student-FLOP budget is matched.
