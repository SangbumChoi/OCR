# DPO versus IPO preference objective

## Question

[`configs/sub1b_preference_objective_sweep.yaml`](../../configs/sub1b_preference_objective_sweep.yaml)
tests whether a finite-margin Identity Preference Optimization objective is more robust than DPO
for the same verifier-ranked document-VLM preference pairs.

This is an executable six-run design with two arms and three paired stochastic replicates. It is
not evidence that either objective improves heldout capability until the GPU runs and paired
evaluation artifacts exist.

## Matched preference data

Every arm starts from the same evidence-linked SFT checkpoint. For each prompt, the frozen SFT
reference samples eight completions with matched top-p, temperature, completion horizon, and
visual-prefix cache controls. The structured verifier chooses the highest-reward completion
\(y_w\) and lowest-reward completion \(y_l\) only when their reward margin reaches the configured
minimum.

Define the policy-minus-reference log-ratio margin:

\[
r_\theta =
\left[\log \pi_\theta(y_w|x)-\log \pi_\theta(y_l|x)\right]
-
\left[\log \pi_{\mathrm{ref}}(y_w|x)-\log \pi_{\mathrm{ref}}(y_l|x)\right].
\]

The DPO arm minimizes:

\[
\mathcal{L}_{DPO}=-\log \sigma(\beta r_\theta).
\]

The IPO arm minimizes the sampled squared regression objective:

\[
\mathcal{L}_{IPO}=\left(r_\theta-\frac{1}{2\tau}\right)^2.
\]

The implementation follows [Azar et al. (2023)](https://arxiv.org/abs/2310.12036). The default
controls are `dpo_beta: 0.10`, `ipo_tau: 0.10`, summed completion-token log probabilities, and a
minimum verifier margin of `0.05`.

## Controlled comparison

Only `training.posttraining.preference.objective` changes between arms. Candidate pairs, verifier
weights, malformed-response reward, group size, rollout controls, optimizer, activation
checkpointing, and the `192e15` algorithmic student-FLOP budget are matched within each replicate.
Skipped pairs consume rollout compute but do not perform an optimizer step.

Inspect or execute the sweep:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_objective_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_objective_sweep.yaml
```

Run one objective directly by setting `training.posttraining.preference.objective` in the
blueprint:

```bash
python scripts/posttrain_student.py preference \
  --samples data/posttraining/train.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/checkpoints/step-00002000/student \
  --output outputs/student_preference/verifier_ranked
```

## Continuation and interpretation

Preference checkpoints pin the objective, DPO beta, IPO tau, candidate sampling, verifier,
reference checkpoint, compute budget, optimizer, and random state. Resume rejects any change to
that contract. Tests compare resumed and uninterrupted DPO and IPO policy weights exactly.

IPO's finite target may resist overfitting deterministic verifier pairs, but that remains a
hypothesis. Promotion requires a paired heldout-score interval above zero without a worse
train-minus-heldout gap, plus non-regressive grounding, multilingual, reliability, and structural
validity slices. Accepted-pair rate and skipped rollout compute must be reported beside quality.
