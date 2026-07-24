# Multi-loss gradient conflict audit

## Question

The native student jointly optimizes autoregressive generation, teacher KL, hidden-feature
distillation, region/text contrast, box regression, and orientation classification. PCGrad and
GradNorm are plausible remedies when these objectives interfere, but applying either method before
measuring interference would add compute and policy complexity without a falsifiable reason.

This audit measures whether active weighted losses produce persistently opposing gradients and
whether a vision-only measurement gives the same conclusion as a broader shared-trunk
measurement. It is diagnostic only. It does not change the optimizer update and it does not claim
that gradient surgery improves held-out document understanding.

## Probe

`training.pretraining.gradient_conflict_probe` controls the measurement:

```yaml
gradient_conflict_probe:
  enabled: false
  every_steps: 1000
  components: [vision, connector, language]
```

On the first microbatch of each scheduled optimizer window, the runner performs one separate
forward pass. For every active positive-weight loss, it computes gradients with
`torch.autograd.grad` against small shared-trunk anchors:

- the final vision RMSNorm scale;
- the final connector block RMSNorm scale;
- the final language RMSNorm scale.

For every pair of losses, the runner records cosine similarity over anchors touched by both losses,
the shared anchor-element count, individual gradient norms, the negative-pair fraction, and the
minimum cosine. A negative cosine is evidence of local destructive interference. Zero shared
elements means the pair was not measured; it is not evidence that the pair is compatible.

These anchors are a low-memory proxy for full-parameter gradients. They can reveal conflict at
important shared trunks, but they cannot prove that the complete parameter gradient has the same
direction. The replicated audit therefore compares a vision-only anchor with all three trunks.

## Non-interference contract

The probe saves and restores Python, CPU Torch, and CUDA RNG state around its diagnostic forward.
It never writes parameter `.grad` buffers, and the ordinary distributed forward/backward follows
on a fresh graph. The resolved probe configuration is part of the checkpoint supervision contract,
so resume cannot silently change the schedule or anchors.

Every diagnostic record reports one extra forward pass and one autograd traversal per active loss.
This overhead is excluded from the student optimization FLOP budget because it does not update the
model, but it is explicitly logged as `gradient_probe/extra_forward_passes` and
`gradient_probe/extra_backward_passes`. The audit also compares final model hashes between matched
arms against a no-probe control inside each replicate. Any mismatch invalidates the probe before
its cosine evidence is used.

## Replicated audit

[`configs/sub1b_gradient_conflict_audit.yaml`](../../configs/sub1b_gradient_conflict_audit.yaml)
compiles three arms by three paired stochastic replicates:

| Arm | Anchors | Purpose |
| --- | --- | --- |
| `no_probe` | none | Proves that diagnostic work preserves the production trajectory |
| `vision_anchor` | vision | Lowest-cost conflict proxy |
| `all_trunks` | vision, connector, language | Detect conflicts hidden outside the vision trunk |

Data, initialization, optimization, token budget, curriculum, evaluation, and teacher controls are
matched within each replicate. Only diagnostic enablement and anchor selection change.

Inspect the six-run DAG without starting training:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_gradient_conflict_audit.yaml \
  --dry-run
```

Run the audit and aggregate its telemetry:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_gradient_conflict_audit.yaml \
  --to-stage pretrain

python scripts/analyze_gradient_conflicts.py \
  --sweep configs/sub1b_gradient_conflict_audit.yaml
```

The analyzer writes `gradient_conflict_audit.json` and
`gradient_conflict_audit.md` under the sweep root.

## Promotion gate

Gradient surgery is promoted to a new compute-matched experiment only when all of the following
hold:

1. Final student model hashes from both probe arms match the no-probe control in every replicate.
2. A pair has at least 20 measurements in total and appears in every replicate.
3. The all-trunks arm has either an observed negative fraction of at least `0.25` or a mean cosine
   at most `-0.05`.
4. The all-trunks result adds material evidence: a newly measurable pair or a mean-cosine change of
   at least `0.05` from the vision-only proxy.

The result is `promote_gradient_surgery`, `retain_weighted_sum`,
`insufficient_evidence`, or `invalid_probe`. A promotion authorizes a subsequent matched
weighted-sum versus PCGrad versus GradNorm experiment; it is not itself evidence that either
intervention improves quality.
