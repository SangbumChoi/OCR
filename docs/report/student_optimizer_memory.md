# Native student optimizer memory

## Decision

Use one configurable optimizer contract across native pretraining, SFT, preference optimization,
and RLVR. The production blueprint selects bitsandbytes `AdamW8bit`; the CPU tiny experiment
selects standard PyTorch `AdamW`. This is an implementation choice to be validated on the target
GPU, not a claim that a production run already fits.

Install the production training dependencies with:

```bash
pip install -e ".[student,student-gpu]"
```

Every optimizer block exposes:

```yaml
name: adamw_8bit
eps: 0.00000001
min_8bit_size: 4096
block_wise: true
```

`min_8bit_size` leaves small tensors in higher precision. `block_wise` bounds quantization
statistics locally. Learning rate, betas, weight decay, clipping, schedule, and stage-specific
budgets remain separate controls.

## Fail-closed construction

[`optim.py`](../../src/docvlm_eval/student/optim.py) is the only native optimizer factory.
`adamw_8bit` imports and constructs the requested bitsandbytes implementation or raises a clear
runtime error. It never substitutes PyTorch AdamW. The runtime contract records:

- the normalized requested spec;
- the fully qualified implementation class;
- the installed bitsandbytes version for `adamw_8bit`.

Pretraining and SFT store this contract in their student checkpoint metadata. Preference and RLVR
store the same contract with their rollout, objective, reference, and compute contracts. Exact
resume compares the active runtime contract before loading optimizer state. A family, epsilon,
minimum quantized tensor size, block-wise mode, implementation, or bitsandbytes-version change
therefore starts a new trajectory rather than masquerading as an exact resume.

## Target-GPU evidence

Run the production-shaped probe before model initialization:

```bash
python scripts/benchmark_student_training_step.py \
  --config configs/sub1b_architecture.yaml \
  --require-deployment-gate \
  --output artifacts/benchmarks/training_feasibility.json
```

Schema-v2 evidence binds the exact student fingerprint and optimizer spec to the realized runtime.
After a warmup update materializes state, the report measures optimizer tensor bytes, state-bearing
parameter count, maximum state step, setup peak, materialization peak, steady-state peak, latency,
finiteness, and successful update advancement. The deployment gate rejects:

- a different requested or realized optimizer;
- missing bitsandbytes version evidence for `adamw_8bit`;
- stale schema, architecture, precision, batch, accumulation, or checkpointing controls;
- non-finite values, skipped updates, OOM, or insufficient memory headroom.

The authoritative decision is the measured target-GPU report. Package documentation or theoretical
state-width arithmetic is not a substitute for the full model, auxiliary losses, allocator,
activation-checkpointing, and contrastive-memory workload.

## Comparison rule

When comparing `adamw` and `adamw_8bit`, hold the model fingerprint, sample order, batch and
accumulation, precision, losses, schedule, activation checkpointing, visual shape, and device
constant. Compare at least optimizer tensor bytes, effective peak reserved bytes, step latency,
finite updates, and matched held-out quality. Lower memory alone is not a promotion if update
stability or held-out document capability regresses.
