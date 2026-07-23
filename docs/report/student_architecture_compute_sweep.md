# Compute-matched visual architecture sweep

## Research question

Input resolution and visual latent count trade document legibility against vision, connector, and
decoder cost. Equal optimizer steps or equal effective-token budgets do not isolate this trade:
896px pages process sixteen times as many dense ViT attention pairs as 448px pages, while extra
visual latents lengthen every decoder layer.

[`scripts/run_student_architecture_sweep.py`](../../scripts/run_student_architecture_sweep.py)
therefore compares five visual profiles under identical analytical student FLOP budgets:

| Arm | Square canvas | ViT patches | Visual latents |
| --- | ---: | ---: | ---: |
| `r896_l64` | 896px | 4,096 | 64 |
| `r896_l32` | 896px | 4,096 | 32 |
| `r672_l48` | 672px | 2,304 | 48 |
| `r448_l64` | 448px | 1,024 | 64 |
| `r448_l32` | 448px | 1,024 | 32 |

Every arm is repeated in the three paired stochastic blocks inherited from the core sweep. The
896px/64-latent model is the reference because it is the current full architecture.

## FLOP convention

[`student/compute.py`](../../src/docvlm_eval/student/compute.py) analytically counts dense
multiply-add operations as two FLOPs. It includes:

- patch projection, ViT projections, dense attention products, and MLPs;
- resampler projections, cross-attention products, and SwiGLU MLPs;
- grouped-query decoder projections, dense masked attention, and SwiGLU MLPs;
- the full-vocabulary LM head and enabled task heads;
- three times forward FLOPs for training;
- autoregressive RLVR rollout, policy training, frozen-reference scoring, and periodic supervised
  replay.

The estimate excludes normalization, activation, softmax, loss, optimizer, data loading, and
external-teacher compute. It is a reproducible student-compute estimand, not a claim about hardware
throughput, energy, or total experiment cost. The current collator uses a fixed square canvas, so
the estimator intentionally counts the dense padded shape executed by the model.

## Budget derivation

[`configs/sub1b_architecture_compute_sweep.yaml`](../../configs/sub1b_architecture_compute_sweep.yaml)
defines one reference workload:

- 1,024 text tokens per training sample;
- 20B pretraining effective tokens;
- 1B SFT effective tokens;
- 1,000 RLVR steps with a 256-token prompt, 128-token completion, group size eight, and replay
  every 20 steps.

The compiler evaluates that workload on `r896_l64`, converts it to one integer FLOP budget per
phase, and patches the same integers into every arm. Pretraining and SFT use compute-progress
cosine schedules. Pretraining curriculum boundaries also use compute fraction. RLVR stops on its
student FLOP counter rather than a common step count.

Actual batch shapes drive runtime accounting. The final update can cross the target, so each
completed phase must reach its budget without exceeding the configured 2% overshoot tolerance.
`compute_budget_report.json` fails the suite when any run violates that gate.

## Run

Compile all 15 DAGs without downloading data or checkpoints:

```bash
python scripts/run_student_architecture_sweep.py --dry-run
```

Run one paired cell first:

```bash
python scripts/run_student_architecture_sweep.py \
  --variant r448_l32 \
  --replicate seed_0
```

The complete invocation resumes prior cells:

```bash
python scripts/run_student_architecture_sweep.py
```

Each W&B run receives `resolution:<pixels>`, `visual-latents:<count>`, and
`compute-matched-architecture` tags. The standard sweep aggregation reports heldout means, paired
deltas, capability-axis deltas, confidence intervals, and deployment gates. The compute report is
an additional prerequisite for interpreting those deltas as fixed-student-compute evidence.

## Interpretation

Select a profile only after the full paired rectangle is complete. A lower-resolution arm that
uses more optimizer updates is not disadvantaged: that is the intended consequence of matching
total student compute. Report both quality and realized steps because one reveals the architecture
trade and the other explains how the fixed budget was spent.

This sweep does not isolate native aspect-ratio packing, tiling, wall-clock latency, memory, or
teacher-generation cost. Those require separate controlled experiments.
