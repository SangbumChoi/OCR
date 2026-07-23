# Packed visual sequence sweep

## Question

The document collator preserves aspect ratio, but shared rectangular canvases still execute ViT
projections, attention, and MLPs over cross-example padding. This is especially wasteful for
receipts, text lines, charts, and phone screens. The packed policy asks whether independent
per-image patch sequences reduce visual compute without changing image scale, model parameters,
visual-position capacity, data, or training dose.

## Spatial contract

All policies apply the same rotation, long-side resize, normalization, evidence-box transform, and
canonical position grid:

| Policy | Visual allocation |
| --- | --- |
| `packed` | each image's own patch-aligned sequence; no shared visual canvas |
| `batch_adaptive` | smallest patch-aligned rectangle containing every resized image in the batch |
| `fixed_square` | configured maximum square, currently 896 by 896 |

Packed collation unfolds normalized patches into one concatenated tensor and records position IDs
and cumulative sequence lengths. Conv2d patch weights, 2D positions, losses, and checkpoints are
unchanged. Projections and MLPs run once over concatenated tokens. Attention uses either:

- compiled [PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
  with a block-diagonal `BlockMask` on supported CUDA systems;
- range-wise SDPA as a portable fallback.

`auto` attempts FlexAttention only with CUDA and zero vision dropout, caches a failed device, and
falls back to `loop`. Explicit `flex` fails rather than silently changing the experiment backend.
The PyTorch API remains prototype, so the resolved backend is logged for every training step.

For dense `batch_adaptive`, one portrait and one landscape page can recreate a large near-square
tensor. The optional bucket sampler quantizes the post-augmentation log2 width/height ratio in
0.5-octave steps. It selects a bucket and then samples task/source groups conditionally so the
marginal group probabilities remain equal to their configured weights. The same global bucket is
sliced across distributed ranks. Rows without valid geometry remain trainable in an `unknown`
bucket.

The ViT indexes every patch against the configured two-dimensional 64-by-64 position grid. For
example, a two-row canvas of width two uses IDs `0, 1, 64, 65`; neither denser peers nor packing
changes those IDs. Evidence boxes remain normalized against the canonical maximum square canvas,
so the same sample's target cannot change with its batch peers or execution mode.

## Measured efficiency

Every optimizer step records:

- cumulative `train/student_flops_seen`, summed over actual per-image sequence lengths;
- `train/dense_visual_tokens_per_sample` for backward-compatible allocation reporting;
- `train/executed_visual_tokens_per_sample`;
- `train/valid_visual_token_fraction`.
- `train/visual_attention_backend` (`flex`, `loop`, `dense`, or `none`).

The checkpoint trainer state stores cumulative allocated/executed tokens, valid tokens, and visual
sample count with backward-compatible migration. Sweep aggregation reads the final checkpoint and
reports the resolved attention backend, per-arm means, paired baseline deltas, and deterministic
95% intervals under `pretraining_efficiency*`.

## Target-GPU backend gate

[`scripts/benchmark_student_visual_backend.py`](../../scripts/benchmark_student_visual_backend.py)
isolates the exact blueprint ViT and gated resampler affected by packed attention. It deliberately
does not allocate the unchanged language decoder. The runner excludes warmup and compilation from
timing, measures either forward-only or forward-plus-backward latency, synchronizes CUDA around
every sample, resets peak-memory statistics after warmup, and compares every output with the
portable loop backend using the same weights and input.

Run this on the same GPU type and software image intended for training:

```bash
python scripts/benchmark_student_visual_backend.py \
  --config configs/sub1b_architecture.yaml \
  --sequence-lengths 2520,2520 \
  --backends loop auto flex \
  --mode training \
  --precision bfloat16 \
  --warmup-iterations 3 \
  --iterations 10 \
  --device cuda \
  --require-flex \
  --output outputs/visual_backend_a100_bf16.json \
  --wandb-project docvlm-ablation \
  --wandb-group visual-backend-gate
```

Use `float16` on devices without BF16. `--require-flex` exits nonzero if either `auto` or explicit
`flex` errors or resolves to `loop`, but writes the JSON and W&B evidence first. The report includes
requested and resolved backends, median/p95 latency, visual-token throughput, allocated and
reserved peak CUDA memory, output checksum, maximum absolute delta from `loop`, PyTorch/CUDA/device
metadata, loop-relative speed and memory ratios, and a fingerprint of the complete student
configuration.

Do not compare records across different sequence lengths, modes, precision, GPU models, or
configuration fingerprints. Accept FlexAttention only when the gate passes, numerical delta is
within the recorded tolerance, and repeated target-GPU runs improve median latency or peak memory.
An `auto` record resolving to `loop` is valid fallback evidence, not FlexAttention performance.
The end-to-end evaluator consumes this JSON as the `visual_efficiency` deployment gate; throughput
is therefore part of the same final acceptance report as held-out quality, grounding, reasoning,
multilingual retention, and reliability rather than an unattached benchmark.
This packed-backend runner is enabled only for the packed sweep arm. Dense controls disable the
preflight and receive `insufficient_evidence` for this gate rather than being incorrectly approved
using packed-path timing.

## Matched experiment

[`configs/sub1b_visual_canvas_sweep.yaml`](../../configs/sub1b_visual_canvas_sweep.yaml) compiles
four policies by three paired replicates:

| Arm | Sequence | Canvas | Aspect buckets | Isolated comparison |
| --- | --- | --- | --- | --- |
| `packed` | packed | none | off | deployment candidate |
| `dense_adaptive_bucketed` | dense | adaptive | on | best dense alternative |
| `dense_adaptive_unbucketed` | dense | adaptive | off | bucket contribution |
| `dense_fixed_square` | dense | fixed | off | maximum-padding control |

The sweep uses pretraining micro-batches of two and four accumulation steps, preserving the
blueprint's effective batch of eight while making cross-example padding observable. It also fixes:

- student parameters, patch size, 4,096-position grid, and 64 resampler latents;
- 896-pixel image long side and no-upscale policy;
- authored train and heldout artifacts, public-data folds, and teacher dose;
- pretraining and SFT token budgets, RLVR steps, effective batch size, and all stochastic seeds
  within each replicate.

Only visual sequence, canvas, and aspect-bucketing policies change across arms. Run:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml
```

The packed arm is the baseline because it is the deployment candidate. W&B tags include
`visual-canvas-ablation`, sequence mode, canvas policy where applicable, and bucket policy.

## Decision rule

First compare packed with dense adaptive unbucketed to isolate sequence allocation, compare the two
adaptive arms to isolate bucketing, then compare unbucketed adaptive with fixed square to isolate
canvas allocation. Retain packed only when it lowers executed visual tokens and student FLOPs while
preserving heldout score, grounding, multilingual controls, robustness slices, and reliability
gates. A higher valid-token fraction alone is not a quality result. The loop fallback launches
attention per image while FlexAttention has compile and mask-construction costs, so wall-clock
speed and peak memory must be reported from the target GPU before claiming deployment throughput.
