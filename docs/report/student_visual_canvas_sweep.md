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
and cumulative sequence lengths. The vision tower and gated resampler execute each image range
independently. Conv2d patch weights, 2D positions, losses, and checkpoints are unchanged. This is an
exact portable implementation rather than a fused NaViT/FlashAttention varlen kernel.

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

The checkpoint trainer state stores cumulative allocated/executed tokens, valid tokens, and visual
sample count with backward-compatible migration. Sweep aggregation reads the final checkpoint and
reports per-arm means, paired baseline deltas, and deterministic 95% intervals under
`pretraining_efficiency*`.

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
gates. A higher valid-token fraction alone is not a quality result. The portable path launches
attention per image, so wall-clock speed and peak memory must be reported from the target GPU
before claiming deployment throughput.
