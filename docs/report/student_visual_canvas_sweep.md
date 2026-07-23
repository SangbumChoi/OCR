# Batch-adaptive visual canvas sweep

## Question

The document collator preserves aspect ratio, but a fixed square canvas executes dense ViT
attention over padding. This is especially wasteful for receipts, text lines, charts, and phone
screens. The batch-adaptive policy asks whether patch-aligned rectangular tensors can reduce
visual compute without changing image scale, model parameters, visual-position capacity, data, or
training dose.

This is a bounded step toward NaViT-style inputs, not full sequence packing. Images in one batch
still share one dense rectangular tensor. The default sampler therefore uses UDD image dimensions
to form rotation-aware aspect-ratio buckets before the collator constructs that tensor.

## Spatial contract

Both policies apply the same rotation, long-side resize, normalization, evidence-box transform,
and pixel mask:

| Policy | Dense canvas |
| --- | --- |
| `batch_adaptive` | smallest patch-aligned rectangle containing every resized image in the batch |
| `fixed_square` | configured maximum square, currently 896 by 896 |

For `batch_adaptive`, one portrait and one landscape page can otherwise recreate a large near-square
tensor. The bucket sampler quantizes the post-augmentation log2 width/height ratio in 0.5-octave
steps. It selects a bucket and then samples task/source groups conditionally so the marginal group
probabilities remain equal to their configured weights. The same global bucket is sliced across
distributed ranks. Rows without valid geometry remain trainable in an `unknown` bucket.

The ViT indexes every patch against the configured two-dimensional 64-by-64 position grid. For
example, a two-row canvas of width two uses IDs `0, 1, 64, 65`; extending the batch canvas to three
columns does not change those four IDs. Fully padded patches remain excluded from ViT and resampler
attention. Evidence boxes remain normalized against the canonical maximum square canvas rather
than the dense batch tensor, so the same sample's target cannot change with its batch peers.

## Measured efficiency

Every optimizer step records:

- cumulative `train/student_flops_seen`, calculated from the actual dense tensor shape;
- `train/dense_visual_tokens_per_sample`;
- `train/valid_visual_token_fraction`.

The checkpoint trainer state stores cumulative dense tokens, valid tokens, and visual sample
count. Sweep aggregation reads the final checkpoint and reports per-arm means, paired baseline
deltas, and deterministic 95% intervals under `pretraining_efficiency*`.

## Matched experiment

[`configs/sub1b_visual_canvas_sweep.yaml`](../../configs/sub1b_visual_canvas_sweep.yaml) compiles
three policies by three paired replicates:

| Arm | Canvas | Aspect buckets | Isolated comparison |
| --- | --- | --- | --- |
| `batch_adaptive_bucketed` | adaptive | on | deployment candidate |
| `batch_adaptive_unbucketed` | adaptive | off | bucket contribution |
| `fixed_square` | fixed | off | dense-canvas control |

The sweep uses pretraining micro-batches of two and four accumulation steps, preserving the
blueprint's effective batch of eight while making cross-example padding observable. It also fixes:

- student parameters, patch size, 4,096-position grid, and 64 resampler latents;
- 896-pixel image long side and no-upscale policy;
- authored train and heldout artifacts, public-data folds, and teacher dose;
- pretraining and SFT token budgets, RLVR steps, effective batch size, and all stochastic seeds within each
  replicate.

Only the visual canvas and aspect-bucketing policies change across arms. Run:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml
```

The bucketed adaptive arm is the baseline because it is the deployment candidate. W&B tags include
`visual-canvas-ablation`, the canvas policy, and the bucket policy.

## Decision rule

First compare the two adaptive arms to isolate bucketing, then compare unbucketed adaptive with
fixed square to isolate canvas allocation. Retain the combined default only when it lowers dense
visual tokens and student FLOPs while preserving heldout score, grounding, multilingual controls,
robustness slices, and reliability gates. A higher valid-token fraction alone is not a quality
result. Wall-clock speed and peak memory must also be reported from the target GPU before claiming
deployment throughput.
