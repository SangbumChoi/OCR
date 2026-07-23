# Batch-adaptive visual canvas sweep

## Question

The document collator preserves aspect ratio, but a fixed square canvas executes dense ViT
attention over padding. This is especially wasteful for receipts, text lines, charts, and phone
screens. The batch-adaptive policy asks whether patch-aligned rectangular tensors can reduce
visual compute without changing image scale, model parameters, visual-position capacity, data, or
training dose.

This is a bounded step toward NaViT-style inputs, not full sequence packing. Images in one batch
still share one dense rectangular tensor.

## Spatial contract

Both policies apply the same rotation, long-side resize, normalization, evidence-box transform,
and pixel mask:

| Policy | Dense canvas |
| --- | --- |
| `batch_adaptive` | smallest patch-aligned rectangle containing every resized image in the batch |
| `fixed_square` | configured maximum square, currently 896 by 896 |

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
two policies by three paired replicates. It fixes:

- student parameters, patch size, 4,096-position grid, and 64 resampler latents;
- 896-pixel image long side and no-upscale policy;
- authored train and heldout artifacts, public-data folds, and teacher dose;
- pretraining and SFT token budgets, RLVR steps, batch size, and all stochastic seeds within each
  replicate.

Only `training.pretraining.input_pipeline.visual_canvas_mode` changes. Run:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_visual_canvas_sweep.yaml
```

The adaptive arm is the baseline because it is the deployment candidate. W&B tags include
`visual-canvas-ablation` and the selected policy.

## Decision rule

Retain `batch_adaptive` only when it lowers dense visual tokens and student FLOPs while preserving
heldout score, grounding, multilingual controls, robustness slices, and reliability gates. A
higher valid-token fraction alone is not a quality result. Wall-clock speed and peak memory must
also be reported from the target GPU before claiming deployment throughput.
