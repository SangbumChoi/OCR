# Contrastive memory for one-image microbatches

The production student uses `micro_batch_size: 1` with gradient accumulation. Gradient accumulation
does not enlarge the similarity matrix because each microbatch runs its own forward and backward
pass. Batch-only softmax InfoNCE therefore has one positive and no negatives, yielding exactly zero.
Batch-only SigLIP sees only its positive term. Neither objective provides the intended
cross-document discrimination.

## Training contract

`training.pretraining.contrastive_memory` controls a detached per-rank FIFO:

```yaml
contrastive_memory:
  enabled: true
  size: 1024
  min_negatives: 16
  scope: local_fifo
```

The current image and text embeddings remain gradient-bearing queries. Prior image and text
embeddings are detached keys. Both image-to-text and text-to-image directions include current
positives plus memory keys. Stable 63-bit IDs derived from source and image identity prevent
augmented views of the same image from becoming false negatives.

The queue is local to each DDP rank, so it introduces no all-gather synchronization or remote
activation retention. Every rank's queue tensors and IDs are saved in `training_state.pt` and
restored to that rank. Exact resume requires the same world size and queue contract. Diagnostic
gradient-conflict forwards read the queue but do not mutate it.

Queue similarity matmuls are included in `student_flops_seen`. For batch size \(B\), queue occupancy
\(Q\), and contrastive width \(D\), the additional training estimate is \(12BQD\) FLOPs: two
directional query-key matmuls and the existing 3x forward/backward convention. Projection-head
compute remains in the base estimator.

The trainer logs:

```text
train/contrastive_memory_size
train/contrastive_negative_pairs
```

The first value confirms FIFO warmup and saturation. The second is zero until every current query
has at least `min_negatives` nonmatching keys; positive-only batch-one losses are skipped during
that warmup.

## Matched decision experiment

[`configs/sub1b_contrastive_memory_sweep.yaml`](../../configs/sub1b_contrastive_memory_sweep.yaml)
defines two arms by three paired seeds:

| Arm | Memory | Production meaning |
|---|---:|---|
| `memory_1024` | local FIFO, 1,024 keys | real cross-microbatch negatives |
| `batch_only` | disabled | current one-image microbatch only |

Both arms stop at the same total student-FLOP budget. The queue arm's additional similarity compute
is charged, so any quality comparison is not bought with an unreported compute increase. All
student weights, data, augmentation, objective, losses, curriculum, optimizer, and post-training
settings remain matched.

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_memory_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_memory_sweep.yaml \
  --smoke-max-steps 2

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_memory_sweep.yaml
```

Keep the queue only if the paired heldout gain is consistent and
`train/contrastive_negative_pairs` confirms that the loss was active. A nominal contrastive-loss
curve without effective negatives is not evidence for either SigLIP or InfoNCE.
