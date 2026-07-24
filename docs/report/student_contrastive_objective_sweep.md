# Contrastive objective sweep

The native student now separates the vision initialization family from the region-text training
objective. Loading a SigLIP or SigLIP2 vision checkpoint does not by itself make the student
contrastive loss SigLIP.

## Compared objectives

| Arm | Region-text loss | Positive contract |
|---|---|---|
| `siglip` | pairwise sigmoid loss with learned scale and bias | every view with the same image ID |
| `softmax` | symmetric multi-positive InfoNCE with learned scale | every view with the same image ID |

The SigLIP arm follows the pairwise formulation in
[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). Its loss sums all
positive and negative pair terms and divides by local batch size. It therefore does not require
softmax normalization over a global device batch. The initial temperature is `0.07`, the initial
bias is `-10.0`, and the exponential logit scale is capped at `100` for numerical stability.

Both arms instantiate the same two scalar parameters. The softmax arm leaves the additive bias
unused because a common bias cancels under softmax. This preserves exact parameter and student-FLOP
matching while isolating the objective.

## Matched experiment

[`configs/sub1b_contrastive_objective_sweep.yaml`](../../configs/sub1b_contrastive_objective_sweep.yaml)
defines three paired replicates for six runs. It fixes the student, datasets, initialization,
augmentations, all other losses, curriculum, optimizer, and post-training. It also uses the fixed
student-FLOP budget from the box-objective sweep.

The primary decision metric is paired heldout quality across OCR, reading order, table, chart, and
evidence-grounding axes. Also inspect `train/region_text_contrastive`,
`eval/train_region_text_contrastive`, and `eval/held_region_text_contrastive`. Raw loss values are
not comparable across objectives because their normalizations differ.

## Commands

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_objective_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_objective_sweep.yaml \
  --smoke-max-steps 2

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_contrastive_objective_sweep.yaml
```

Keep `siglip` only if its paired heldout improvement is consistent across seeds and does not trade
away dense recognition for retrieval-style alignment. Otherwise switch
`student.task_heads.contrastive_objective` back to `softmax`.
