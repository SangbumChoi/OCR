# Structured MLP transfer sweep

## Question

Does a structure-preserving reduction of the pinned Qwen teacher's wider SwiGLU improve the fixed
sub-1B student beyond the existing exact-shape selective transfer?

The estimand is intentionally narrow. Both arms use the same immutable SigLIP and Qwen sources,
alternating half-depth block map, student architecture, data rows, token budget, post-training,
evaluation set, and replicate seeds. The only intervention is the transfer shape policy.

## Transfer contract

For each mapped language block, `structured_mlp` requires all of:

- source and target gate, up, and down weights;
- rank-two weights with one common hidden width;
- internally consistent SwiGLU shapes;
- a source intermediate width strictly larger than the target width.

It scores every source channel by the sum of squared L2 norms from its gate row, up row, and down
column. The top target-width channels are restored to source order and copied with one shared index
set. This preserves the gate/up/down channel relation. It is not independent tensor cropping.

Unsupported groups are left random and reported. Attention KV transfer remains unsupported because
the pinned Qwen source has 256 KV channels while the student requires 512; fabricating channels
would not be pruning. Token embeddings still require an explicit vocabulary identity map.

## Realized dose

Safetensors-header analysis at the pinned Qwen revision gives:

| Half-depth language policy | Copied tensors | Copied parameters | Language fraction |
| --- | ---: | ---: | ---: |
| exact | 61 | 56,679,936 | 8.4% |
| structured MLP | 97 | 283,172,352 | 41.8% |
| incremental structured dose | 36 | 226,492,416 | 33.4% |

The 36 added tensors are the three MLP weights in 12 mapped blocks. The deployment model remains
799,919,884 parameters; only initialization values change. The arm fails closed below a 25%
realized language-tower dose.

## Matched test

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_structured_mlp_transfer_sweep.yaml \
  --dry-run
```

The two arms run over three paired replicates. Promotion requires a Bonferroni-corrected heldout
score improvement of at least 0.005, all six deployment gates, and non-regression on `L1-locate`,
`L1-region`, `multilingual`, `ocr-full`, and `reading-order`.

This is evidence for or against a specific initialization mechanism, not a claim that the broader
Minitron recipe is reproduced. Depth, attention-head, and hidden-width pruning remain separate
estimands. Same-shape attention projections are not proof of semantic compatibility; the strict
geometry gate and its 2x2 factorial are specified in
[`student_attention_geometry_transfer_factorial.md`](student_attention_geometry_transfer_factorial.md).
