# Selective initialization acquisition and sweep

The native student can start fully random, inherit exact-shape tensors, or reduce a wider source
SwiGLU into the fixed student MLP with one shared salience-selected channel map. This path is part
of the experiment DAG: source acquisition, revision validation, file provenance, transfer, and the
resulting report are not manual setup steps.

## Pinned sources

The matched initialization sweep uses:

| Component | Source | Immutable revision | Why it is useful | Known mismatch |
| --- | --- | --- | --- | --- |
| vision | [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224/tree/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed) | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | 768-wide, 12-layer, 12-head ViT with 3072-wide MLP blocks | 16-pixel patch kernel and source position table do not match the student's 14-pixel/4096-token inputs |
| language | [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) | `8faed761d45a263340a0528343f099c05c9a4323` | 1536 hidden width and multilingual decoder representations | KV-head geometry differs; its 8960-wide MLP is eligible only for the explicit `structured_mlp` policy; external vocabulary rows are not copied without an identity map |

The source model can exceed one billion parameters because it is an initialization teacher, not the
deployed student. The default student remains 799,919,884 parameters.

Inspect compatibility from safetensors headers without downloading model weights:

```bash
python scripts/analyze_transfer_compatibility.py \
  --repo-id google/siglip-base-patch16-224 \
  --revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed \
  --family siglip --component vision --fraction 0.5

python scripts/analyze_transfer_compatibility.py \
  --repo-id Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --family llama --component language --fraction 0.5 \
  --shape-policy structured_mlp
```

At the pinned revisions, the `I4_selective` half-depth policy finds 99 vision tensors containing
42,529,536 parameters and 61 language tensors containing 56,679,936 parameters. This is 48.0% of
the student vision tower and 8.4% of the language tower, or 99,209,472 parameters (12.4% of the
whole student). Full-depth compatibility is 95.9% for vision and 16.0% for language. These are
compatibility counts, not evidence of quality improvement; the matched sweep must establish that.

For the same half-depth language mapping, `structured_mlp` raises compatible language parameters
from 56,679,936 to 283,172,352. The added 226,492,416 parameters are exactly 36 weights across 12
complete SwiGLU groups. This is 41.8% of the language tower and 35.4% of the whole student. The
deployed architecture and its 799,919,884 parameters do not change.

## Acquisition contract

An experiment source may be a local checkpoint path or a pinned Hub mapping:

```yaml
initialization:
  arm: I4_selective
  vision_family: siglip
  vision_source:
    hub:
      repo_id: google/siglip-base-patch16-224
      revision: 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
  language_family: llama
  language_source:
    hub:
      repo_id: Qwen/Qwen2.5-1.5B
      revision: 8faed761d45a263340a0528343f099c05c9a4323
```

`scripts/acquire_student_checkpoint.py` verifies that the Hub resolves to the exact 40-character
commit, checks `config.json` against the declared adapter family, verifies every indexed shard,
and records SHA-256 plus byte size for the config, index, and weights. Model files remain in the
shared Hugging Face cache, so paired runs do not duplicate multi-gigabyte checkpoints. Each run
stores a checkpoint manifest under `artifacts/initialization_sources/`; missing cache files
invalidate the acquisition stage.

Local checkpoint paths are content-addressed directly in the experiment fingerprint. Changing
their weights invalidates initialization and every downstream stage.

## Transfer gate

`scripts/build_sub1b_student.py` canonicalizes native, SigLIP, Llama-style, LFM2, and LFM2-VL names,
depth-maps the selected blocks, and applies the arm's declared shape policy. The default `exact`
policy copies only exact-shape tensors. `structured_mlp` first requires complete gate, up, and down
weights with equal source and target hidden width and a strictly wider source intermediate axis.
It ranks channels by the joint squared L2 norm of gate rows, up rows, and down columns, preserves
the selected channels in source order, and applies that same index set to all three weights.
Incomplete groups, hidden-width mismatches, and source-smaller-than-target groups remain random.
The LFM2 adapter covers attention, gated short convolution, norms, and SwiGLU projections for
hybrid students. Metadata records copied keys and parameters,
missing source keys, and shape mismatches in `artifacts/initial/metadata.json`. A non-random arm
fails if any required component copies zero parameters. It also checks the realized copied
parameters against a component-relative floor declared by the arm:

| Arm | Minimum vision dose | Minimum language dose |
| --- | ---: | ---: |
| `I1_vision` | 80% | n/a |
| `I2_language` | n/a | 15% |
| `I3_dual` | 80% | 15% |
| `I4_selective` | 40% | 7.5% |
| `I5_structured_mlp` | 40% | 25% |
| `I6_strict_structured` | 40% | 25% |

These conservative floors sit below the pinned compatibility counts but reject a source,
canonicalization, or architecture change that leaves only a token number of copied tensors.
Metadata records the target component size, realized component fraction, and required floor for
each report. The connector remains random in all shipped arms.

Token embeddings and the tied output head require an explicit target-to-source token identity map.
Matching width or row count alone is not accepted as proof that two vocabulary rows mean the same
thing.

Every materialized structured group records its source and target widths, selection method,
channel-index SHA-256, and a bounded index preview. Header-only compatibility analysis records
`shape_only_compatibility` instead of pretending that salience can be known without weights.

`I6_strict_structured` adds a semantic attention gate. It reads hidden width, query heads, KV
heads, head dimension, and RoPE base from the checkpoint config. Missing geometry fails closed;
any mismatch leaves selected attention tensors random and records `skipped_semantic`. This matters
for the pinned Qwen source: Q and O matrix shapes match the default student even though Qwen uses
12/2 heads, 128-dimensional heads, and RoPE base 1,000,000 while the default student uses 24/8,
64-dimensional heads, and RoPE base 10,000.

## Matched experiment

Compile the five-arm by three-replicate suite:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_initialization_sweep.yaml \
  --dry-run
```

The 15 runs compare fully random, vision-only, language-only, dual-tower, and half-depth selective
initialization. Within each replicate they hold synthetic and public data, the 20B effective-token
budget, teacher targets, SFT, RLVR, evaluation sampling, and every stochastic seed fixed. The same
Hub cache serves all arms and replicates.

The promotion contract requires three complete paired replicates, a Bonferroni-corrected
heldout-score lower bound above 0.005, all six deployment gates, and simultaneous non-regression on
both `L1-locate` and `L1-region`. Mean-score rank alone cannot promote a pretrained initialization
arm. This keeps selective transfer a measured sample-efficiency intervention rather than an
assumed default.

This suite isolates initialization at the full data scale. Sample efficiency is tested separately
by the executable 45-run initialization-by-data-scale design in
[`student_factorial_runner.md`](student_factorial_runner.md). It holds optimization tokens and
heldout documents fixed, records actual training rows, and reports paired
difference-in-differences rather than inferring low-data behavior from this suite.

Run the focused exact-versus-structured estimand with:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_structured_mlp_transfer_sweep.yaml \
  --dry-run
```

This six-run suite holds both pinned sources, depth fraction, data, optimization, post-training,
evaluation, and all stochastic seeds fixed. Only `I4_selective`'s exact shape policy versus
`I5_structured_mlp`'s joint MLP channel reduction changes. Promotion requires three paired
replicates, the same six deployment gates, and simultaneous non-regression on locating, region
grounding, multilingual, OCR, and reading-order axes.

The geometry-by-transfer factorial is documented in
[`student_attention_geometry_transfer_factorial.md`](student_attention_geometry_transfer_factorial.md).
It uses a strict transfer arm and paired linear contrasts to distinguish a generally better
attention architecture from an architecture that specifically receives the pinned teacher better.
