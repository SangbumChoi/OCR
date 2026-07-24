# Selective initialization acquisition and sweep

The native student can start fully random or inherit only exact-shape tensors from immutable
pretrained checkpoints. This path is part of the experiment DAG: source acquisition, revision
validation, file provenance, transfer, and the resulting report are not manual setup steps.

## Pinned sources

The matched initialization sweep uses:

| Component | Source | Immutable revision | Why it is useful | Known mismatch |
| --- | --- | --- | --- | --- |
| vision | [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224/tree/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed) | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | 768-wide, 12-layer, 12-head ViT with 3072-wide MLP blocks | 16-pixel patch kernel and source position table do not match the student's 14-pixel/4096-token inputs |
| language | [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) | `8faed761d45a263340a0528343f099c05c9a4323` | 1536 hidden width and multilingual decoder representations | KV-head geometry and 8960-wide MLP differ; external vocabulary rows are not copied without an identity map |

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
  --family llama --component language --fraction 0.5
```

At the pinned revisions, the `I4_selective` half-depth policy finds 87 vision tensors containing
38,985,984 parameters and 61 language tensors containing 56,679,936 parameters. The combined
95,665,920 parameters are about 12% of the student. These are compatibility counts, not evidence of
quality improvement; the matched sweep must establish that.

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
depth-maps the selected blocks, and copies only exact-shape tensors. The LFM2 adapter covers
attention, gated short convolution, norms, and SwiGLU projections for hybrid students. It records
copied keys and parameters,
missing source keys, and shape mismatches in `artifacts/initial/metadata.json`. A non-random arm
fails if any required component copies zero parameters. The connector remains random in all
shipped arms.

Token embeddings and the tied output head require an explicit target-to-source token identity map.
Matching width or row count alone is not accepted as proof that two vocabulary rows mean the same
thing.

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

This suite isolates initialization at the full data scale. Sample efficiency is tested separately
by the executable 45-run initialization-by-data-scale design in
[`student_factorial_runner.md`](student_factorial_runner.md). It holds optimization tokens and
heldout documents fixed, records actual training rows, and reports paired
difference-in-differences rather than inferring low-data behavior from this suite.
