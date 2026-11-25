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

The cross-family value analysis is separate from this shape preflight. Run
[`scripts/analyze_small_vlm_weights.py`](../../scripts/analyze_small_vlm_weights.py) to range-read
bounded samples from five pinned public checkpoints. The resulting
[`small_vlm_weight_commonality.md`](small_vlm_weight_commonality.md) finds stable normalized scales
for recurrent vision attention and MLP roles, while language attention and SwiGLU scales vary too
widely to serve as an architecture-agnostic prior. This absence of a population prior does not
override a healthy pairwise transfer that passes the stricter semantic and geometry checks.

Run
[`scripts/build_selective_transfer_source_matrix.py`](../../scripts/build_selective_transfer_source_matrix.py)
to compose those sampled values with the pinned topology analysis and available real-payload
evidence. The resulting
[`selective_transfer_source_matrix.md`](selective_transfer_source_matrix.md) finds no qualified
language-copy source for the native 800M operator among the five audited models. For the
LFM-aligned 814M operator, it selects LFM attention and short convolution as direct-copy candidates
and the reduced SwiGLU as a structured-transfer candidate; the existing real-payload preflight
verifies that path. SmolVLM2 vision blocks remain an unexecuted pairwise candidate rather than
evidence for a tested dual-source initialization.

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

Acquisition inspects the pinned repository file list before downloading weights. It selects
exactly one representation in this order: sharded safetensors, single-file safetensors, sharded
PyTorch bin, then single-file PyTorch bin. The manifest records the selected format and exact Hub
allow patterns. This prevents repositories that publish both formats from doubling network,
cache, hashing, and provenance cost. Resume checks use file presence and byte size; immediately
before an initialization command resolves the checkpoint path, every manifest SHA-256 is
recomputed. Same-size source tampering therefore fails before model allocation and transfer.

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
hybrid students. Hybrid depth reduction maps attention targets only to attention sources and
short-convolution targets only to short-convolution sources. Metadata records copied keys and
parameters, missing source keys, and shape
mismatches in `artifacts/initial/metadata.json`. It also records canonical source and target
topology SHA-256 values and an ordered mapping manifest with source key, target key, shape, dtype,
copy method, and copied parameter count for every transferred tensor. Exact, token-row, and
structured-MLP mappings are distinguishable; the latter two carry selection fingerprints.
Immediately before loading, the builder hashes every consumed source config, index, and weight
file. It also hashes each target-dtype copy payload, writes it, and verifies the same value hash
from the target tensor. The source-file identity, per-mapping value hashes, and their aggregate
hash are sealed into the transfer report. A non-random arm fails if any required component copies
zero parameters. It also checks the realized
copied parameters against a component-relative floor declared by the arm:

Before copying, every shipped transfer arm also computes a bounded in-memory role sketch from the
materialized source. The report stores aggregate finite, scale, sparsity, sign, and outlier
statistics plus selection and sample fingerprints, never raw values. A sampled unhealthy role is
left random and recorded as `unhealthy_source_weight_role`; the component-dose floor then prevents
silent success after excessive rejection.

| Arm | Minimum vision dose | Minimum language dose |
| --- | ---: | ---: |
| `I1_vision` | 80% | n/a |
| `I2_language` | n/a | 15% |
| `I3_dual` | 80% | 15% |
| `I4_selective` | 40% | 7.5% |
| `I5_structured_mlp` | 40% | 25% |
| `I6_strict_structured` | 40% | 25% |
| `I8_lfm_aligned_language` | n/a | 50% |

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

Saving the initialized model validates every mapping and seals the arm, seed, runtime architecture
fingerprint, source-file identities, copied-value hashes, and complete transfer reports into
`initialization_lineage` schema version 2. The lineage is copied unchanged into pretraining, SFT,
preference, and RLVR checkpoints. Native load rejects a modified lineage, source identity, mapping,
or copied-value fingerprint; exact resume rejects lineage drift; and experiment evidence requires
the use-time source identity to match the planned local checkpoint or pinned acquisition manifest.
Legacy schema-version-1 lineages remain loadable but are insufficient execution evidence.

`I6_strict_structured` and `I8_lfm_aligned_language` add semantic operator gates. They read hidden
width, query heads, KV heads, head dimension, RoPE base and layout, norm epsilon, Q/K
normalization, projection bias, MLP bias, short-convolution kernel, and convolution bias from the
checkpoint config. Missing geometry fails closed; any mismatch leaves the affected tensors random
and records `skipped_semantic`. This matters even when matrix shapes happen to match: changing
RoPE channel layout or omitting Q/K normalization changes the operator implemented by copied
weights.

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

The LFM-specific aligned profile and three-arm paired sweep are documented in
[`student_lfm_language_transfer_sweep.md`](student_lfm_language_transfer_sweep.md). Its meta-device
preflight executes the real adapter and proves an 80.49% language-parameter transfer dose before
checkpoint payloads are downloaded.

## Executed CPU transfer contract

[`configs/sub1b_experiment_selective_tiny.yaml`](../../configs/sub1b_experiment_selective_tiny.yaml)
materializes deterministic cross-architecture fixture checkpoints inside the experiment DAG.
The vision source has three layers while the target copies the selected depth into its smaller
component. The language source has three layers and a 320-wide MLP while the target uses a
different MLP width, forcing the `structured_mlp` channel-selection path.

The verified 2026-07-25 run copied 50,048 vision parameters, or 59.92% of the target vision
component, above its 40% floor. It copied 205,632 language parameters, including 98,304 through
three structured tensors, for 59.32% of the target language component, above its 25% floor.
Both reports passed copied-value verification and were bound to the generated source checkpoint
content. All 23 stages completed and the evidence attestation passed the execution contract.

This test establishes that selective transfer survives the complete training and feedback loop.
Because both sources contain deterministic random weights and the 587,019-parameter target trains
for one step per phase, its failed capability gate is expected and no quality claim is authorized.
Use the pinned-source matched sweeps above for the causal quality comparison.

The three-seed CPU proxy in
[`configs/sub1b_selective_transfer_fixture_sweep.yaml`](../../configs/sub1b_selective_transfer_fixture_sweep.yaml)
compares this fixture transfer against random initialization with matched generated documents and
evaluation samples. Its purpose is to execute the paired statistical and attestation path, not to
promote the fixture arm. The compact committed result is
[`selective_transfer_fixture_sweep.json`](../results/selective_transfer_fixture_sweep.json).

## Executed real-source materialization

The production 799,919,884-parameter target was also materialized on CPU with the pinned SigLIP
and Qwen checkpoints above under `I5_structured_mlp`. Acquisition selected only safetensors and
verified 812,672,752 SigLIP bytes and 3,087,467,828 Qwen bytes before transfer.

The real transfer copied 42,529,536 vision parameters, or 47.97% of the target vision component,
above its 40% floor. It copied 283,172,352 language parameters, or 41.80% of the target language
component, above its 25% floor; 226,492,416 parameters came from 36 structured SwiGLU tensors.
Both components passed copied-value verification. The final run took 19.81 seconds and reached
5,454,725,120 bytes maximum resident memory on a 32GB Apple M5 host.

This is stronger than header compatibility or random-fixture evidence because real pretrained
payloads were loaded and copied into the production target. It still establishes initialization
feasibility, not downstream quality or CUDA training feasibility. The compact evidence is
[`selective_transfer_real_source_preflight.json`](../results/selective_transfer_real_source_preflight.json).

When `build_sub1b_student.py` materializes transfers without saving a checkpoint, stdout is compact
by default: it preserves component doses, operator compatibility, skip counts, and every
attestation fingerprint while omitting repeated copied-key and tensor-mapping arrays. Pass
`--full-transfer-report` only for a targeted tensor-level audit. Saved checkpoint metadata always
retains the complete transfer report.

The separate LFM-aligned target has also completed real-payload transfer under
`I8_lfm_aligned_language`; see
[`student_lfm_language_transfer_sweep.md`](student_lfm_language_transfer_sweep.md) and its compact
[`preflight evidence`](../results/selective_transfer_lfm_real_source_preflight.json).
