# LFM language-operator transfer sweep

## Question

The default 1536-wide student and LFM2.5 share broad traits such as RMSNorm, SwiGLU, grouped-query
attention, and RoPE, but those labels do not make their tensors interchangeable. LFM2.5 uses width
2048, 32 query heads, 8 KV heads, per-head Q/K RMSNorm, half-split RoPE, bias-free projections,
RMSNorm epsilon `1e-5`, and a hybrid sequence of attention and gated short-convolution layers.

The experiment asks whether matching that complete operator contract makes selective language
transfer useful under the same document data, optimization, and evaluation pipeline.

## Sub-1B aligned profile

The aligned control changes only the language-facing geometry and connector output width:

| Component | Native | LFM-aligned |
| --- | ---: | ---: |
| Language width | 1536 | 2048 |
| Layers | 23 attention | 12 hybrid |
| Full-attention indices | all | 2, 5, 8, 10 |
| Query / KV heads | 24 / 8 | 32 / 8 |
| Head dimension | 64 | 64 |
| SwiGLU width | 4096 | 5120 |
| RoPE | interleaved, base 10,000 | half-split, base 1,000,000 |
| Q/K normalization | none | RMSNorm |
| Projection bias | enabled | disabled |
| Total VLM parameters | 799,919,884 | 814,207,243 |
| Forward FLOPs at 2,048 text + 2,520 visual tokens | 4.184T | 3.738T |
| 2,176-token language state | 102,498,304 bytes | 17,891,328 bytes |

The 5,120-channel target MLP is a structured reduction of the 12,288-channel LFM source. The
document vision tower remains unchanged, and the connector remains randomly initialized. The
hybrid profile uses only four full-attention layers, so its analytical forward cost is 10.7% lower
and its language generation state is 82.5% smaller than the all-attention native profile despite
the wider hidden state. Runtime latency and peak memory remain measured promotion gates.

## Transfer proof

[`scripts/analyze_small_vlm_architectures.py`](../../scripts/analyze_small_vlm_architectures.py)
constructs both the Transformers LFM source and native target on the meta device. It then executes
the real canonicalization, like-typed depth mapping, strict semantic gates, and structured MLP
selection without allocating checkpoint payloads.

The pinned preflight reports:

- 553,748,992 copied language parameters, or 80.49% of the deployed language stack;
- 73 exact tensors and 36 structured tensors across all 12 SwiGLU groups;
- zero shape skips, semantic skips, and missing source keys;
- compatible attention, short-convolution, and MLP operator contracts;
- a fingerprinted mapping from the 12 target blocks to like-typed source blocks.

The `I8_lfm_aligned_language` arm fails initialization unless at least 50% of deployed language
parameters are copied. Token embeddings remain random unless a separate exact token-identity map is
provided.

## Paired design

[`configs/sub1b_lfm_language_transfer_sweep.yaml`](../../configs/sub1b_lfm_language_transfer_sweep.yaml)
compiles three cells over three paired seeds:

| Cell | Operator | Initialization |
| --- | --- | --- |
| `native_random` | native | random |
| `lfm_random` | LFM-aligned | random |
| `lfm_strict_transfer` | LFM-aligned | strict exact and structured LFM transfer |

Run the complete dry-run compilation with:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_lfm_language_transfer_sweep.yaml \
  --dry-run
```

The paired `lfm_random - native_random` contrast estimates the architecture effect. The paired
`lfm_strict_transfer - lfm_random` contrast estimates the transfer effect under the only operator
contract for which LFM copy is valid. The end-to-end aligned-transfer contrast is secondary. An
incompatible native-transfer cell is deliberately absent because fail-closed initialization would
copy zero parameters and terminate rather than produce a meaningful measurement.

## Screening pilot

[`configs/sub1b_lfm_language_transfer_pilot.yaml`](../../configs/sub1b_lfm_language_transfer_pilot.yaml)
compiles the same three cells for one paired seed with 32 generated training documents, at most
256 public rows, 25 pretraining steps, 10 SFT steps, 5 RLVR steps, and 64 evaluation samples.
Repeated backend feasibility benchmarks and adaptive synthesis are disabled. The pilot is intended
to catch initialization failures, unstable losses, latency or memory surprises, generation loops,
and catastrophic long-context, table, or reasoning regressions before spending the three-seed
budget.

Run its compilation check with:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_lfm_language_transfer_pilot.yaml \
  --dry-run
```

The pilot has no promotion block. Its single-seed deltas are screening signals only and must not be
combined with the confirmatory replicates or used to select a deployment model.

## Decision rule

The confirmatory sweep uses `lfm_random` as the statistical baseline. Only
`lfm_strict_transfer` is eligible for promotion; `native_random` remains an architecture control
in the descriptive results and the `geometry_effect_without_transfer` contrast. This prevents the
architecture control from consuming a promotion hypothesis or weakening the transfer test.

Promote the aligned initialization only if all of the following hold:

1. the 50% realized-transfer gate and every deployment gate pass;
2. the Bonferroni-adjusted one-sided paired-bootstrap lower bound for heldout score exceeds 0.005;
3. the corresponding simultaneous lower bounds for localization, region grounding, comprehension,
   accounting, multilingual OCR, full-page OCR, and reading order are all non-negative;
4. initial-to-final learning progress exceeds the aligned random control rather than only its
   initial score;
5. measured latency and memory remain acceptable for the target deployment despite the wider
   hidden state.

Until those measurements exist, the profile is an executable hypothesis, not the new default.
