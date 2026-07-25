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

## Structural transfer preflight

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

## Executed pretrained payload transfer

The aligned 814,207,243-parameter target was also materialized on CPU with the pinned
`LiquidAI/LFM2.5-VL-1.6B` safetensors payload. Acquisition verified 3,193,336,592 bytes at the
declared revision before initialization.

The real run copied 553,748,992 language parameters, or 80.49% of the 687,966,720-parameter
language component, above the 50% arm floor. This included 109 tensors and 377,487,360 parameters
from 36 structured SwiGLU tensors across all 12 target blocks. Attention geometry, short
convolution, and MLP operator checks passed, with zero shape, semantic, or missing-source skips.
Copied-value verification passed. The run took 21.92 seconds and reached 6,469,451,776 bytes
maximum resident memory on a 32GB Apple M5 host.

This proves that real pretrained LFM payloads can initialize the sub-1B aligned target; it does not
prove downstream quality or target-CUDA feasibility. The compact evidence is
[`selective_transfer_lfm_real_source_preflight.json`](../results/selective_transfer_lfm_real_source_preflight.json).

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
The shared visual-backend benchmark and adaptive synthesis are disabled. The strict-transfer cell
alone runs the production-size 2,048-token full forward/backward/AdamW deployment gate before
synthetic generation or teacher inference. The geometry-identical random cell does not repeat that
expensive check. The pilot is intended to catch initialization failures, unstable losses, memory
surprises, generation loops, and catastrophic long-context, table, or reasoning regressions before
spending the three-seed budget. Visual-backend latency remains a confirmatory-sweep gate rather
than a pilot claim.

Run its compilation check with:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_lfm_language_transfer_pilot.yaml \
  --dry-run
```

The pilot has no promotion block. Its single-seed deltas are screening signals only and must not be
combined with the confirmatory replicates or used to select a deployment model.

### Submission readiness

[`scripts/audit_lfm_transfer_pilot.py`](../../scripts/audit_lfm_transfer_pilot.py) compiles the
current pilot and checks it against the executed real-payload preflight before GPU submission. The
compact fail-closed artifact records decisions rather than full commands, patch lists, checkpoint
tensors, HTML, or target text. Its 14 checks cover:

- the one-seed, non-promotional three-cell design and matched LFM geometry;
- the sub-1B parameter bound and exact random-versus-transfer initialization contrast;
- the immutable Hub revision and its match to the executed safetensors identity;
- the realized 80.49% language transfer, operator compatibility, zero skips, and copied-value
  verification;
- end-to-end pretraining, SFT, RLVR, baseline, and final evaluation stages;
- strict-cell-only target-CUDA feasibility preflight;
- the CUDA, native-BF16, FlexAttention, and non-reentrant full-component checkpointing contract;
- explicit `sbdc/docvlm-ablation` tracking for pretraining, SFT, RLVR, matched baseline/final
  evaluation, and the strict-cell CUDA preflight;
- matched task-aware token budgets, exact-cycle termination, task-balanced sampling, and the
  table/full-page generation-stability release gate;
- bounded screening data and optimization budgets.

The current result is `pass` and authorizes pilot submission only:
[`lfm_selective_transfer_pilot_readiness.json`](../results/lfm_selective_transfer_pilot_readiness.json).
It explicitly does not authorize a target-CUDA feasibility, quality, or promotion claim. The
target-GPU stage still has to execute successfully, and quality still requires the sealed
three-seed confirmatory sweep.

For Colab execution, use
[`notebooks/lfm_selective_transfer_pilot.ipynb`](../../notebooks/lfm_selective_transfer_pilot.ipynb).
Its launcher checks the same readiness fingerprint, W&B credentials, free disk, CUDA availability,
native BF16 support, and GPU memory before execution. T4 is rejected rather than silently changing
the experiment to FP16; use L4, A10, A100, or newer hardware. Full subprocess output is retained in
`colab_pilot.log`; the notebook prints only state changes, five-minute heartbeats, and a compact
final summary.

The confirmatory compiler emits 63 unique tracked stage names across nine runs, all grouped under
`docvlm-lfm-language-transfer-sweep` in
[`sbdc/docvlm-ablation`](https://wandb.ai/sbdc/docvlm-ablation). The pilot uses the separate
`docvlm-lfm-language-transfer-pilot` group, so screening metrics cannot be mistaken for
confirmatory replicates.

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
