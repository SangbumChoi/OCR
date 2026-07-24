# Attention geometry by transfer factorial

## Problem

Equal projection shapes do not prove equal attention semantics. The pinned Qwen source and the
default student both use hidden width 1536, so Q and O weights have matching matrix shapes.
However, their head decomposition and rotary geometry differ:

| Profile | Query heads | KV heads | Head dimension | RoPE base | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| native student | 24 | 8 | 64 | 10,000 | 799,919,884 |
| pinned Qwen / aligned student | 12 | 2 | 128 | 1,000,000 | 781,820,172 |

Copying the native profile's same-shape Q and O matrices silently changes the channel groups to
which RoPE and grouped-query attention are applied. The strict transfer arm therefore treats
hidden width, query heads, KV heads, head dimension, and RoPE base as one semantic contract.

## Strict transfer gate

`I6_strict_structured` retains the complete-group structured SwiGLU reduction but copies language
attention tensors only when all five geometry fields match the checkpoint config. Missing source
config fails before transfer. A mismatch skips the complete selected attention group and records
each tensor under `skipped_semantic` with `attention_geometry_mismatch`.

Header-only analysis of the same half-depth Qwen source gives:

| Student geometry | Attention compatible | Copied tensors | Copied parameters | Language fraction | Semantic skips |
| --- | --- | ---: | ---: | ---: | ---: |
| native 24/8 | no | 61 | 226,530,816 | 33.4% | 96 |
| Qwen-aligned 12/2 | yes | 145 | 292,615,680 | 44.4% | 0 |

Both profiles have zero residual weight-shape mismatches under structured MLP transfer. The aligned
profile also removes 18,099,712 deployed parameters by reducing KV projection and cache width.

## Causal design

The 12-run suite crosses two geometry levels with two initialization levels over three paired
replicates:

| Cell | Geometry | Initialization |
| --- | --- | --- |
| `native_random` | native 24/8, RoPE 10,000 | random |
| `qwen_random` | Qwen-aligned 12/2, RoPE 1,000,000 | random |
| `native_strict_transfer` | native | strict structured |
| `qwen_strict_transfer` | Qwen-aligned | strict structured |

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_attention_geometry_transfer_factorial.yaml \
  --dry-run
```

The sweep declares five paired linear contrasts. The primary interaction is:

```text
(qwen_strict_transfer - qwen_random)
- (native_strict_transfer - native_random)
```

A positive heldout interaction means source alignment specifically increases transfer benefit; it
does not merely improve a random model. The runner also reports geometry effects at both
initialization levels and transfer effects at both geometries, with deterministic paired bootstrap
intervals for headline metrics, capability axes, robustness slices, and efficiency counters.

## Decision rule

Do not select the aligned architecture from rank alone. Require:

1. the aligned random cell not to regress the guarded document axes;
2. the aligned strict-transfer cell to pass all deployment gates;
3. a non-negative geometry-by-transfer interaction on heldout score;
4. no regression on `L1-locate`, `L1-region`, multilingual, OCR, or reading order;
5. measured latency and KV-cache evidence consistent with the analytical parameter reduction.

This factorial tests source-aligned attention geometry. It does not establish that one source
geometry generalizes to every future teacher; changing the initialization source requires rerunning
the geometry audit.
