# Compute-matched language-mixer sweep

## Research question

The original native student used full grouped-query attention in all 23 decoder
layers. That is a strong control, but it does not test whether global document
binding can be concentrated in fewer layers while local operators handle
token-scale patterns.

The optional hybrid follows the public LFM2 operator contract without claiming
architectural identity. A short-convolution block applies an input projection,
multiplicative input gate, causal depthwise convolution, output gate, output
projection, and the same SwiGLU feed-forward block as the attention control.
The public
[LFM2 configuration](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/config.json)
uses explicit full-attention layer indices and a three-token convolution cache;
the upstream
[Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py)
is the semantic reference for the gated causal operator and recurrent state.

## Adjustable contract

`student.language.full_attention_layers` is either `null`, meaning every layer
uses GQA, or a sorted list of zero-based full-attention indices. Every remaining
layer uses gated short convolution. `conv_kernel_size` controls its causal
receptive field and `conv_bias` controls all convolution-block biases.

The all-attention default is intentionally unchanged. A hybrid architecture is
selected only through an explicit blueprint patch, so old checkpoints and the
control arm retain the same parameter graph and fingerprint.

During incremental generation:

- attention layers retain compact native-head K/V buffers at the configured
  completion capacity;
- convolution layers retain only `kernel_size - 1` projected states per channel;
- both cache types are prefilled once and then updated in place for one-token
  decoding;
- cached and full-prefix logits are tested for numerical parity.

## Paired profiles

[`configs/sub1b_language_mixer_compute_sweep.yaml`](../../configs/sub1b_language_mixer_compute_sweep.yaml)
compares four profiles:

| Arm | Full-attention layers | Conv kernel | Parameters | Peak bf16 generation cache |
| --- | ---: | ---: | ---: | ---: |
| `all_attention` | 23 | n/a | 799,919,884 | 160.6 MiB |
| `alternating_k3` | 12 | 3 | 834,528,524 | 84.3 MiB |
| `lfm_ratio_k3` | 8 | 3 | 847,113,484 | 56.6 MiB |
| `lfm_ratio_k5` | 8 | 5 | 847,159,564 | 57.3 MiB |

The cache calculation uses a 64-token visual prefix, 256 prompt tokens, 128
completion tokens, and rollout group size eight. Short convolution has more
projection parameters per layer than compact GQA, so fewer attention layers do
not imply fewer parameters. All profiles remain below the strict one-billion
deployment limit.

The architecture compiler derives one pretraining, SFT, and RLVR student-FLOP
budget from `all_attention`, then patches those same integer budgets into all
four arms. This avoids comparing equal steps when mixer FLOPs differ. Each arm
uses the three paired seed blocks from the core sweep, producing 12 complete
DAGs. The usual 2% realized-compute overshoot gate remains mandatory.

## Run

Compile without downloading data or checkpoints:

```bash
python scripts/run_student_architecture_sweep.py \
  --sweep configs/sub1b_language_mixer_compute_sweep.yaml \
  --dry-run
```

Run one paired cell:

```bash
python scripts/run_student_architecture_sweep.py \
  --sweep configs/sub1b_language_mixer_compute_sweep.yaml \
  --variant lfm_ratio_k3 \
  --replicate seed_0
```

Run or resume the complete rectangle:

```bash
python scripts/run_student_architecture_sweep.py \
  --sweep configs/sub1b_language_mixer_compute_sweep.yaml
```

W&B receives `architecture-profile:<id>` and
`compute-matched-architecture` tags. Compare held-out score,
train-minus-heldout gap, long-context document families, counterfactual
reasoning, realized steps, latency, and cache memory. Parameter and analytical
memory advantages are not evidence of quality; no hybrid arm should replace
the all-attention default until the paired held-out suite passes the same
deployment gates.

The machine-readable Pareto contract permits at most 50M additional parameters while requiring a
simultaneous reduction in analytical forward FLOPs or peak bf16 RLVR KV-cache bytes. Heldout score
must remain within 0.005, the train-minus-heldout gap may not widen, and all declared capability
guardrails and deployment gates must pass. Pareto-dominated hybrids are removed before the
lexicographic quality, cache, FLOP, and parameter preference is applied. The realized compute
budget remains an external fail-closed gate on the final selection.

## Initialization

Selective transfer accepts `language_family: lfm2` for both text-only LFM2 and
LFM2-VL wrappers, in addition to native and Llama-style sources. It maps
official attention, short-convolution, norm, and
feed-forward names into native semantic names, then applies the existing
depth-fraction and exact-shape gates. It never crops hidden dimensions or copies
vocabulary rows without an identity map. A width-incompatible LFM2 checkpoint
therefore remains a sequence or feature teacher rather than silently becoming
an initialization source.
