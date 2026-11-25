# Selective-transfer source matrix

This matrix composes pinned architecture compatibility, bounded real-weight health sketches, and available real-payload execution evidence. It never treats similar weight distributions as neuron-basis alignment.

## Decisions

### `docvlm-800m`

| Source | Direct | Structured | Token map | Payload preflight | Distill | Real payload |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smolvlm2-500m` | 1 | 0 | 0 | 0 | 7 | `verified` |
| `fastvlm-0.5b` | 0 | 0 | 0 | 0 | 8 | `not_applicable` |
| `florence-2-base` | 0 | 0 | 0 | 0 | 8 | `not_applicable` |
| `internvl3-1b` | 0 | 0 | 0 | 1 | 7 | `not_applicable` |
| `lfm2.5-vl-1.6b` | 0 | 0 | 0 | 0 | 8 | `not_applicable` |

### `docvlm-lfm-aligned-814m`

| Source | Direct | Structured | Token map | Payload preflight | Distill | Real payload |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lfm2.5-vl-1.6b` | 2 | 1 | 1 | 0 | 4 | `verified` |
| `smolvlm2-500m` | 1 | 0 | 0 | 0 | 7 | `verified` |
| `fastvlm-0.5b` | 0 | 0 | 0 | 0 | 8 | `not_applicable` |
| `florence-2-base` | 0 | 0 | 0 | 0 | 8 | `not_applicable` |
| `internvl3-1b` | 0 | 0 | 0 | 1 | 7 | `not_applicable` |

## Research decision

- The native 800M target has no sampled-and-topology-qualified language copy source in this five-model set. Use language feature, relation, or sequence distillation unless a separate pairwise source preflight passes.
- The LFM-aligned 814M target makes LFM attention and short convolution direct-copy candidates and its reduced SwiGLU a structured-transfer candidate. The recorded real-payload run verifies this language transfer at 80.49% coverage with zero shape, semantic, or missing-source skips.
- SmolVLM2 is a vision-block candidate for both targets, but it remains separately verified at payload level. A checkpoint combining Smol vision and LFM language still requires matched training evidence.
- Position weights without sampled semantic-role evidence require a pairwise payload preflight even when the config convention matches.
- Token embeddings remain identity-map gated. Vocabulary width equality does not establish token identity.

## Claim boundary

This artifact selects experiments; it does not establish downstream quality or authorize promotion. Direct and structured candidates still require pairwise payload checks, and empirical benefit requires matched random-initialized controls.

Report fingerprint: `sha256:ff03a56387d2375e96c99beee4463ebcb83f29846e7f5d5ea6cc28a8dd6df66b`.
