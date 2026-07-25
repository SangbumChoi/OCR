# Cross-architecture weight commonality

This report samples real weights from immutable public checkpoints without downloading full model files. Each tensor contributes at most three evenly spaced byte windows. Raw values are discarded; only aggregate statistics and content fingerprints are retained.

These statistics compare operator distributions, not neuron coordinates. A similar scale never establishes basis alignment and cannot by itself authorize a direct copy.

## Evidence budget

| Model | Roles | Tensors | Values | Bytes read |
| --- | ---: | ---: | ---: | ---: |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 18 | 48 | 91200 | 364,808 |
| `apple/FastVLM-0.5B` | 14 | 39 | 72160 | 144,328 |
| `microsoft/Florence-2-base` | 10 | 30 | 52608 | 105,224 |
| `OpenGVLab/InternVL3-1B-hf` | 17 | 46 | 87680 | 175,368 |
| `LiquidAI/LFM2.5-VL-1.6B` | 19 | 52 | 103808 | 207,624 |

## Recurrent weight characteristics

| Semantic role | Models | Scaled RMS ratio | Median zeros | Stable | Transfer rule |
| --- | ---: | ---: | ---: | --- | --- |
| `connector.projection` | 4 | 3.620 | 0 | yes | `exact_only_with_identical_connector_topology` |
| `language.attention.k` | 5 | 8.889 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.attention.o` | 5 | 8.484 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.attention.q` | 5 | 7.255 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.attention.v` | 5 | 9.641 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.mlp.down` | 4 | 6.075 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.mlp.gate` | 4 | 6.941 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.mlp.up` | 4 | 8.010 | 0 | no | `pairwise_preflight_no_population_prior` |
| `language.norm` | 5 | 3.038 | 0 | yes | `exact_only_with_semantic_and_geometry_match` |
| `language.token_embedding` | 4 | 7.563 | 0 | no | `pairwise_preflight_no_population_prior` |
| `vision.attention.k` | 3 | 1.315 | 0 | yes | `exact_only_with_semantic_and_geometry_match` |
| `vision.attention.q` | 3 | 1.284 | 0 | yes | `exact_only_with_semantic_and_geometry_match` |
| `vision.attention.v` | 3 | 1.315 | 0 | yes | `exact_only_with_semantic_and_geometry_match` |
| `vision.mlp.in` | 5 | 2.040 | 0 | yes | `exact_or_joint_structured_channel_selection` |
| `vision.mlp.out` | 5 | 2.153 | 0 | yes | `exact_or_joint_structured_channel_selection` |
| `vision.norm` | 5 | 1.686 | 0 | yes | `exact_only_with_semantic_and_geometry_match` |
| `vision.patch_embedding` | 4 | 15.187 | 0 | no | `pairwise_preflight_no_population_prior` |

## Selective-transfer contract

- Exact copy requires a stable sampled source plus matching semantic role, shape, normalization, attention geometry, and position convention.
- Wider SwiGLU transfer uses one joint channel selection across gate, up, and down matrices; independent cropping is forbidden.
- Token embeddings require an explicit tokenizer identity map.
- Cross-model scale instability removes the population prior but does not veto a healthy pairwise transfer that passes the full semantic and geometry preflight.
- A source role with non-finite, degenerate, sparse, or extreme sampled weights is distillation-only.
- Topology mismatches remain feature or relation distillation targets even when their aggregate weight scales look similar.

Report fingerprint: `sha256:f7adced9840e0bba8560e97dfdda8bd6fab54cd04d0b46696b409652497ba384`.
