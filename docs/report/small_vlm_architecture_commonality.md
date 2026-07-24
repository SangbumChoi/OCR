# Small-VLM architecture commonality

This report compares pinned public configs before any selective weight copy. A compatible tensor shape is necessary but not sufficient: module semantics, attention geometry, position convention, and tokenizer identity are checked separately.

## Compared models

| Model | Parameters | Vision | Connector | Language |
| --- | ---: | --- | --- | --- |
| [smolvlm2-500m](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/blob/7b375e1b73b11138ff12fe22c8f2822d8fe03467/config.json) | 500M | vit / global_attention | pixel_shuffle_projection | llama / global_attention |
| [fastvlm-0.5b](https://huggingface.co/apple/FastVLM-0.5B/blob/16375720c2d673fa583e57e9876afde27549c7d0/config.json) | 500M | fastvithd / hybrid_convolution_attention | mlp2x | qwen2 / global_attention |
| [florence-2-base](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/config.json) | 230M | davit / hybrid_window_attention_convolution | linear_projection | bart / global_attention |
| [internvl3-1b](https://huggingface.co/OpenGVLab/InternVL3-1B-hf/blob/014c0583a0d4bedf29fbe2dbff4f865eb998e171/config.json) | 1000M | vit / global_attention | pixel_shuffle_mlp | qwen2 / global_attention |
| [lfm2.5-vl-1.6b](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/919fde3d022e3f90a4716006f993938ee8c2eb97/config.json) | 1600M | vit / global_attention | downsample_mlp | lfm2 / hybrid_short_convolution_attention |

## Common characteristics

| Feature | Modal value | Prevalence | Target match |
| --- | --- | ---: | --- |
| `vision.family` | `"vit"` | 3/5 | yes |
| `vision.mixer` | `"global_attention"` | 3/5 | yes |
| `vision.norm` | `"layer_norm"` | 5/5 | yes |
| `vision.activation` | `"gelu"` | 5/5 | yes |
| `vision.position` | `"learned_absolute_2d"` | 4/5 | yes |
| `vision.dynamic_resolution` | `true` | 4/5 | yes |
| `connector.activation` | `"gelu"` | 3/5 | no |
| `language.mode` | `"decoder_only"` | 4/5 | yes |
| `language.mixer` | `"global_attention"` | 4/5 | yes |
| `language.head_dim` | `64` | 5/5 | yes |
| `language.norm` | `"rms_norm"` | 4/5 | yes |
| `language.activation` | `"swiglu"` | 4/5 | yes |
| `language.position` | `"rope"` | 4/5 | yes |
| `language.rope_base` | `1000000` | 3/5 | no |

## Transfer preflight

| Source | Compatible subcomponents | Exact | Structured | Token rows | Distill only |
| --- | ---: | ---: | ---: | ---: | ---: |
| [internvl3-1b](https://huggingface.co/OpenGVLab/InternVL3-1B-hf/blob/014c0583a0d4bedf29fbe2dbff4f865eb998e171/config.json) | 1/7 | 1 | 0 | 0 | 6 |
| [smolvlm2-500m](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/blob/7b375e1b73b11138ff12fe22c8f2822d8fe03467/config.json) | 1/7 | 1 | 0 | 0 | 6 |
| [fastvlm-0.5b](https://huggingface.co/apple/FastVLM-0.5B/blob/16375720c2d673fa583e57e9876afde27549c7d0/config.json) | 0/7 | 0 | 0 | 0 | 7 |
| [florence-2-base](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/config.json) | 0/7 | 0 | 0 | 0 | 7 |
| [lfm2.5-vl-1.6b](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/919fde3d022e3f90a4716006f993938ee8c2eb97/config.json) | 0/7 | 0 | 0 | 0 | 7 |

## Decision rules

- Copy attention only when hidden width, query heads, KV heads, head dimension, normalization, RoPE convention, and RoPE base all match.
- Reduce an MLP only as one complete SwiGLU group with one shared channel selection; never crop independent matrices.
- Copy token embeddings only through an explicit tokenizer identity map.
- Treat position embeddings, hybrid-convolution blocks, encoder-decoder text stacks, and non-identical connectors as distillation targets rather than weight-copy targets.
- Run an initialization factorial against random, vision-only, language-only, dual, and selective controls; this report establishes compatibility, not downstream benefit.
