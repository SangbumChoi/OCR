# SmolVLM2 vision-block transfer pilot

## Question

Does SmolVLM2 contribute a transferable visual representation to the LFM-aligned sub-1B student
when the language initialization, data, optimization budget, and evaluation policy are fixed?

Shape equality alone is not enough. SmolVLM2 uses a 16-pixel patch interface while the student
uses 14-pixel patches. An unrestricted shape-based preflight also admitted the source patch bias
and final normalization tensors even though those tensors belong to different visual interfaces.
The transfer arm therefore uses `vision_scope: transformer_blocks` and accepts only canonical
`vision.blocks.*` tensors. Patch projection, position encoding, final normalization, and connector
tensors remain target-initialized.

## Executed payload evidence

The pinned source is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at commit
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The selected transformer blocks form one contiguous
safetensors interval:

| Property | Value |
| --- | ---: |
| Selected tensors | 192 |
| Selected parameters | 85,054,464 |
| Source payload bytes | 340,217,856 |
| Target vision coverage | 95.94% |
| Shape, semantic, or missing-source skips | 0 |

The real-source audit materialized that interval, loaded its actual values, and verified the copied
target tensors. The compact evidence is
[`selective_transfer_smol_vision_real_source_preflight.json`](../results/selective_transfer_smol_vision_real_source_preflight.json).
This establishes the initialization contract only, not downstream quality.

## Bounded acquisition

Hub initialization sources can now declare `tensor_prefixes`. The experiment DAG uses exact HTTP
range acquisition for a single contiguous safetensors interval and writes a hash-verified
checkpoint manifest understood by normal resume and placeholder resolution. The pilot downloads
the selected 340 MB payload instead of the complete roughly 2.03 GB source checkpoint. Stage logs
print only the tensor count, payload bytes, output path, and manifest fingerprint; they do not
repeat tensor tables or model payloads.

## Matched pilot

[`sub1b_smol_vision_transfer_pilot.yaml`](../../configs/sub1b_smol_vision_transfer_pilot.yaml)
compiles two one-seed cells:

| Cell | Language initialization | Vision initialization |
| --- | --- | --- |
| `lfm_language_only` | strict LFM exact and structured transfer | random |
| `lfm_smol_dual` | same strict LFM transfer | exact SmolVLM2 transformer blocks |

Both cells acquire the same pinned sources and share the LFM-aligned 814M geometry, data,
initialization seed, synthetic splits, teacher dose, pretraining/SFT/RLVR steps, and heldout
evaluation. The sole contrast is `lfm_smol_dual - lfm_language_only`. The configuration has no
promotion block; one seed is screening evidence only.

Long table, HTML, full-page, transcription, and reading-order outputs retain the production
generation safeguards. Exact trailing token cycles terminate with EOS, task labels receive bounded
token horizons up to the fixed hard cap, and the generation-stability gate rejects repetition,
max-token termination, malformed structure, or score regressions. HTML rendering separately
requires all-page coverage, table-cell survival, and a bounded canvas, so a large token allowance
cannot conceal missing pages or unreadable downscaling.

## Decision rule

Advance to a sealed multi-seed run only if the dual cell passes all execution and generation
stability gates and improves heldout document perception without degrading grounding, reasoning,
multilingual behavior, or reliability. Until the matched run completes, SmolVLM2 remains a
payload-verified candidate rather than a promoted source.
