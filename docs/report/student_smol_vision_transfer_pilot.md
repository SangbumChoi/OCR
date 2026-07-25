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

Both cells share the same pinned LFM source, LFM-aligned 814M geometry, data, initialization seed,
synthetic splits, teacher dose, pretraining/SFT/RLVR steps, and heldout evaluation. Only the dual
cell acquires the Smol source because the language-only arm never reads vision weights; this avoids
a duplicate 340 MB range request without changing either trained model or its compute budget. The
sole contrast is `lfm_smol_dual - lfm_language_only`. The configuration has no promotion block;
one seed is screening evidence only.

The 256-row public UDD cap uses deterministic task-stratified acquisition with
a 16-row minimum for every observed task. The component manifest records
eligible counts, allocated quotas, and realized counts, and submission
readiness rejects an unstratified fallback. Within those quotas, a deterministic
joint reservation guarantees at least eight English, Korean, Chinese, and
Japanese rows; an unavailable or infeasible language floor stops acquisition.

Long table, HTML, full-page, transcription, and reading-order outputs retain the production
generation safeguards. Exact trailing token cycles terminate with EOS, task labels receive bounded
token horizons up to the fixed hard cap, and the generation-stability gate rejects repetition,
max-token termination, malformed structure, or score regressions. HTML rendering separately
requires all-page coverage, table-cell survival, and a bounded canvas, so a large token allowance
cannot conceal missing pages or unreadable downscaling.

The gated follow-up is the three-seed
[`student_smol_vision_transfer_sweep.md`](student_smol_vision_transfer_sweep.md). Its treatment
replicates share one content-addressed selective checkpoint payload while retaining independent
training state and matched seed controls.

## Decision rule

Advance to a sealed multi-seed run only if the dual cell passes all execution and generation
stability gates and improves heldout document perception without degrading grounding, reasoning,
multilingual behavior, or reliability. Until the matched run completes, SmolVLM2 remains a
payload-verified candidate rather than a promoted source.

## Submission readiness

[`audit_smol_vision_transfer_pilot.py`](../../scripts/audit_smol_vision_transfer_pilot.py)
compiles the exact pilot and binds it to both executed source preflights and the cross-model source
matrix. The compact fail-closed artifact passes 14/14 checks:
[`smol_vision_transfer_pilot_readiness.json`](../results/smol_vision_transfer_pilot_readiness.json).
It stores the compiled plan fingerprint, not only the sweep YAML hash. Confirmatory submission and
end-to-end readiness therefore reject a readiness artifact produced before a training-code or
compiled-DAG change. The artifact authorizes pilot submission only. Target-CUDA feasibility,
training execution, quality, and promotion remain unauthorized until their respective stages
produce sealed evidence.

The dual cell alone runs the target-GPU feasibility gate and the selective Smol acquisition. Both
cells retain tracked pretraining, SFT, RLVR, baseline evaluation, and final evaluation stages in
`sbdc/docvlm-ablation`. To launch from Colab with compact console output and full file logging:

```bash
python scripts/run_transfer_pilot_colab.py --pilot smol-vision
```

The launcher re-runs readiness, checks W&B credentials, free disk, CUDA memory, and native BF16,
then delegates to the resumable sweep runner. T4 is rejected because the experiment contract
requires native BF16; L4, A10, A100, or newer hardware is appropriate.

The ready-to-run notebook is
[`smol_vision_transfer_pilot.ipynb`](../../notebooks/smol_vision_transfer_pilot.ipynb). It checks
CUDA, native BF16, at least 14 GiB GPU memory, and 25 GiB free disk before installing anything. It
then checks out the pinned experiment branch, installs the production extras, refreshes the public
UDD feasibility artifact, authenticates W&B, performs the compact dry run, launches the resumable
matched pilot, and prints only bounded status and comparison fields. The launcher independently
revalidates the committed UDD fingerprint, revision, task quotas, and language reservations before
allowing a live run.

## Observed execution state

[`audit_smol_vision_transfer_pilot_execution.py`](../../scripts/audit_smol_vision_transfer_pilot_execution.py)
keeps launch readiness separate from execution. It accepts completion only when the local sweep
summary contains both expected arms and each has a sealed passing execution attestation. A W&B run
name alone is external activity, not proof that all training stages completed.

The authenticated W&B workspace was checked on July 25, 2026 and contains ten legacy LFM ablation
runs, no Smol pilot runs, and no local Smol sweep summary. The current bounded observation is
therefore `not_started_in_observed_state`:
[`smol_vision_transfer_pilot_execution_state.json`](../results/smol_vision_transfer_pilot_execution_state.json).
Because the project is private and the inventory is a point-in-time observation, this does not
prove that no newer run exists after capture. Training execution, quality, and promotion remain
unattested.

The execution audit accepts both execution-only seals and stronger deployment-capability seals,
but only when claim scope and quality authorization agree and the attestation digest is valid. It
also fingerprints the compact arm-to-attestation mapping so the confirmatory submission gate can
prove that its pilot comparison references the same completed runs.

External activity now uses the metric-free
[`docvlm_ablation_wandb_run_inventory.json`](../results/docvlm_ablation_wandb_run_inventory.json)
rather than the legacy ablation metric snapshot. It stores only run identity, state, timestamps,
when available, state counts, and a content fingerprint. Configs, histories, summaries, artifacts,
and metric tables remain excluded. Refresh it after `wandb.login()` with:

```bash
python scripts/snapshot_wandb_run_inventory.py
```

Both Smol Colab notebooks refresh this inventory before rebuilding execution evidence.

## Cross-runtime evidence handoff

A completed pilot can outlive its Colab runtime without uploading checkpoints or long metric
histories. [`publish_smol_pilot_handoff.py`](../../scripts/publish_smol_pilot_handoff.py) validates
the exact two-run topology, completed summary, schema-6 comparison, current sweep fingerprint, and
matching sealed attestation hashes. It then publishes only the summary, comparison, and a
content-addressed manifest as the W&B Artifact
`smol-vision-transfer-pilot-handoff`.

The confirmatory notebook restores that Artifact when the local pilot summary is absent.
[`restore_smol_pilot_handoff.py`](../../scripts/restore_smol_pilot_handoff.py) rejects a changed
config, invalid manifest fingerprint, modified file, unsealed run, mismatched attestation, or
different existing local evidence. The restored evidence authorizes submission only; it does not
authorize quality or promotion.
