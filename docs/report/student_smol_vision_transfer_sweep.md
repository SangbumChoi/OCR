# SmolVLM2 vision-transfer confirmatory sweep

## Purpose

The one-seed SmolVLM2 pilot screens whether exact vision transformer-block initialization is worth
a full experiment. The confirmatory sweep answers the promotion question with three paired seeds
after the pilot has completed with sealed execution evidence.

[`sub1b_smol_vision_transfer_sweep.yaml`](../../configs/sub1b_smol_vision_transfer_sweep.yaml)
holds LFM language initialization, 814,207,243-parameter geometry, data, optimization, and
evaluation fixed. Its only treatment is exact SmolVLM2 vision-block initialization:

- baseline: `lfm_language_only`;
- candidate: `lfm_smol_dual`;
- paired replicates: `seed_0`, `seed_1`, and `seed_2`;
- contrast: `lfm_smol_dual - lfm_language_only`.

This is the production-budget experiment. It does not inherit the shortened pilot training or
evaluation limits.

## Shared selective payload

Selective Hub checkpoints are stored in a content-addressed cache keyed by the immutable repository
revision and sorted tensor prefixes. Every Smol treatment replicate therefore references the same
hash-verified 340 MB transformer-block payload. A filesystem lock makes concurrent acquisition
safe, and each run receives its own manifest copy pointing to that shared immutable payload.

This changes storage and download work only. Each replicate still constructs and trains an
independent student with its own matched seed controls.

## Promotion contract

Promotion requires all three paired replicates, a positive heldout effect of at least `0.005`, and
Bonferroni-controlled one-sided paired bootstrap evidence at familywise alpha `0.05`. The candidate
must also pass parameter-budget, target-GPU feasibility, generalization, grounding, reasoning,
multilingual, reliability, and generation-stability gates.

The generation-stability gate explicitly covers exact suffix-cycle repetition and max-token
termination. Table, HTML, and full-page tasks retain the 512-token hard cap plus rendering checks
for all-page coverage, table-cell survival, and bounded canvas size. A longer context is therefore
not accepted as evidence when the output merely repeats tokens or produces an unreadable page.

## Submission gate

[`audit_smol_confirmatory_submission.py`](../../scripts/audit_smol_confirmatory_submission.py)
binds the confirmatory budget to the exact pilot and confirmatory config fingerprints. Submission
requires both pilot arms to have matching sealed attestations, every candidate gate to pass, and a
strictly positive heldout pilot effect with an `improved` conclusion. A stale comparison, changed
config, extra run, missing run, failed generation gate, or attestation mismatch fails closed.

The current observed state has no local pilot summary or comparison, so the compact submission
artifact is `pending` with 2 pass, 5 pending, and 0 fail:
[`smol_vision_confirmatory_submission.json`](../results/smol_vision_confirmatory_submission.json).
This is an execution dependency, not evidence against the method.

After the pilot completes, the ready-to-run notebook is
[`smol_vision_transfer_confirmatory.ipynb`](../../notebooks/smol_vision_transfer_confirmatory.ipynb).
When opened in a fresh runtime, it restores the content-addressed pilot evidence from W&B before
performing the submission audit. No pilot checkpoint is required for this independent confirmatory
run. Its launcher performs the submission audit again before any production-budget training begins:

```bash
python scripts/run_smol_confirmatory_colab.py
```

## Compact evidence

After a completed sweep has written `comparison.json`, build the two claim artifacts with:

```bash
python scripts/build_smol_confirmatory_evidence.py
```

The builder verifies the current sweep fingerprint, all six sealed run attestations, required gate
statuses, replicate count, selected candidate, and multiplicity method. It writes compact
fingerprints and decisions rather than duplicating per-sample outputs, metric tables, HTML, or
full-page render payloads. Those detailed artifacts remain in their run directories.

No quality or promotion evidence is generated before execution. Until the pilot and confirmatory
runs complete, this file documents an executable contract, not a model-quality result.
