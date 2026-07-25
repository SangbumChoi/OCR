# Public student-data acquisition

The native student experiment can acquire a public UDD component directly from Hugging Face before
weighted mixing. [`scripts/acquire_student_data.py`](../../scripts/acquire_student_data.py) turns an
immutable Hub revision into a validated local Arrow artifact with complete provenance.

## Immutable source

The full experiment pins `danelcsb/UDD` to commit
`f5eb52104627d20ddd1eab2130ad78f87cb0d7c9`. Branch names such as `main` are rejected because the
same experiment YAML must not silently receive different rows later.

Hub metadata at that commit reports 39,837 image-rows, 77,063 QAs, five Parquet shards, and
2,305,254,751 dataset bytes. Its 21-column schema matches the acquisition contract.

[`audit_public_udd_readiness.py`](../../scripts/audit_public_udd_readiness.py) verifies this
snapshot without downloading the image payload. It binds the experiment's repo, immutable
revision, split, fold, and 0.55 weight to the dataset card and Dataset Viewer schema, all five
Parquet LFS SHA-256 values, seven task counts, 32-source inventory, and multilingual distribution.
The compact result is
[`public_udd_training_readiness.json`](../results/public_udd_training_readiness.json), which is a
required input to the end-to-end readiness audit. It authorizes this training component only and
cannot authorize model quality.

```bash
python scripts/acquire_student_data.py \
  --repo-id danelcsb/UDD \
  --revision f5eb52104627d20ddd1eab2130ad78f87cb0d7c9 \
  --split train \
  --fold train \
  --decode-checks 32 \
  --output artifacts/data/components/public_udd
```

Private datasets use `HF_TOKEN` from the environment. Tokens are never written to manifests.

## Selection

The component can independently filter repeated `--source`, `--task`, and `--language` values.
By default, `--max-rows` applies a deterministic SHA-256 rank over seed, sample ID, and source row
index rather than taking a potentially source-ordered prefix. Capped pilots use
`--sampling-strategy task_stratified --min-rows-per-task 16`: every observed task first receives
up to 16 rows, and the remaining budget is apportioned by residual task capacity. Rows inside each
task are still selected by the same deterministic hash rank. Acquisition fails when the cap cannot
satisfy the requested floors. Leaving `--max-rows` unset retains every matching row.

The full executable mixture is:

| Component | Weight | Selection |
| --- | ---: | --- |
| authored hard documents | 0.45 | four generated hard families, difficulty 5 |
| public UDD | 0.55 | pinned Hub commit, deterministic `fold=train` |

Weights remain adjustable in
[`configs/sub1b_experiment.yaml`](../../configs/sub1b_experiment.yaml). Additional local or pinned
Hub components can be added without changing code.

## Validation gates

Before `save_to_disk`, acquisition requires:

- the canonical image, QA, source, task, language, metric, and fold columns;
- aligned non-empty `instructions` and `answers`;
- conforming `elements_json` payload and bounding-box DTOs;
- unique sample IDs and unique non-empty `(phash, width, height)` identities;
- no row outside the requested fold or filters;
- deterministic image decode checks with stored width and height agreement;
- a Hub-resolved commit exactly equal to the requested immutable revision.

`component_manifest.json` records the complete selection spec, resolved commit, source and selected
row counts, QA count, task/source/language/license distributions, decoded-image count, Arrow
fingerprint, and selected-index fingerprint. For capped inputs it also records eligible task
counts, deterministic quotas, realized task counts, and whether the requested floor was satisfied.
`mixture_manifest.json` carries a fingerprint of this upstream manifest, preserving the chain into
tokenizer training and pretraining.

## Experiment integration

A component uses either `path` or `hub`, never both:

```yaml
data:
  components:
    - name: public_udd
      weight: 0.55
      hub:
        repo_id: danelcsb/UDD
        revision: f5eb52104627d20ddd1eab2130ad78f87cb0d7c9
        split: train
        fold: train
        sources: []
        tasks: []
        languages: []
        max_rows: null
        seed: 7
        decode_checks: 32
```

The compiled DAG declares `mix_pretraining_data` dependent on every Hub acquisition stage. A
download, schema, split, duplicate, or image failure therefore stops the experiment before public
rows can reach the tokenizer or model.
