# End-to-end student experiment runner

[`scripts/run_student_experiment.py`](../../scripts/run_student_experiment.py) compiles one validated
YAML experiment into a resumable stage DAG. It connects hard-document synthesis, semantic split
validation, UDD conversion, weighted data mixing, quality-gated cross-tokenizer distillation,
tokenizer training, student initialization, pretraining, grounded SFT, RLVR, and train/heldout
generation evaluation.

Random initialization is reproducible: `initialization.seed` is validated, passed to model
construction before parameter allocation, included in the stage signature, and written into the
initial checkpoint metadata.

External generation inputs are content-addressed. The plan records the byte count and SHA-256 of
`synthetic.config`; changing that YAML invalidates the experiment fingerprint and every dependent
stage instead of incorrectly resuming old documents. Configured sequence-teacher prediction files
and initialization token maps receive the same treatment. A combined SHA-256 over the
`docvlm_eval` Python source tree and every compiled script entrypoint also invalidates resume after
generator, model, loss, reward, or runner implementation changes.

For matched multi-run ablations, use
[`student_sweep_runner.md`](student_sweep_runner.md). It compiles RFC 6902 experiment/blueprint
patches, rejects changes to declared fixed controls, reuses this runner for every variant, and
aggregates baseline deltas from the final train/heldout comparisons.

## Configurations

The full approximately 800M experiment is
[`configs/sub1b_experiment.yaml`](../../configs/sub1b_experiment.yaml). Inspect its commands and
dependencies without creating files:

```bash
python scripts/run_student_experiment.py --dry-run
```

The CPU contract test uses the same 16 stages with a dummy cross-tokenizer teacher, one
587k-parameter student, and one optimizer step per training phase:

```bash
python scripts/run_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml
```

The experiment YAML controls synthetic families, difficulty, independent split seeds, degradation,
data components and weights, tokenizer size, initialization arm and transfer sources, training
limits, evaluation settings, and W&B metadata. The runner writes a resolved architecture blueprint
whose `data_mix`, sampler groups, and tokenizer/model dimensions match the experiment.

Public components may use a local `path` or a pinned Hugging Face `hub` specification. The full
configuration acquires the public UDD train fold at an immutable commit, validates its schema,
payload, duplicate identities, and image dimensions, then mixes it at 55% with 45% authored hard
documents. See
[`student_data_acquisition.md`](student_data_acquisition.md).

The full configuration uses `lfm2_5-vl-1.6b` as an offline sequence teacher. It exports
fingerprinted image-question requests, resumes deterministic teacher generation, applies each
sample's native metric as a quality gate, and preserves rejected targets only as aggregate
provenance. The tiny configuration uses `dummy-echo`; its deliberately incorrect outputs prove
that all teacher responses can be rejected while the pipeline safely trains on gold.

## Data mixture

Each `data.components` entry names an on-disk or pinned-Hub UDD dataset, sampling weight, and
optional source fold. `path: "@synthetic"` refers to the UDD produced by the experiment. The mixer
normalizes every component to the canonical UDD superset, marks selected rows as training data, and
records their component identity without duplicating rows.

`mixture_manifest.json` records row counts, fingerprints, paths, and normalized weights. The
balanced batch sampler applies those weights at runtime with `balance_by: component`. This keeps
the physical corpus stable while allowing mixture probabilities to change explicitly.

## Resume and provenance

Every stage has a command signature, dependencies, and required artifacts. Successful stages are
skipped only when their state signature still matches and every artifact remains valid. Interrupted
pretraining, SFT, and RLVR stages automatically pass `--resume latest` only when the interrupted
state has the same signature and a checkpoint pointer exists. A changed upstream checkpoint starts
the dependent stage fresh.

When a signature changes, the runner removes only that stage's declared outputs inside the
experiment root before rebuilding them. It applies the same cleanup to interrupted
non-checkpoint stages and completed stages with invalid artifacts. Checkpoints remain intact for
an interrupted training stage with the same signature, preserving exact resume.

Each run root contains:

- `resolved_blueprint.yaml`, `experiment_spec.json`, and `experiment_plan.json`;
- `state/stages/<stage>.json` with status, command, timing, return code, and signature;
- `logs/<stage>.log` with combined process output;
- `artifacts/` with immutable stage handoffs and final split comparison;
- `run_summary.json` with completed or skipped outcomes.

Run a bounded section only after its external dependencies have completed:

```bash
python scripts/run_student_experiment.py \
  --from-stage pretrain \
  --to-stage evaluate
```

The runner fails closed when a command exits successfully but its declared artifacts are absent.
Train and heldout synthesis must use different seeds, and split validation runs before either split
can enter training or evaluation.
