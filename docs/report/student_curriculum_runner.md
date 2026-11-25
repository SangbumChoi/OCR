# Attested multi-round student curriculum

[`run_student_curriculum.py`](../../scripts/run_student_curriculum.py) executes the
failure-driven synthesis loop as an exact sequence of student-training rounds. Round zero runs the
complete experiment, including tokenizer construction, initialization, and pretraining. Every
later round preserves the parent tokenizer and full final student checkpoint, generates the
validation-authorized failure batch, adds deterministic cumulative replay, and runs only SFT,
optional preference optimization, RLVR, evaluation, and next-batch planning.

## Run

Inspect the initial DAG without creating a run:

```bash
python scripts/run_student_curriculum.py \
  --experiment configs/sub1b_experiment_curriculum_tiny.yaml \
  --output-root outputs/curriculum-smoke \
  --rounds 2 \
  --dry-run
```

Execute the rounds:

```bash
python scripts/run_student_curriculum.py \
  --experiment configs/sub1b_experiment.yaml \
  --output-root outputs/student-curriculum \
  --rounds 3 \
  --replay-fraction 0.5 \
  --replay-seed-base 20000
```

The base experiment must enable validation-backed `synthetic.adaptation_policy`, provide an
independent validation count and seed, and start without `synthetic.training_policy_plan`.
`configs/sub1b_experiment_curriculum_tiny.yaml` is the CPU contract test.

## Continuation contract

Each completed round receives a full-hash `evidence_attestation.json`. Before compiling the child
DAG, the continuation resolver verifies:

- the parent experiment plan, resolved spec, cumulative run summary, and attestation fingerprint;
- successful parent SFT, optional preference, RLVR, evaluation, and next-batch planning stages;
- the exact final student configuration, checkpoint files, tokenizer vocabulary, and tokenizer
  fingerprint;
- a byte-identical tokenizer copy materialized inside every child root, so the following round
  depends only on its immediate parent's attested artifacts;
- an untampered validation-only, training-authorized synthesis plan whose current and matched
  baseline sources are the parent's exact validation `per_sample.jsonl` artifacts;
- an exact round increment and an explicit `reset_per_stage` optimizer policy.

The child begins with `attest_continuation`. No synthesis or optimizer stage can run until the same
contract is rechecked at execution time. The child omits data acquisition, teacher generation,
tokenizer training, student initialization, and pretraining. This preserves learned model state
without silently pretending to resume an optimizer schedule from a different stage. SFT,
preference optimization, and RLVR each initialize their own optimizer, as declared by the
continuation policy. When both optional post-training stages are enabled, preference consumes the
new SFT checkpoint, RLVR consumes the preference checkpoint as its policy start, and the same
round's SFT checkpoint remains RLVR's frozen reference.

The next-round spec also replaces the fresh-run
`evaluation.baseline_checkpoint_stage` with `inherited`. The child then evaluates the parent's
attested final checkpoint on the child's exact current train/validation/heldout files before
evaluating the updated checkpoint. This preserves matched heldout sample IDs and a valid
train-minus-heldout gap even after cumulative replay changes the active train set. The synthesis
planner additionally joins the inherited-baseline and updated validation rows by exact sample ID,
rejects any changed benchmark identity, and records residual failure, signed learning progress,
and combined allocation utility. Each continuation therefore measures incremental improvement
against the exact inherited model without pretending that an omitted initialization or pretraining
stage can be evaluated locally.

## Replay and provenance

Every new failure-driven sample is retained in
`artifacts/continuation/replay_memory.jsonl`. Round one seeds this memory from the base train set;
later rounds require the immediate parent's attested memory and append the new batch with stable
origin-round lineage. Missing origins, duplicate IDs, altered lineage, or a memory file absent from
the parent full-hash attestation fail before child synthesis.

The active training set remains bounded by `replay_fraction`. Samples are selected without
replacement in round-robin strata over all prior origin rounds, with deterministic within-stratum
shuffling. This prevents the immediately previous failure batch from erasing the base distribution
while keeping each post-training round at the configured size. Replay IDs are namespaced, active rows are
deterministically shuffled, and `train_with_replay.manifest.json` records source and output hashes,
requested and realized fractions, selected counts by origin, cumulative counts by origin, and the
parent round.

`curriculum_summary.json` records each round's experiment fingerprint, full attestation hash,
capability status, synthesis-plan fingerprint, stage count, and final root. A passing execution
contract proves that the declared round handoffs and optimizer steps occurred. It does not by
itself establish document-VLM quality: promotion still requires the independent heldout gates and
reports `quality_claim_authorized` only when those capability gates pass.
