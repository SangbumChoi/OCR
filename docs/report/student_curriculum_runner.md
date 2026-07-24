# Attested multi-round student curriculum

[`run_student_curriculum.py`](../../scripts/run_student_curriculum.py) executes the
failure-driven synthesis loop as an exact sequence of student-training rounds. Round zero runs the
complete experiment, including tokenizer construction, initialization, and pretraining. Every
later round preserves the parent tokenizer and full final student checkpoint, generates the
validation-authorized failure batch, adds deterministic parent replay, and runs only SFT, RLVR,
evaluation, and next-batch planning.

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
- successful parent SFT/RLVR, evaluation, and next-batch planning stages;
- the exact final student configuration, checkpoint files, tokenizer vocabulary, and tokenizer
  fingerprint;
- an untampered validation-only, training-authorized synthesis plan whose source is the parent's
  exact validation `per_sample.jsonl`;
- an exact round increment and an explicit `reset_per_stage` optimizer policy.

The child begins with `attest_continuation`. No synthesis or optimizer stage can run until the same
contract is rechecked at execution time. The child omits data acquisition, teacher generation,
tokenizer training, student initialization, and pretraining. This preserves learned model state
without silently pretending to resume an optimizer schedule from a different stage. SFT and RLVR
each initialize their own optimizer, as declared by the continuation policy.

## Replay and provenance

Every new failure-driven sample is retained. Parent samples are selected without replacement using
the configured replay seed; the requested fraction is capped by available parent data. Replay IDs
are namespaced, rows are deterministically shuffled, and
`artifacts/continuation/train_with_replay.manifest.json` records source and output hashes, requested
and realized fractions, counts, and the parent round.

`curriculum_summary.json` records each round's experiment fingerprint, full attestation hash,
capability status, synthesis-plan fingerprint, stage count, and final root. A passing execution
contract proves that the declared round handoffs and optimizer steps occurred. It does not by
itself establish document-VLM quality: promotion still requires the independent heldout gates and
reports `quality_claim_authorized` only when those capability gates pass.
