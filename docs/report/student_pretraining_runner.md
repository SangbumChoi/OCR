# Native student pretraining runner

## Scope

[`scripts/pretrain_student.py`](../../scripts/pretrain_student.py) trains the native sub-1B
student directly from UDD. It supports random or selectively initialized students, an optional
native online teacher, token-count scheduling, mixed precision, `torchrun` data parallelism,
held-out evaluation, and exact checkpoint resume.

This is training infrastructure, not evidence that the default 20B-token run has been completed.
Published model claims still require the controlled initialization, teacher, data-scale, and
held-out ablations in the architecture blueprint.

## Single-device runs

Train the tokenizer once:

```bash
pip install -e ".[student]"
python scripts/train_student_tokenizer.py \
  --repo danelcsb/UDD \
  --output artifacts/student_tokenizer
```

Run the random-initialization baseline:

```bash
python scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --output outputs/student_pretrain/I0_random
```

Start from a selectively transferred student:

```bash
python scripts/build_sub1b_student.py \
  --init-arm I4_selective \
  --vision-source /path/to/vision/checkpoint \
  --vision-family siglip \
  --language-source /path/to/language/checkpoint \
  --language-family llama \
  --token-map /path/to/target_to_source_token_ids.json \
  --save artifacts/student_I4

python scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --student-checkpoint artifacts/student_I4 \
  --output outputs/student_pretrain/I4_selective
```

Use `--max-steps` for a smoke run before allocating the full token budget.

## Teacher contract

The online path accepts a frozen native `DocumentVLMStudent` checkpoint:

```bash
python scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --teacher-checkpoint /path/to/native_teacher/student \
  --output outputs/student_pretrain/distilled
```

Online distillation is deliberately strict:

- student and teacher must use the identical tokenizer artifact, not merely the same vocabulary
  size;
- the tokenizer SHA-256 fingerprint stored in the teacher checkpoint must match `--tokenizer`;
- teacher logits are reduced immediately to the configured top-k tokens plus one exact
  remaining-vocabulary mass bucket;
- selected vision and language depth anchors are retained only for the current step;
- trainable linear projections align incompatible hidden widths before cosine feature loss;
- text-only batches skip vision feature pairs without inventing visual inputs.

LFM2.5-VL and other cross-tokenizer teachers do not satisfy this contract. Their outputs enter as
offline sequence targets or pseudo-labels in UDD, where the student tokenizer encodes the resulting
text. They must not be connected to token-level KL by matching vocabulary size or token position.
The policy is controlled by
`training.pretraining.distillation.cross_tokenizer_policy: sequence_targets_only`.

The offline path is executable and fail-closed:

```bash
python scripts/build_teacher_targets.py export \
  --src artifacts/data/mixture \
  --output artifacts/data/teacher_requests
python scripts/build_teacher_targets.py generate \
  --requests artifacts/data/teacher_requests/requests.jsonl \
  --model lfm2_5-vl-1.6b \
  --device cuda \
  --output artifacts/data/teacher_predictions.jsonl
python scripts/build_teacher_targets.py apply \
  --src artifacts/data/mixture \
  --requests artifacts/data/teacher_requests/requests.jsonl \
  --predictions artifacts/data/teacher_predictions.jsonl \
  --min-score 0.8 \
  --output artifacts/data/distilled_mixture
```

Exported requests include immutable image, question, answer, metric, and source-dataset
fingerprints. Generation resumes by request ID. Apply rejects unknown, duplicate, mismatched,
degenerate, or below-threshold responses and never changes native gold. Accepted targets are stored
in aligned `teacher_answers`, `teacher_scores`, and `teacher_provenance_json` columns. The
`sequence_targets.probability`, `min_score`, and `seed` controls deterministically choose accepted
teacher text or gold for each QA; missing or rejected teacher output always falls back to gold.
Generation fingerprints prevent a resumed file from mixing teachers or decoding settings. The full
experiment also requires a minimum acceptance rate, so a broken teacher run cannot silently become
a gold-only run.

## Distributed training

Launch one process per GPU:

```bash
torchrun --standalone --nproc-per-node=4 scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --output outputs/student_pretrain/I0_random
```

The balanced sampler first draws one deterministic global batch and then assigns a disjoint local
slice to each rank. Held-out task/source/language groups are also sharded without duplication.
Evaluation losses are weighted by sample count and reduced across ranks. The learning-rate
schedule advances by the globally reduced count of supervised answer tokens, not by microbatch or
optimizer-step count.

## Exact resume

Resume the latest atomic checkpoint:

```bash
python scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --output outputs/student_pretrain/I0_random \
  --resume latest
```

Each checkpoint contains:

- student weights and architecture;
- trainable distillation projections;
- optimizer and AMP scaler state;
- Python, CPU Torch, and CUDA RNG state;
- epoch, batch cursor, optimizer step, and supervised-token count;
- tokenizer fingerprint and a `latest_checkpoint.txt` pointer.

Rotation is a stable hash of tokenizer-independent sample ID, epoch, and augmentation seed.
Combined with the deterministic balanced sampler and `persistent_workers=False`, an interrupted run
reconstructs the same next batch and augmentation. A tokenizer mismatch is rejected before state
loading. Exact continuation also requires the original `torchrun` world size because every rank has
its own saved RNG stream and sampler slice.

## Loss and metric boundary

The runner currently optimizes and logs separate scalars for:

- answer-only autoregressive generation;
- native-teacher KL;
- selected hidden-feature distillation;
- multi-positive region/text contrast;
- normalized box regression with generalized IoU;
- four-way orientation classification.

Reading-order questions are generative examples and therefore contribute to autoregressive loss;
there is no separate reading-order scalar. Metrics are appended to `metrics.jsonl`, and checkpoints
are written under `outputs/.../checkpoints/step-NNNNNNNN`. Downstream capability claims must use
generation metrics on template-, source-, and image-identity-held-out data rather than treating
training loss as document-understanding accuracy.

Continue with structured SFT and optional verifiable-reward GRPO using
[`student_posttraining_runner.md`](student_posttraining_runner.md).
