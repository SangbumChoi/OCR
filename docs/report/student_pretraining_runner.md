# Native student pretraining runner

## Scope

[`scripts/pretrain_student.py`](../../scripts/pretrain_student.py) trains the native sub-1B
student directly from UDD. It supports random or selectively initialized students, an optional
native online teacher, token- or student-FLOP learning-rate scheduling, deterministic curriculum scheduling,
mixed precision, `torchrun` data parallelism, held-out evaluation, and exact checkpoint resume.

This is training infrastructure, not evidence that the default 20B-effective-token run has been
completed.
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

## Activation checkpointing

`training.activation_checkpointing` is one shared contract for pretraining, SFT, RLVR, and the
full-student feasibility benchmark. The production blueprint enables non-reentrant block
checkpointing for `vision`, `connector`, and `language`. Forward values remain unchanged, while
backward recomputes each selected block instead of retaining its internal activations. Evaluation,
teacher inference, and autoregressive generation do not checkpoint because autograd is disabled.

Components are independently selectable for memory/throughput ablations. `use_reentrant: false`
is the production setting because packed FlexAttention uses captured execution plans and selective
distillation can retain intermediate features. Every training checkpoint records the resolved
contract, and exact resume rejects a change rather than continuing with a different memory and
compute regime.

## Executable token budget

The full blueprint sets `epochs: null`, `stop_at_total_tokens: true`, and
`token_unit: effective`. The runner therefore repeats deterministic sampler epochs until it reaches
20B effective tokens instead of stopping after an arbitrary number of corpus passes. One effective
token is either a non-padding text token or one of the connector's 64 resampled visual-prefix
tokens. The accounting deliberately does not call every raw image patch a decoder token.

Every optimizer update records six cumulative counters:

- `train/tokens_seen`: supervised answer tokens;
- `train/text_tokens_seen`: all non-padding prompt and answer tokens;
- `train/effective_tokens_seen`: text plus resampled visual-prefix tokens.
- `train/student_flops_seen`: analytical dense student training FLOPs for the actual padded
  microbatch shapes.
- `train/checkpoint_recompute_flops_seen`: estimated block-forward FLOPs re-executed by activation
  checkpointing.
- `train/executed_student_flops_seen`: the sum of algorithmic student FLOPs and checkpoint
  recomputation.

`train/budget_tokens_seen` selects one of those counters through `token_unit` and drives both the
cosine schedule and the hard stopping condition. Since an optimizer update is atomic, the final
count may exceed `total_tokens` by at most one global update. `--max-steps` remains an explicit
smoke/debug ceiling and may stop before the token target. A finite `epochs` value is also supported,
but a production token-budget run fails rather than silently succeeding if that epoch ceiling is
exhausted first.

For architecture comparisons, `schedule_unit: student_flops`,
`stop_at_student_flops: true`, and `total_student_flops` move the cosine schedule and hard stop to
the compute counter. `training_compute_fraction` does the same for curriculum boundaries. These
fields are part of the checkpoint resume contract, so a resumed run cannot silently change its
compute estimand or budget. The complete fixed-compute design is documented in
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md).

`student_flops_seen` deliberately remains the scientific, implementation-independent estimand used
for compute matching. Changing checkpoint placement therefore does not silently grant an arm more
model optimization. The two additional counters expose the runtime compute cost of that memory
choice, and all three counters resume exactly from checkpoint state.

## Fail-closed supervision

The default DAG uses cross-tokenizer LFM outputs as quality-gated offline sequence targets.
`teacher_kl` and `hidden_feature_distillation` remain zero unless
`pretraining.teacher_checkpoint` supplies a same-tokenizer native teacher. The runner resolves
every curriculum stage before optimization and rejects active online-teacher losses without that
checkpoint, a teacher checkpoint with no active online loss, or any stage with no active loss.
Teacher inference is skipped in stages where its losses are both zero.

Checkpoint metadata records stage-level active losses, online-teacher status, selected
gold/offline-teacher target counts, and the box IoU-family objective. Exact resume requires the same
supervision contract. The paired leave-one-loss-out design is
[`student_pretraining_loss_sweep.md`](student_pretraining_loss_sweep.md); the matched softmax and
SigLIP comparison is
[`student_contrastive_objective_sweep.md`](student_contrastive_objective_sweep.md), while the GIoU,
DIoU, and CIoU comparison is
[`student_box_iou_loss_sweep.md`](student_box_iou_loss_sweep.md).

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
  --max-requests 4096 \
  --selection-seed 7 \
  --output artifacts/data/teacher_requests
python scripts/build_teacher_targets.py generate \
  --requests artifacts/data/teacher_requests/requests.jsonl \
  --model lfm2_5-vl-1.6b \
  --model-revision 919fde3d022e3f90a4716006f993938ee8c2eb97 \
  --device cuda \
  --output artifacts/data/teacher_predictions.jsonl
python scripts/build_teacher_targets.py apply \
  --src artifacts/data/mixture \
  --requests artifacts/data/teacher_requests/requests.jsonl \
  --predictions artifacts/data/teacher_predictions.jsonl \
  --min-score 0.8 \
  --min-acceptance-rate 0.1 \
  --accepted-target-count 400 \
  --selection-seed 7 \
  --expected-model lfm2_5-vl-1.6b \
  --expected-revision 919fde3d022e3f90a4716006f993938ee8c2eb97 \
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
a gold-only run. It bounds generation to 4,096 deterministic requests and retains exactly 400
eligible targets. Model and processor use one pinned Hub revision, which is checked again when
targets are applied. The paired LFM/Qwen design is
[`student_sequence_teacher_sweep.md`](student_sequence_teacher_sweep.md).

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

## Executable curriculum

`training.pretraining.curriculum` supports `optimizer_step_fraction` for bounded runs and
`training_token_fraction` for the production token-budget run. Every stage has a unique ID, an
increasing `until_fraction`, and optional partial overrides:

```yaml
curriculum:
  unit: training_token_fraction
  stages:
    - id: recognition_bootstrap
      until_fraction: 0.2
      group_weights:
        recognition: 3.0
        localization: 1.0
      loss_weights:
        region_text_contrastive: 0.25
        box_regression: 0.30
    - id: mixed_reasoning
      until_fraction: 1.0
      group_weights: {}
      loss_weights: {}
```

The base `input_pipeline.group_weights` and `losses` remain in force for keys a stage does not
override. Sampler group names must exist under the active `balance_by` dimension; unknown groups,
negative weights, a stage that disables every available group, duplicate IDs, non-increasing
boundaries, and a final boundary other than `1.0` fail closed.

In token mode, loss-stage selection uses the exact global budget counter before each update.
Sampler group weights stay fixed because prefetched batches cannot observe an exact token boundary;
a token-fraction curriculum that tries to override them is rejected. Step mode retains the
zero-based optimizer update contract, including gradient accumulation, epoch-end partial windows,
and `max_steps`. Each training record includes `train/curriculum_stage`,
`train/curriculum_progress`, and `train/loss_weight/<name>` values for audit and W&B ingestion.

## Gradient-conflict diagnostics

`training.pretraining.gradient_conflict_probe` optionally measures pairwise weighted-loss gradient
cosines on final vision, connector, and language normalization anchors. The diagnostic forward
restores every RNG stream, uses `torch.autograd.grad` without writing optimizer gradients, and
records its extra forward and autograd traversals explicitly. Its configuration is included in the
resume supervision contract.

This is an anchor proxy, not a full-gradient claim. The three-arm by three-replicate design,
trajectory-hash invariant, aggregation command, and PCGrad/GradNorm promotion gate are specified in
[`student_gradient_conflict_audit.md`](student_gradient_conflict_audit.md).

## Validation-adaptive mixture

`training.pretraining.input_pipeline.adaptive_mixture` optionally uses periodic group-specific
validation losses to change the next sampler epoch. The sampler and `--eval-group-by` dimensions
must match exactly, periodic evaluation must be enabled, and curriculum stages may not also
override group weights. Loss-only curricula remain compatible.

The explicit `--eval-src` path is an optimizer-heldout validation dataset, not the final test
artifact. Evaluation only updates a checkpointed EMA and pending flag. The probability vector
changes at the next epoch boundary, preserving deterministic prefetching and exact mid-epoch
resume. See
[`student_adaptive_mixture.md`](student_adaptive_mixture.md) for the update rule, split contract,
metrics, and the three-arm paired sweep.

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
- epoch, batch cursor, optimizer step, and supervised/text/effective token counts;
- cumulative actual-shape student FLOPs, dense visual tokens, valid visual tokens, and visual
  sample count;
- tokenizer, curriculum, token-budget, and gradient-checkpointing contracts plus a
  `latest_checkpoint.txt` pointer;
- the gradient-conflict probe schedule and anchor selection when configured;
- adaptive mixture probabilities, EMA losses, pending update, and counters when enabled.

Rotation is a stable hash of tokenizer-independent sample ID, epoch, and augmentation seed.
Combined with the deterministic balanced sampler and `persistent_workers=False`, an interrupted run
reconstructs the same next batch and augmentation. A tokenizer mismatch is rejected before state
loading, as is a changed curriculum, token unit, visual-prefix count, or training horizon. Exact
continuation also rejects a changed activation-checkpointing selection and requires the original
`torchrun` world size because every rank has its own saved RNG stream and sampler slice.

## Loss and metric boundary

The runner currently optimizes and logs separate scalars for:

- answer-only autoregressive generation;
- native-teacher KL;
- selected hidden-feature distillation;
- multi-positive region/text contrast;
- normalized box regression with selectable GIoU, DIoU, or CIoU;
- four-way orientation classification.

Reading-order questions are generative examples and therefore contribute to autoregressive loss;
there is no separate reading-order scalar. Metrics are appended to `metrics.jsonl`, and checkpoints
are written under `outputs/.../checkpoints/step-NNNNNNNN`. Downstream capability claims must use
generation metrics on template-, source-, and image-identity-held-out data rather than treating
training loss as document-understanding accuracy.

Input-efficiency records additionally include `train/dense_visual_tokens_per_sample`,
`train/executed_visual_tokens_per_sample`, and `train/valid_visual_token_fraction`. These are
cumulative utilization diagnostics, not quality metrics.
[`student_visual_canvas_sweep.md`](student_visual_canvas_sweep.md) pairs them with heldout
generation scores and deployment gates. `train/visual_attention_backend` records whether a step
resolved to dense, portable loop, or compiled FlexAttention execution.

Continue with structured SFT and optional verifiable-reward GRPO using
[`student_posttraining_runner.md`](student_posttraining_runner.md).
