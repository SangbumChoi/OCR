# Native student post-training runner

## Scope

[`scripts/posttrain_student.py`](../../scripts/posttrain_student.py) provides the executable second
stage for the native sub-1B document VLM. It supports:

- exhaustive structured SFT with answer-only, free-rationale, and evidence-linked targets;
- strict answer/evidence/rationale JSON generation;
- single-update group-relative policy optimization from an SFT checkpoint;
- independently reported verifiable reward components;
- atomic checkpoints and exact continuation with tokenizer and reference guards.

This is training infrastructure, not evidence that the default SFT or RLVR recipe improves held-out
capability. Claims still require matched SFT-only and RLVR runs on template-, graph-, and
source-held-out evaluations.

## Response contract

Every target and sampled completion is exactly one JSON object:

```json
{"answer":"42","evidence":[[0.12,0.31,0.44,0.39]],"rationale":"The total cell states 42."}
```

`answer` is a non-empty string. `evidence` contains at most 32 ordered `xyxy` boxes normalized to
the original image coordinate frame, or is empty. `rationale` is a concise string or is empty.
Unknown fields, surrounding prose, malformed JSON, pixel-coordinate boxes, and reversed boxes fail
the structural gate. The default malformed reward is zero.

The three SFT ablation modes keep the schema fixed:

| Mode | Answer | Evidence | Rationale |
| --- | --- | --- | --- |
| `answer_only` | gold | empty | empty |
| `free_rationale` | gold | empty | authored rationale when present |
| `evidence_linked` | gold | authored normalized boxes when present | authored rationale when present |

SFT disables rotation because output evidence uses original-image normalized coordinates. This is
separate from the temporary pretraining box head, whose targets are transformed into the padded
training canvas.

## Structured SFT

Start from a native pretraining checkpoint:

```bash
python scripts/posttrain_student.py sft \
  --samples data/posttraining/train.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_pretrain/I4_selective/checkpoints/step-00010000/student \
  --output outputs/student_sft/evidence_linked
```

The same runner accepts generated realistic cases:

```bash
python scripts/posttrain_student.py sft \
  --realistic-root data/probes/realistic_cases \
  --variant degraded \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_pretrain/I4_selective/checkpoints/step-00010000/student \
  --target-mode evidence_linked \
  --output outputs/student_sft/evidence_linked
```

The SFT sampler shuffles deterministically and exhausts every example once per epoch. A
single-process final batch may be smaller. Distributed runs pad only enough examples to keep all
ranks on equal batch counts. The inherited pretraining auxiliary and distillation losses are
explicitly zeroed, so SFT optimizes only supervised autoregressive tokens.

Use `torchrun` for SFT data parallelism and `--resume latest` for exact continuation. The checkpoint
records `run_stage: sft:<target_mode>`; SFT resume rejects a different target mode, and RLVR rejects
checkpoints without an SFT marker.

## Verifiable-reward GRPO

Run RLVR from an SFT checkpoint:

```bash
python scripts/posttrain_student.py rlvr \
  --samples data/posttraining/rlvr.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_rlvr/full_reward
```

For each selected prompt, the policy samples one group with top-p sampling. Rewards are normalized
within that group. The runner performs one on-policy update, applies a frozen-reference KL penalty,
then discards the group. It does not implement PPO clipping or multiple optimization epochs over a
replay buffer, so results must be described as single-update GRPO.

The visual tower and connector run once per image during autoregressive rollout; the fixed visual
prefix is reused across every group member and generated token. Policy and frozen-reference
log-probabilities are each computed in one teacher-forced pass over the completed sequences.

## Reward contract

The active reward weights are renormalized per task, so unavailable annotations do not silently
become zeros:

| Component | Applies when | Current verifier |
| --- | --- | --- |
| `answer_correctness` | all samples | normalized exact match |
| `normalized_text_similarity` | all samples | repository semantic text matcher |
| `box_iou` | authored evidence exists | bidirectional best-IoU F1 with box-count agreement |
| `table_tree_similarity` | table/TEDS sample | TEDS |
| `chart_numeric_tolerance` | numeric/chart sample | tolerance-aware numeric accuracy |
| `formula_equivalence` | formula/LaTeX sample | deterministic LaTeX normalization equality |
| `grounded_rationale_consistency` | evidence and authored rationale exist | non-empty rationale gated by evidence IoU |
| `calibrated_abstention` | all samples | abstain iff the sample requires abstention |

The formula verifier is currently a normalization proxy, not symbolic algebra. The rationale
verifier proves cited-region overlap and rationale presence, not semantic entailment. Keep their
metrics separate and add stronger offline verifiers before making symbolic-equivalence or
faithful-reasoning claims.

`metrics.jsonl` reports total reward, reward variance, policy loss, reference KL, gradient norm,
structural-validity fraction, and every applicable reward component independently. A group with no
reward variance receives zero policy advantage; at the initial SFT reference it therefore produces
no update.

## Resume and operational boundary

RLVR checkpoints contain policy weights, optimizer and AMP scaler state, Python/Torch/CUDA RNG
state, rollout and optimizer cursors, tokenizer fingerprint, and a frozen-reference identifier.
Resume rejects a changed tokenizer or reference checkpoint.

Native RLVR currently runs in one process. An 800M policy, frozen 800M reference, gradients, and
AdamW state must fit on that process; shard independent experiments by seed when one device is not
large enough. Distributed rollout, optimizer sharding, KV caching, RL replay mixing, held-out
evaluation, and a symbolic formula engine remain future measured extensions. The current KL term is
the implemented collapse constraint; general multimodal replay is not yet interleaved inside the
RLVR loop.
