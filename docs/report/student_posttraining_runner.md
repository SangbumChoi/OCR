# Native student post-training runner

## Scope

[`scripts/posttrain_student.py`](../../scripts/posttrain_student.py) provides the executable second
stage for the native sub-1B document VLM. It supports:

- exhaustive structured SFT with answer-only, free-rationale, and evidence-linked targets;
- strict answer/evidence/rationale JSON generation;
- single-update group-relative policy optimization from an SFT checkpoint;
- periodic supervised multimodal replay during RLVR;
- independently reported verifiable reward components;
- atomic checkpoints and exact continuation with tokenizer and reference guards.

This is training infrastructure, not evidence that the default SFT or RLVR recipe improves held-out
capability. Claims still require matched SFT-only and RLVR runs on template-, graph-, and
source-held-out evaluations.

Those comparisons are compiled by
[`student_posttraining_sweeps.md`](student_posttraining_sweeps.md). The SFT-target suite evaluates
all three target modes before RLVR. The reward suite fixes evidence-linked SFT and compares its
checkpoint directly with correctness-only and full-reward RLVR.

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

Every 20 rollout updates by default, the optimizer also receives an evidence-linked supervised
cross-entropy anchor with coefficient 0.10. This preserves structured answering and supplies a
learning signal when every completion in a reward group receives the same score. Without
`--replay-samples`, the anchor is sampled from the active RLVR dataset. Pass a separate
general-multimodal JSONL when broader capability retention is required:

```bash
python scripts/posttrain_student.py rlvr \
  --samples data/posttraining/rlvr.jsonl \
  --replay-samples data/posttraining/general_multimodal_replay.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_rlvr/replay_anchored
```

`supervised_replay.every_steps` and `loss_coefficient` are explicit blueprint controls. Set both
to zero for the no-replay ablation. A checkpoint records this pair and rejects resume under a
different replay contract.

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
| `formula_equivalence` | formula/LaTeX sample | bounded symbolic equivalence with exact-normalized fast path |
| `grounded_rationale_consistency` | evidence and authored rationale exist | non-empty rationale gated by evidence IoU |
| `calibrated_abstention` | all samples | abstain iff the sample requires abstention |

Formula verification first applies deterministic LaTeX normalization, then parses elementary
algebra, trigonometry, and equations with
[SymPy's strict ANTLR LaTeX parser](https://docs.sympy.org/latest/modules/parsing.html#sympy.parsing.latex.parse_latex).
It accepts expansions, factorizations, constant-scaled equations, and standard identities. Inputs
are capped by character, command, symbol, operation, and expression-tree limits. Unknown commands,
malformed LaTeX, integrals, sums, products, derivatives, limits, and parser failures receive zero
equivalence reward. This is intentionally narrower than a theorem prover. The rationale verifier
proves cited-region overlap and rationale presence, not semantic entailment; keep that metric
separate before making faithful-reasoning claims.

`metrics.jsonl` reports total reward, reward variance, policy loss, reference KL, total loss,
gradient norm, structural-validity fraction, replay application/loss/token count, the replay sample
ID, cumulative and per-step analytical student FLOPs, and every applicable reward component
independently. A group with no reward variance receives zero policy advantage; a scheduled replay
anchor can still provide a supervised update.

## Resume and operational boundary

RLVR checkpoints contain policy weights, optimizer and AMP scaler state, Python/Torch/CUDA RNG
state, rollout and optimizer cursors, student FLOPs consumed, tokenizer fingerprint, and a
frozen-reference identifier. Resume rejects a changed tokenizer, reference checkpoint, replay
contract, or compute-budget contract. Setting `max_steps: null`, `stop_at_student_flops: true`, and
`total_student_flops` makes the compute budget the production stop; this is used by
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md).

Native RLVR currently runs in one process. An 800M policy, frozen 800M reference, gradients, and
AdamW state must fit on that process; shard independent experiments by seed when one device is not
large enough. Distributed rollout, optimizer sharding, KV caching, multi-epoch off-policy replay,
and semantic rationale entailment remain future measured extensions. The implemented collapse
constraints are frozen-reference KL and periodic supervised replay.

## Held-out generation evaluation

Evaluate train and leakage-safe heldout JSONL with one loaded checkpoint:

```bash
python scripts/eval_student.py \
  --split train=data/posttraining/train.jsonl \
  --split heldout=data/posttraining/heldout.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_rlvr/full_reward/checkpoints/step-00001000/student \
  --output outputs/student_eval/full_reward
```

The evaluator never passes gold targets to `generate`. It decodes the strict JSON completion, sends
only its `answer` field through the sample's standard benchmark metric, and reports structured
reward separately. This distinction prevents a high reward from being presented as ANLS, TEDS, or
grounding accuracy.

Each split writes:

- `summary.json`: headline score, reward, structural validity, answer rate, latency,
  answer-type/source/language slices, and canonical robustness slices with coverage counts;
- `per_sample.jsonl`: raw structured output, parsed fields, standard score, reward components, and
  structural error plus canonical robustness labels;
- root `comparison.json`: train-minus-heldout headline, matched answer-type gaps, and matched
  robustness-slice gaps;
- root `manifest.json`: checkpoint, tokenizer, split, and decoding provenance.

Use `--max-samples N --seed S` for a deterministic smoke subset. The reported latency uses
visual-prefix reuse but not a decoder KV cache, so it is a correctness baseline rather than a final
serving benchmark.

### Paired W&B metrics

Add W&B logging without changing the evaluation:

```bash
python scripts/eval_student.py \
  --split train=data/posttraining/train.jsonl \
  --split heldout=data/posttraining/heldout.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_eval/sft \
  --wandb-project docvlm-ablation \
  --wandb-run native-sft-heldout
```

Both splits are logged in one call at the same checkpoint step. Every capability appears in both
orientations:

```text
eval/train_<axis>                 eval/heldout_<axis>
eval_by_axis/<axis>/train         eval_by_axis/<axis>/heldout
```

The actual split name is `heldout`, not `held`. Reward components use
`eval_reward/<component>/<split>`; source and language slices use `eval_by_source` and
`eval_by_language`. Robustness slices use `eval_by_slice/<axis>/<value>/<split>`, for example
`eval_by_slice/degradation/scan/train` and
`eval_by_slice/degradation/scan/heldout`. This makes one W&B panel per suffix possible without
duplicating or manually aligning run steps.
