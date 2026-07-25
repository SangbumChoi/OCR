# Native student post-training runner

## Scope

[`scripts/posttrain_student.py`](../../scripts/posttrain_student.py) provides the executable second
stage for the native sub-1B document VLM. It supports:

- exhaustive structured SFT with answer-only, free-rationale, and evidence-linked targets;
- strict answer/evidence/rationale JSON generation;
- verifier-ranked DPO or IPO from frozen SFT candidates;
- single-update group-relative policy optimization from SFT or a DPO/IPO policy warm start;
- periodic supervised multimodal replay during RLVR;
- independently reported verifiable reward components;
- atomic checkpoints and exact continuation with tokenizer, policy-start, and reference guards.

This is training infrastructure, not evidence that the default SFT, preference, or RLVR recipe improves
held-out capability. Claims still require matched runs on template-, graph-, and source-held-out
evaluations.

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
SFT inherits the production component-level activation-checkpointing contract used by pretraining.

Use `torchrun` for SFT data parallelism and `--resume latest` for exact continuation. The checkpoint
records `run_stage: sft:<target_mode>`; SFT resume rejects a different target mode, and RLVR rejects
checkpoints without an SFT marker.

All three post-training modes accept optional W&B project, entity, run, group, tags, and run-ID
arguments. SFT streams `train/*` against `train/global_step`; preference and RLVR use their own
rollout counters and stream decomposed `reward/*` and `reward_diagnostic/*` values alongside policy
loss, KL, replay, gradient, and compute telemetry. Each stage remains a separate run under one
experiment or sweep group.

## Verifier-ranked preference optimization

Run the configured DPO or IPO objective from an SFT checkpoint:

```bash
python scripts/posttrain_student.py preference \
  --samples data/posttraining/train.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_preference/verifier_ranked
```

The frozen SFT reference samples a candidate group. By default,
`gold_anchored_verifier_ranked` replaces one candidate with the exact token sequence that the
collator used for evidence-linked SFT, then the structured verifier selects the highest- and
lowest-reward responses. This supplies a valid chosen response even when a weak SFT model emits
only malformed candidates. The policy receives one direct preference update only when the reward
margin passes the configured threshold. A truncated or malformed collated anchor fails closed.
Set `preference_source: reference_verifier_ranked` for the model-candidates-only control.
Candidate ties are logged as skipped pairs and consume rollout FLOPs without an optimizer step.
Policy and reference pair scoring reuse one visual encoding across the chosen and rejected
sequences.

The checkpointed objective contract covers objective, preference source, reward margin, DPO beta,
IPO tau, sequence reduction, reward weights, and malformed-response reward. See
[`student_preference_method_sweep.md`](student_preference_method_sweep.md) for the equation,
compute-matched GRPO comparison, continuation guarantees, and interpretation boundary. The
loss-only DPO-versus-IPO design is
[`student_preference_objective_sweep.md`](student_preference_objective_sweep.md). The matched
candidate-source ablation is
[`student_preference_source_sweep.md`](student_preference_source_sweep.md).

## Verifiable-reward GRPO

Run RLVR from an SFT checkpoint:

```bash
python scripts/posttrain_student.py rlvr \
  --samples data/posttraining/rlvr.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_rlvr/full_reward
```

To test sequential preference optimization followed by RLVR, pass the preference checkpoint as the
trainable policy and the exact SFT checkpoint as the frozen reference:

```bash
python scripts/posttrain_student.py rlvr \
  --samples data/posttraining/rlvr.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_preference/dpo/checkpoints/step-00001000/student \
  --reference-checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_rlvr/dpo_warm_start
```

`--reference-checkpoint` is required when `--checkpoint` has a `preference:dpo` or
`preference:ipo` run stage. The preference checkpoint initializes only the trainable policy. The
frozen KL reference remains the SFT model so the regularizer measures drift from the supervised
anchor rather than from the immediately preceding preference update. The runner recomputes the SFT
checkpoint's content identity and requires it to equal the immutable reference identity recorded by
the preference checkpoint; a same-tokenizer but unrelated SFT checkpoint is rejected.

For each selected prompt, the policy samples one group with top-p sampling. The default
`group_standardized` estimator centers and standardizes rewards within that group. The executable
`leave_one_out` alternative subtracts the mean reward of the other completions without dividing
away verifier scale. The runner performs one on-policy update, applies a frozen-reference KL
penalty, then discards the group. It does not implement PPO clipping or multiple optimization
epochs over a replay buffer, so results must be described as single-update group-relative RL.
The paired estimator design is
[`student_rlvr_advantage_sweep.md`](student_rlvr_advantage_sweep.md).

The visual tower and connector run once per image during autoregressive rollout; the fixed visual
prefix is reused across every group member and generated token. The language decoder performs one
prompt prefill and then appends one-token queries to its generation cache. Attention layers retain
only the configured eight KV heads rather than expanding them to all 24 query heads; optional
short-convolution layers retain only their bounded recurrent state. Attention buffers are allocated
once to the configured completion horizon and updated in place, avoiding per-token concatenation
and cache copies. Set
`training.posttraining.rlvr.rollout.use_kv_cache: false` only for a full-prefix compute/latency
ablation. The resolved rollout contract is checkpointed, so resume cannot silently change sampling
or cache semantics. The same contract includes an exact suffix-cycle guard. It emits EOS only when
the trailing period repeats consecutively three times after the minimum completion length; it does
not prohibit recurring table tags or labels elsewhere in a structured answer. Policy and
frozen-reference log-probabilities are each computed in one
teacher-forced pass over the completed sequences.
The contract also resolves a bounded completion horizon from the public task label. Concise tasks
retain the 128-token base budget, while declared full-page, reading-order, table, chart, and
evidence-linked reasoning labels can use up to the 512-token hard cap. Every candidate in a group
receives the same horizon, and `preference/generation_token_budget`,
`rlvr/generation_token_budget`, and their `generation_budget_escalated` companions expose the
executed policy.
Activation checkpointing applies only to the trainable policy's gradient-bearing log-probability
and replay passes. No-grad rollout generation and the frozen reference remain uncheckpointed.
`rlvr/student_flops_seen` remains the algorithmic compute-matching counter, while
`rlvr/checkpoint_recompute_flops_seen` and `rlvr/executed_student_flops_seen` expose the additional
policy recomputation. Per-step forms are logged as `rlvr/step_checkpoint_recompute_flops` and
`rlvr/step_executed_student_flops`.

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
| `grounded_rationale_consistency` | evidence, authored rationale, and a valid program trace exist | evidence IoU multiplied by rationale semantic match and program-fact F1 |
| `calibrated_abstention` | all samples | abstain iff the sample requires abstention |

Formula verification first applies deterministic LaTeX normalization, then parses elementary
algebra, trigonometry, and equations with
[SymPy's strict ANTLR LaTeX parser](https://docs.sympy.org/latest/modules/parsing.html#sympy.parsing.latex.parse_latex).
It accepts expansions, factorizations, constant-scaled equations, and standard identities. Inputs
are capped by character, command, symbol, operation, and expression-tree limits. Unknown commands,
malformed LaTeX, integrals, sums, products, derivatives, limits, and parser failures receive zero
equivalence reward. This is intentionally narrower than a theorem prover.

The production `evidence_program_trace` rationale verifier consumes the exact operation, typed
node or edge inputs, parameters, recomputed result, formatted answer, and required numeric facts
authored by the latent document graph. It verifies the trace fingerprint and independently
re-executes the operation before training. A malformed, tampered, dangling, non-finite, or
result-inconsistent trace fails sample construction. For a valid trace, rationale fact recall
penalizes omitted operands or results, fact precision penalizes hallucinated numbers, and explicit
percentages match their fraction equivalents. The final grounded score is evidence IoU multiplied
by semantic rationale similarity and numeric-fact F1. Trace-free samples do not receive this
component under the strict verifier; other applicable rewards are still normalized normally.
`evidence_semantic` remains available only as the paired ablation control.

`reward_diagnostic/rationale_text_similarity`,
`reward_diagnostic/rationale_program_fact_score`, and
`reward_diagnostic/program_trace_consistency` expose each contribution independently. The program
trace proves arithmetic consistency with the authored latent graph, not that the generated prose
caused the final answer.

`metrics.jsonl` reports the estimator, total reward, reward variance, advantage scale, policy loss,
reference KL, total loss, gradient norm, structural-validity fraction, rationale semantic
similarity, program-fact F1, program-trace consistency, replay application/loss/token count, the
replay sample ID, cumulative and per-step analytical student FLOPs, and every applicable reward
component independently. A group with no reward variance receives zero policy advantage; a
scheduled replay anchor can still provide a supervised update.

## Resume and operational boundary

RLVR and preference checkpoints contain policy weights, optimizer and AMP scaler state,
Python/Torch/CUDA RNG state, stage cursors, student FLOPs consumed, tokenizer fingerprint, and a
frozen-reference identifier. RLVR additionally records the incoming policy stage and exact
config-and-weight content identities for both its policy start and frozen reference. Resume rejects
a changed tokenizer, policy start, reference checkpoint, rationale verifier, objective contract,
compute-budget contract, or activation-checkpointing contract. RLVR additionally guards the replay
contract. SFT, preference optimization, and RLVR all use the same
fail-closed optimizer factory as pretraining. Preference and RLVR checkpoints bind the requested
optimizer controls to the realized implementation and bitsandbytes version; SFT inherits the
pretraining checkpoint contract. Setting `max_steps: null`,
`stop_at_student_flops: true`, and
`total_student_flops` makes the compute budget the production stop; this is used by
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md).

Native preference optimization and RLVR currently run in one process. An 800M policy, frozen 800M
reference, gradients, and the configured optimizer state must fit on that process; shard
independent experiments by seed when one device is not
large enough. Distributed rollout, optimizer sharding, KV caching, multi-epoch off-policy replay,
and learned semantic rationale entailment remain future measured extensions. Exact arithmetic
trace verification, frozen-reference KL, and periodic supervised replay are implemented.

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
  maximum-token rate, degenerate-repetition rate, answer-type/source/language slices, and canonical
  robustness slices with coverage counts;
- `per_sample.jsonl`: raw structured output, parsed fields, standard score, reward components, and
  structural error plus generated-token, token-limit, repetition, and canonical robustness labels;
- root `comparison.json`: train-minus-heldout headline, matched answer-type gaps, and matched
  robustness-slice gaps;
- root `manifest.json`: checkpoint, tokenizer, split, and decoding provenance.

Use `--max-samples N --seed S` for a deterministic smoke subset. The evaluator uses both
visual-prefix reuse and the decoder KV cache by default. Pass `--no-kv-cache` to measure the
full-prefix ablation. Each summary records `generation_backend` so latency results cannot be
compared across hidden generation paths.
See [`student_generation_rendering_safeguards.md`](student_generation_rendering_safeguards.md) for
the shared rollout controls and the separate integrity policy for long HTML tables and full pages.

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
