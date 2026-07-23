# Fixed-dose sequence-teacher sweep

## Question

The deployment student should not depend on one teacher family by assumption. The nine-run
[`configs/sub1b_sequence_teacher_sweep.yaml`](../../configs/sub1b_sequence_teacher_sweep.yaml)
compares:

| Arm | Offline sequence supervision | Pinned Hub revision |
| --- | --- | --- |
| `gold_only` | none | n/a |
| `lfm` | `LiquidAI/LFM2.5-VL-1.6B` | `919fde3d022e3f90a4716006f993938ee8c2eb97` |
| `qwen` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |

The revisions are immutable commits resolved from the model repositories on 2026-07-24. LFM is the
current experimental teacher; Qwen is a non-LFM architecture and tokenizer control. Both remain
teachers only. The native student architecture and deployment parameter count are unchanged.

This file specifies an executable experiment, not a measured result. Teacher gains remain unproven
until all paired GPU runs complete.

## Fixed request and target dose

A teacher with broader quality-gate coverage would otherwise contribute more targets than a weaker
teacher. That is useful for production utility but confounds whether accepted targets from one
teacher are more useful. This sweep therefore fixes two budgets per replicate:

- deterministically select the same 4,096 image-question requests before loading a teacher;
- require at least 10% of those requests to pass their native metric at score 0.8;
- deterministically retain exactly 400 eligible targets;
- choose teacher text with the same 0.5 per-QA probability during pretraining.

The acceptance-rate floor guarantees enough eligible targets for the fixed dose. Compilation rejects
a requested target count larger than that floor. Apply fails if the realized eligible count is
smaller, rather than silently training on fewer teacher targets.

Request selection hashes the source row, QA identity, gold answers, metric, and paired seed. Target
selection independently hashes the accepted request ID and paired seed. The request and target
manifests record eligible counts, selected counts, rates, seeds, and SHA-256 fingerprints.

## Checkpoint identity

Non-dummy sequence teachers require a 40-character Hub commit. The revision is passed to both
`AutoProcessor.from_pretrained` and `AutoModelForImageTextToText.from_pretrained`. Generation
fingerprints and every prediction record include model key, Hub ID, revision, dtype, decoding
budget, and temperature.

Apply verifies that every prediction has the model and revision declared by the experiment. A
resumed prediction file with a different identity or decoding configuration fails before data is
modified. External prediction files receive the same apply-time identity check.

## Tokenizer control

The student tokenizer is trained from native questions, gold answers, full text, table HTML, and
structured elements, with `tokenizer.include_teacher_targets: false`. Teacher answers are excluded
from tokenizer fitting but remain available as pretraining targets. This keeps the token-to-ID
contract identical in meaning across gold, LFM, and Qwen arms and prevents teacher-specific
vocabulary merges from masquerading as distillation gains. Byte fallback still encodes every
teacher response.

The saved tokenizer metadata records this corpus policy. The compiled tokenizer command and
experiment fingerprint also include it.

## Paired design

Three paired replicates change model initialization, authored train and heldout documents, public
data selection, request selection, target selection, augmentation, optimization, and evaluation
seeds together across arms. Within a replicate, all other data, architecture, pretraining, SFT,
RLVR, and evaluation controls are identical.

Inspect or run:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sequence_teacher_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_sequence_teacher_sweep.yaml
```

W&B uses group `docvlm-sequence-teacher-ablation` and tags `teacher-ablation`,
`fixed-teacher-dose`, `teacher:lfm`, or `teacher:qwen`.

## Interpretation

Each teacher arm is paired against `gold_only`. A positive heldout interval for both teacher arms
supports sequence distillation that is not unique to LFM. A gain for only one teacher is
teacher-specific evidence, not a general distillation claim. Directly compare the paired
teacher-minus-gold deltas rather than raw means.

The fixed-dose design estimates the utility of selected, quality-gated targets. It deliberately
does not estimate end-to-end production utility, where acceptance coverage and generation cost are
part of the treatment. If fixed-dose evidence is positive, run a separate natural-coverage and
teacher-cost analysis before choosing the production teacher.
