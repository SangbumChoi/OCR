# Native student input pipeline

## Purpose

The approximately 800M student consumes the published UDD schema directly instead of routing data
through a model-specific chat processor. The implementation lives in
[`docvlm_eval.student.data`](../../src/docvlm_eval/student/data.py) and
[`docvlm_eval.student.tokenizer`](../../src/docvlm_eval/student/tokenizer.py).

## Tokenizer

The student uses a newly trained 64k byte-level BPE tokenizer rather than inheriting a model
tokenizer whose IDs would silently imply transferred semantics. NFC normalization preserves exact
document strings without compatibility-folding distinct glyphs. The byte alphabet guarantees a
path for every UTF-8 sequence, including CJK, Korean, Arabic, financial symbols, HTML, and formulas.

```bash
pip install -e ".[student,models]"
python scripts/train_student_tokenizer.py \
  --repo danelcsb/UDD \
  --output artifacts/student_tokenizer
```

Training text includes every instruction and answer variant, full-page text, table HTML, and
localized element key/value. The saved `tokenizer_config.json` records special-token IDs, requested
and actual vocabulary sizes, normalization, and minimum frequency. The collator rejects any token
ID outside the model vocabulary before the forward pass.

## UDD expansion

`UDDStudentDataset` scans metadata without decoding images, then decodes an image only when an
example is requested. It expands:

- each native `instructions[i]` and `answers[i]` pair into one generative example;
- each localized field or region into one evidence-box example;
- duplicate labels into text-qualified or reading-order-qualified questions so a target is not
  ambiguous.

Every expanded record retains task, source, language, sample ID, and image identity. The same image
can therefore supply several tasks without duplicating stored pixels. `BalancedGroupBatchSampler`
can balance by task, source, or language with explicit weights and deterministic epoch seeds.

## Spatial contract

The collator applies a uniformly sampled 0, 90, 180, or 270 degree clockwise rotation. Its class is
the orientation target. Evidence boxes receive the exact same transform:

| Rotation | `[x1, y1, x2, y2]` becomes |
| --- | --- |
| 0 | `[x1, y1, x2, y2]` |
| 90 clockwise | `[1-y2, x1, 1-y1, x2]` |
| 180 | `[1-x2, 1-y2, 1-x1, 1-y1]` |
| 270 clockwise | `[y1, 1-x2, y2, 1-x1]` |

Images retain aspect ratio and are padded at the bottom and right into a fixed 896 by 896 canvas.
This is exactly 64 by 64 patches at patch size 14, matching the 4,096 visual-position limit.
Targets are converted from original-image coordinates to normalized canvas coordinates after
rotation and resize. The same transformed numbers supervise both the generated box string and the
box head.

`pixel_mask` is pooled into a patch mask. Invalid patches are excluded from ViT self-attention,
resampler cross-attention, and vision pooling. Fixed canvas positions also make a sample's visual
position IDs independent of which other examples share its batch.

## Text and loss contract

The sequence is:

```text
<bos>User: {instruction}
Assistant:{answer}<eos>
```

Prompt and right-padding labels are `-100`; only answer and EOS tokens receive autoregressive loss.
The collator records the final prompt position, and the box head pools that hidden state before gold
answer tokens. This prevents answer leakage into box regression.

Contrastive batches carry an integer image identity. QA and grounding views of the same image are
multi-positive pairs, so one view is never treated as another view's negative. Text-only replay is
supported in separate batches without creating fake images or orientation labels.

## Adjustable controls

The authoritative defaults are under `training.pretraining.input_pipeline` in
[`configs/sub1b_architecture.yaml`](../../configs/sub1b_architecture.yaml):

- maximum text tokens;
- image long side and visual-token budget;
- quarter-turn augmentation probability;
- upscaling policy;
- contrastive objective switch;
- task/source/language balancing key and group weights.

`StudentCollatorConfig.from_blueprint()` binds these controls to the model's patch size, visual
position count, and language vocabulary.

## Remaining boundary

This module establishes trustworthy model inputs; it does not claim a completed pretraining run.
The next layer must provide teacher logits/features, mixed-precision distributed optimization,
token-based scheduling, resumable data/sampler state, and held-out evaluation by source, task, and
language. True NaViT-style packed visual sequences remain an efficiency ablation beyond the fixed
masked canvas baseline.
