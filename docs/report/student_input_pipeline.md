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
localized element key/value. The full experiment excludes offline teacher targets from tokenizer
fitting so teacher-family ablations retain one gold-defined token-to-ID contract; byte fallback
still encodes every teacher response. The saved `tokenizer_config.json` records this corpus policy,
special-token IDs, requested and actual vocabulary sizes, normalization, minimum frequency, and a
SHA-256 fingerprint of the complete token-to-ID contract. The collator rejects any token ID outside
the model vocabulary before the forward pass.

## UDD expansion

`UDDStudentDataset` scans metadata without decoding images, then decodes an image only when an
example is requested. It expands:

- each native `instructions[i]` and `answers[i]` pair into one generative example;
- each accepted `teacher_answers[i]` as a deterministic alternative target without replacing gold;
- each localized field or region into one evidence-box example;
- duplicate labels into text-qualified or reading-order-qualified questions so a target is not
  ambiguous.

Every expanded record retains task, source, language, mixture component, target source, sample ID,
image identity, and UDD image geometry. The same image can therefore supply several tasks without
duplicating stored pixels. `BalancedGroupBatchSampler` can balance by task, source, language, or
mixture component with explicit weights and deterministic epoch seeds. Under `torchrun`, it draws
one global batch and gives each rank a disjoint local slice.

The dense control can also group the global batch into log2 aspect-ratio buckets. It applies the
same sample/epoch rotation hash as the collator before assigning a bucket, so a 90-degree augmented
page changes orientation in both places. The bucket width is configurable and defaults to 0.5
octaves. Missing or invalid UDD dimensions enter a separate `unknown` bucket. Bucketing is disabled
for the packed default because batch peers do not determine its visual allocation.

Bucketing does not replace group balancing. For group weight \(w_g\), group size \(n_g\), and the
number \(n_{gb}\) of its examples in bucket \(b\), the sampler chooses a bucket with mass
\(\sum_g w_g n_{gb}/n_g\), then chooses groups within that bucket proportional to
\(w_g n_{gb}/n_g\). Marginalizing over buckets recovers the requested group distribution while
keeping every distributed global batch shape-homogeneous.

## Spatial contract

The collator applies a deterministic, epoch-varying 0, 90, 180, or 270 degree clockwise rotation.
The choice is a stable hash of augmentation seed, epoch, and sample ID, so exact checkpoint resume
does not depend on worker RNG history. Its class is the orientation target. Evidence boxes receive
the exact same transform:

| Rotation | `[x1, y1, x2, y2]` becomes |
| --- | --- |
| 0 | `[x1, y1, x2, y2]` |
| 90 clockwise | `[1-y2, x1, 1-y1, x2]` |
| 180 | `[1-x2, 1-y2, 1-x1, 1-y1]` |
| 270 clockwise | `[y1, 1-x2, y2, 1-x1]` |

Images retain aspect ratio and are resized only when their long side exceeds 896 pixels. The
default `packed` sequence mode pads each image only to its own final patch boundary, unfolds
`[total_patches, 3, 14, 14]`, and carries canonical position IDs plus cumulative sequence offsets.
The ViT and resampler keep projections and MLPs concatenated while attention is block-diagonal, so
no image attends to another image and no batch-level visual padding enters execution. The `auto`
backend uses compiled PyTorch FlexAttention block masks on supported CUDA systems and falls back to
range-wise SDPA elsewhere. This exact path reuses the same Conv2d patch weights.

Dense controls remain available. `batch_adaptive` pads to the batch's patch-aligned maximum height
and width; `fixed_square` pads every image to 896 by 896. Neither dimension can exceed 896, or 64
patches at patch size 14, so the 4,096 visual-position limit remains hard. Targets are converted
from original-image coordinates to the normalized canonical 896-square coordinate canvas after
rotation and resize, regardless of visual execution mode. The same document therefore keeps
identical box targets across batch compositions. The transformed numbers supervise both the
generated box string and the box head.

Synthetic multi-page packets are composed before this resize. The default page grid packs three
portrait pages into two columns instead of one tall strip, preserving about 313px rather than
209px of effective page width at the 896px long-side limit. Exact page origins remain in sample
metadata, so cross-page evidence is auditable even though the student consumes one image tensor.
Vertical composition remains a matched data ablation.

Independent-document bundles use the same pre-resize contract but retain a higher-level source
map. The default grid packs an audited filing, market snapshot, and analyst memo into one canvas;
`document_origins_px`, `document_sizes_px`, and `page_document_ids` keep evidence attribution
exact. A cross-document QA therefore reaches the one-image student unchanged while evaluation can
distinguish local, two-source, and three-source reasoning. Vertical document composition remains a
matched control and is intentionally resolution-stressed.

Dense `pixel_mask` is pooled into a patch mask. Invalid patches are excluded from ViT
self-attention, resampler cross-attention, and vision pooling. Packed inputs carry only valid patch
slots. Both modes use a fixed two-dimensional 64-by-64 position grid rather than the flattened
batch tensor width. A sample's valid patch position IDs therefore remain independent of which
other examples share its batch. Dense and packed parity tests cover logits, total loss,
orientation output, greedy generation, and selected vision features.

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
- visual sequence mode (`packed` or `dense`);
- packed attention backend (`auto`, `flex`, or `loop`);
- visual canvas mode (`batch_adaptive` or `fixed_square`);
- rotation-aware aspect-ratio bucketing and its log2 bucket width;
- quarter-turn augmentation probability;
- upscaling policy;
- contrastive objective switch;
- task/source/language balancing key and group weights;
- cross-tokenizer teacher target probability, minimum quality score, and deterministic seed.

`StudentCollatorConfig.from_blueprint()` binds these controls to the model's patch size, visual
position count, and language vocabulary.

## Runner integration

The input path is consumed by the mixed-precision, token-scheduled, exactly resumable runner
documented in [`student_pretraining_runner.md`](student_pretraining_runner.md). Batch provenance is
retained for auditing but stripped before model calls. The runner records cumulative dense visual
tokens per sample, executed-token utilization, valid-token fraction, sequence-aware student FLOPs,
and the resolved `train/visual_attention_backend`. The packed path removes visual padding FLOPs,
but analytical savings are not evidence of higher wall-clock throughput. Fused-kernel latency and
peak-memory measurements remain deployment gates.
