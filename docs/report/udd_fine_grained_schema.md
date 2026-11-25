# UDD Fine-Grained Annotation Schema

## Why refine the labels

UDD's `task` describes the supervised operation, not the appearance of the image. Its current values
(`recognition`, `kie`, `vqa`, `localization`, `table`, `reasoning`, and `classification`) are useful
for broad balancing, but they combine different dimensions. For example, a chart question can be
VQA or reasoning, while a table can be read as text, reconstructed structurally, or queried.

Keep `task` stable for compatibility. Add orthogonal visual-content labels and preserve optional
rationales separately from gold answers. These labels describe annotations and source intent; they
must not be treated as visual predictions inferred from the pixels.

## Visual-content taxonomy

| Field | Values | Meaning |
|---|---|---|
| `visual_type` | `document`, `chart`, `table`, `diagram`, `scene_text`, `interface`, `webpage`, `formula`, `natural_image`, `mixed`, `other` | Primary visual structure needed to interpret the example. |
| `visual_subtype` | Source-specific values such as `receipt`, `form`, `scientific_chart`, `infographic`, `mobile_ui_screenshot`, `printed_formula`, `book_cover`, `document_layout`, `mixed_ocr_suite` | More precise source-supported content category; `other` means unavailable or not safely inferable. |
| `task` | Existing seven values | Supervised operation; do not derive it solely from `visual_type`. |
| `qa_tasks[i]` | Top-level task aligned to QA index `i` | `recognition`, `kie`, `vqa`, `localization`, `table`, `reasoning`, or `classification`. |
| `task_detail[i]` | Source-supported job label aligned to QA index `i` | Fine-grained operation such as `chart_question_answering`, `visual_math_reasoning`, or `table_structure_reconstruction`. |

Examples: AI2D is `diagram / scientific_diagram` with task `vqa`; a row-level grid diagram can be
annotated as `table / labeled_grid_diagram` without changing its VQA task. ChartQA is `chart / chart`
with `task_detail=chart_question_answering`; CORD is `document / receipt` with task `kie` and
`task_detail=receipt_field_extraction`; PubTabNet is `table / scientific_table` with task `table`
and `task_detail=table_structure_reconstruction`. An infographic can be `mixed / infographic` because
it combines text, charts, and diagram-like layout. `mixed` is preferable to forcing a misleading
single visual class.

The initial mapping is source-level metadata. A dataset with heterogeneous image types should
eventually provide row-level labels from its annotations. Do not silently promote heuristic labels
to ground truth.

## Fine-grained operation labels

For analysis and future row-level annotation, subdivide the operation into independently applicable
facets instead of proliferating mutually exclusive top-level tasks:

- **Text**: page transcription, line recognition, word recognition, scene-text spotting, reading
  order, multilingual or vertical-text recognition.
- **Structure**: layout-region detection, region classification, key-value extraction, relation
  linking, table detection, table structure recovery, cell-content recognition, formula-to-LaTeX.
- **Question answering**: direct lookup, entity resolution, cross-region relation, comparison,
  counting, arithmetic, temporal/causal diagram reasoning, chart-data retrieval, chart comparison,
  and document-level synthesis.
- **Reliability conditions**: language/script, page count, document count, rotation, degradation,
  handwriting, density, and whether the target is localized.

These are annotation facets, not claims that every current UDD row already has these labels. The
schema uses source-supported `visual_type`, `visual_subtype`, and `task_detail`; absent row-specific
evidence is not guessed. `task_detail` identifies the source's intended job, not the difficulty or
exact reasoning steps of each question.

## Rationale between instruction and answer

`instructions[i]`, `reasoning[i]`, and `answers[i]` are aligned by index. `reasoning[i]` is an empty
string when no rationale is available. It contains a concise evidence/operation explanation when
the source supplies one; it is not an invitation to generate unrestricted hidden chain-of-thought.
Keep the gold answer unchanged for scoring. During supervised training, a non-empty rationale is
formatted as:

```text
Reasoning: <concise evidence and operation>
Answer: <gold answer>
```

Teacher-generated answers do not inherit a source rationale unless their provenance explicitly
supports that pairing. Reasoning quality should be evaluated separately from answer correctness.
`build_task_trainsets.py --include-reasoning` enables this target format for training pools;
held-out evaluation answers remain answer-only.
How rationales are derived per benchmark (with provenance tiers and a quality gate) is described
in [`benchmark_reasoning_extraction.md`](benchmark_reasoning_extraction.md).

The `fine-grained-v2` migration folds legacy HallusionBench entries ending in "Explain your
answer" into the preceding question's rationale. It preserves the original yes/no gold answer and
removes the duplicate explanation QA so that rationale text is not scored as another answer. Other
missing rationales remain empty.

## Full Annotation

The public schema stores full-page annotation in one `full_annotation` string containing a
versioned JSON object. Its `text` member is the linear transcription; its `html` member can preserve
structured tables, including diagrams arranged as labeled grids. Use ordinary HTML `<table>`,
`<figure>`, and `<figcaption>` elements with stable IDs and `data-bbox` attributes when applicable;
the IDs connect visible labels to their figures without inventing a special `<figure>` token.
Either member may be empty when unavailable, and neither is derived from the other.

- `text`: linear transcription in reading order. It is suitable for OCR metrics such as CER,
  WER, or normalized edit distance. For a table, it can contain the visible cell text in row order,
  but it does not encode reliable cell boundaries, spans, or header relationships.
- `html`: a structured table reconstruction. Rows, cells, nesting, and spans are represented
  with HTML table markup. Grid diagrams may place a label and a figure in each cell; figures are
  linked to labels by IDs, while their descriptions remain in figure captions. It is evaluated with
  structure-aware metrics such as TEDS or GriTS, not ordinary OCR edit distance.

`elements_json` stores localized elements as `{bbox, information, class, kind}` with optional `id`
and `ref`. `class` describes the element (`text`, `figure`, `page_title`); `information` is its
recognized text or concise figure description; `bbox` is `[x1,y1,x2,y2,normalized]` or `null` when
coordinates are unavailable. For example, a label `A` can reference a pumpkin figure through paired
IDs. Hand-curated example annotations must not be presented as original benchmark ground truth.

A table sample may contain both members: one for text fidelity and one for structural fidelity.
Never convert HTML into text by stripping tags and claim structure was preserved. Empty means
unavailable, not a negative label. Keep provenance for pseudo-labels and do not overwrite source
ground truth.

## Compact v3 Schema

The public `fine-grained-v3` revision changes the QA columns to aligned, one-dimensional primary
answers plus explicit alternatives:

```text
instructions: list[string]
answers: list[string]
answer_variants: list[list[string]]
reasoning: list[string]
task_detail: list[string]
qa_tasks: list[string]
```

The nested answers in v2 represented multiple equivalent gold phrasings for the same question,
not multiple reasoning steps or instructions. v3 keeps the first canonical gold in `answers[i]` and
all accepted scoring alternatives in `answer_variants[i]`. Thus each question has one clear primary
target while evaluation retains every accepted reference. The `n_fields` and `n_regions` columns
are removed because they duplicate counts derivable from `elements_json`.

## Compatibility and validation

The scalar `task` remains as the row-level compatibility/filter label. `qa_tasks` is the authoritative
per-question task list, aligned with `instructions`, `answers`, `answer_variants`, `reasoning`, and
`task_detail`. `visual_type` / `visual_subtype` remain row-level because they describe the shared
image. Validation checks every per-question list for matching lengths.
Older rows without these columns remain readable and are treated as empty reasoning with source-derived
visual and task labels.
