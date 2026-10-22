# Extracting reasoning from public benchmarks, and diversifying instructions and answers

UDD ([`unified_loader.md`](unified_loader.md), schema in
[`udd_fine_grained_schema.md`](udd_fine_grained_schema.md)) has an aligned `reasoning[i]` column
for every QA. Before this work, only HallusionBench filled it: **325 of 76,730 QAs**. The public
corpus also has narrow phrasing. Every SROIE row asks the same "Transcribe all the text ...", and
the_cauldron sources rotate 18 brevity suffixes. This document covers two questions:

1. **For each benchmark, what can a rationale honestly be built from?** The rationale must not
   invent a value, and it must be checked.
2. **Under which conditions can instructions and answers be varied** without changing what the
   question asks or what the gold is?

Both are implemented and were run over the full UDD corpus (39,837 rows, 32 sources):
`src/docvlm_eval/unified/rationale.py`, `src/docvlm_eval/unified/diversify.py`,
`configs/instruction_diversity.yaml` and `scripts/derive_udd_reasoning.py`. Results are in
[`../results/udd_reasoning_stats.json`](../results/udd_reasoning_stats.json), with worked
examples per source in [`../results/udd_reasoning_preview.md`](../results/udd_reasoning_preview.md).

## 1. Results at a glance

| | before | after |
| --- | --- | --- |
| QAs | 76,730 | 87,464 (+10,734 derived QAs, each with a rationale) |
| QAs with a rationale | 325 | **17,632** |
| corrupted golds repaired | — | 3,057 (TAT-QA 3,051, Docmatix 6) |
| short golds with a stray trailing period ("Mean.") | 23,076 | 0 |
| (instruction, answer) training variants | 76,730 | 197,036 |
| distinct instruction templates (digits / quoted spans masked) | 40,876 | 103,084 |
| distinct trailing format clauses | 28,447 | 52,976 |
| answer-format entropy | 0 bits (1 format) | 2.31 bits (9 formats) |
| accepted answer variants per QA | 1.00 | 1.66 |

## 2. Principle: a rationale is evidence plus an operation, and it is checked

The schema already defines `reasoning[i]` as a *concise evidence/operation explanation*, not
free chain-of-thought, and it keeps the gold unchanged. We add three rules.

**Never invent.** Every value a rationale states must come from the row: the instruction, the
answers, `elements_json`, `full_annotation`, or the source's own rationale. If the material is not
there, the QA stays answer-only. A rationale that only restates the answer ("The tallest bar is
Mean") teaches nothing and is rejected.

**Tier every rationale** (recorded per QA in `reasoning_provenance`):

| tier | meaning | example |
| --- | --- | --- |
| `source` | the benchmark shipped it | HallusionBench `gt_answer_details`; TAT-QA span evidence |
| `source_verified` | shipped, and re-checked | the TAT-QA expression evaluates to the gold |
| `derived_exact` | computed from the row's own ground truth | CORD totals, FUNSD links, table cells, layout counts |
| `evidence_match` | the answer was matched to OCR/UI text in the row | OCR-VQA cover words, ScreenQA elements |

A future `teacher` tier (a VLM-written rationale) should be kept only under rejection sampling:
the teacher's final answer must equal the gold, and its cited strings must exist in the row.

**Gate every rationale** (`rationale.quality`). A rationale is rejected when any of these holds:

- it is empty or longer than 600 characters;
- it is *tautological*: it adds fewer than 2 content words beyond the question and answer (fewer
  than 1 for `source` tiers, whose corrections are often short, e.g. "…the most in **2005**");
- none of its evidence strings is present in the row payload;
- for arithmetic, its last number does not equal the gold at the gold's rounding;
- for yes/no golds, it opens with the opposite verdict ("Yes. …" for gold "no").

A rejected **source** rationale is moved to `reasoning_rejected[i]`, so it is kept but never
trained on.

## 3. Per-benchmark strategy

Yields are measured on the full corpus (`udd_reasoning_stats.json`). "Derived QAs" are new
questions whose gold is computed from the same ground truth. The template paraphrase pools for
them are in the config.

| source | material in UDD | rationale recipe | tier | QAs with rationale (derived) |
| --- | --- | --- | --- | --- |
| **tatqa** | the answer string *contains* the original derivation: `"74.5-26.0\nAnswer: 48.5."` | split it out, **repair the gold** to `48.5`, and evaluate the expression with a safe evaluator (accounting negatives `(6,919)`, both percent readings, `scale: percent` ×100). Count derivations (`2019\nAnswer: 1.`) are checked by counting. Plan-only text ("Find the year with …") is dropped. | `source_verified` (2,513) / `source` (479) | 2,992 |
| **hallusionbench** | `gt_answer_details` (already in `reasoning`) | gate only: 123 restate the question, 3 contradict the gold | `source` | 199 (126 moved aside) |
| **cord** | the full receipt parse as the KIE gold | subtotal = Σ line prices; total = subtotal + tax + service − discount; change = cash − total; most expensive item. **Emitted only when the printed numbers satisfy the relation.** Handles `60.000`-style thousands. | `derived_exact` | 164 (164) |
| **funsd** | the KIE gold is the source key→value linking, plus word boxes with BIO tags | merge BIO words into entities, ground both the key and the value, and state the relation ("to the right of the label") | `derived_exact` | 24 (24) |
| **ocrvqa** | OCR words with normalized boxes; 5 QAs per cover | title/author: answer words matched on the cover (≥60% of key words) with their position. "Is this book related to X?": answered from the **same row's genre answer** (cross-QA consistency). | `evidence_match` / `derived_exact` | 2,369 |
| **screenqa** | UI elements (text) whose content forms the sentence answer | cite the elements whose text is in the answer | `evidence_match` | 1,338 |
| **pubtabnet** | full HTML table | span-free tables only: body-row count, and cell lookup by unique (row label, column header) | `derived_exact` | 2,471 (2,471) |
| **omnidocbench** | layout blocks with class, text and box | page number, title (unique blocks), and per-class counts with positions | `derived_exact` | 3,920 (3,920) |
| **doclaynet**, **publaynet** | layout boxes with class | per-class counts with positions (≤6 instances, so the count is checkable) | `derived_exact` | 2,157 / 1,998 (all derived) |
| **docmatix** | 6 answers restate the question before `\nAnswer:` | gold repair only | — | 0 |
| chartqa, plotqa, dvqa | question + short answer only (UDD keeps no chart data) | none. A "the tallest bar is X" rationale would be tautological. **Unlock:** ChartQA ships the underlying CSV tables, and DVQA/PlotQA ship bar/point metadata. With that data, read/compare/arithmetic chains become `derived_exact`. | — | 0 |
| infovqa | question + answer | none in UDD. **Unlock:** the source has `operation/reasoning` labels (arithmetic / comparison / counting) and Textract OCR with boxes. Keep both in the extractor, then ground the answer span in the OCR. | — | 0 |
| docvqa, textvqa, stvqa, mtvqa | question + answer | none in UDD. **Unlock:** OCR evidence. TextVQA ships `ocr_tokens`; DocVQA/MTVQA need the pseudo-label OCR filler ([`pseudo_labeling.md`](pseudo_labeling.md)). Then: answer span located in an OCR line → `evidence_match`. | — | 0 |
| charxiv | one reasoning QA per figure | none in UDD. **Unlock:** CharXiv's 4 descriptive QAs per figure (axis labels, legend, …) are natural sub-steps for the reasoning QA. | — | 0 |
| mathvista, ai2d, ocrbench, ocrbench_v2, visualmrc, rvl_cdip | question + answer (+ options) | none. These answers need visual reasoning that the row does not encode. Use the `teacher` tier with rejection sampling. | — | 0 |
| iam, sroie, im2latex, latexocr, synthdog_* | transcription | not applicable (the target *is* the reading). Fine-grained reading probes already exist (`derive_text_probes`). | — | 0 |

**Takeaway:** checkable rationales come from **structure**, not from QA pairs. The six sources
with structure in UDD (receipt parse, form links, OCR words, UI elements, tables, layout) produce
almost all of the 17.6k. The biggest remaining wins are upstream: keep ChartQA tables, InfoVQA
operation labels and OCR, TextVQA OCR tokens and CharXiv descriptive QAs in the extractors. That
would make up to 19,510 more QAs (ChartQA, DVQA, PlotQA, InfoVQA, TextVQA, CharXiv) eligible for
verified rationales.

### Example rationales (from the run)

```text
tatqa     Q: What is the percentage change in the end-of-year valuation allowance from 2017 to 2018?
          gold (was "(104,858-159,154)/159,154\nAnswer: -34.12.") -> -34.12
          R: Compute (104,858-159,154)/159,154 = -0.3412; as a percentage: -34.12%.     [source_verified]
cord      Q: What is the total amount to pay?   gold 174,600
          R: 194,000 (subtotal) − 19,400 (discount) = 174,600, the printed total.      [derived_exact]
ocrvqa    Q: Is this book related to Travel?    gold No
          R: The cover is categorised as 'History', a different category from 'Travel'. [derived_exact]
pubtabnet Q: In the table, what is the 'MAF' value for '-1022 C>G'?   gold 0.043
          R: Find the row '-1022 C>G' and the column 'MAF'; the cell where they meet reads '0.043'.
```

## 4. Data-quality issues found along the way

| issue | scale | fix |
| --- | --- | --- |
| TAT-QA golds contain the derivation (`"74.5-26.0\nAnswer: 48.5."`), so the scored gold is not the answer | 3,051 golds | repaired in the derived rows; the derivation becomes the rationale |
| the_cauldron golds end in a stray period (`"Mean."`): 100% of DVQA/PlotQA, 96% of ST-VQA | 23,076 short golds | canonical form strips it; the original form is kept as an accepted variant |
| **pixel boxes were not rescaled when images were resized to ≤1000 px** (`UnifiedLoader.iter`) | CORD 39% / ScreenQA 60% of boxes fall outside the stored image (the rest are mis-scaled too) | `normalize_pixel_boxes()` now converts them to [0,1] **before** resizing. The rationale code ignores pixel boxes until UDD is rebuilt. |
| TextVQA primary gold is the *first* annotator's answer, not the consensus (`"nous les gosses"` vs 6× `"dakota"`) | TextVQA-style multi-annotator QAs | `canon_answers` now orders by annotator count (ties keep source order); takes effect on rebuild |
| HallusionBench details that restate the question or contradict the gold | 126 of 325 | moved to `reasoning_rejected` |

## 5. Instruction / answer diversification: the conditions

**Rule 1: instruction and answer vary together.** A *format* binds an instruction clause to an
answer renderer. A variant never asks for a number and expects a sentence.

| format | condition (`when`) | clause (one of a pool) | target |
| --- | --- | --- | --- |
| `short` | short answer | "Answer with a single word or phrase." … | canonical gold |
| `number` | numeric gold | "Answer with a number only." … | `2305` (separators/currency stripped) |
| `yes_no` | yes/no gold | "Reply with 'yes' or 'no' only." | `no` |
| `sentence` | short answer **and English question** | "Answer in one complete sentence." | `The answer is X.` |
| `json` | short answer | `Return JSON of the form {"answer": <value>}.` | `{"answer": "X"}` |
| `mcq_index` | options listed **and** the gold is one of them | "Reply with the option number only." | `3` |
| `reasoned` | **a gate-passed rationale exists** | "Explain briefly, then give the final answer after 'Answer:'." … | `Reasoning: …\nAnswer: X` |
| `grounded` | evidence box **and** short answer | "… as: answer \| x1,y1,x2,y2 (0-1000 scale)." | `D. J. Landro \| 185,219,263,230` |

**Rule 2: the question's meaning and the gold value never change.** Human-written source
questions keep their wording. Only a *conflicting* format clause is removed: the 18 measured
the_cauldron suffixes, a few generic ones, and close variants. Semantic clauses such as TAT-QA's "Answer scale should
be: percent." are kept. Derived questions draw from meaning-preserving paraphrase pools. The gold
changes only in rendering, and canonical variants (`2,305.00` ≡ `2305` ≡ `2305.00`, `Yes` ≡ `yes`)
are added for scoring.

**Rule 3: some targets are never re-rendered.** These keep only the original:

- transcriptions: recognition / table / layout tasks, "transcribe …", "convert … to LaTeX";
- abstain golds (`<no answer>`);
- JSON/HTML targets (these may only get `reasoned`).

English answer templates apply only to English questions; otherwise a Japanese MTVQA question
would get "The answer is 午前9時…".

**Rule 4: sampling is deterministic and target-aware.** For each QA, the original (v0) is kept and
`variants_per_qa` formats are drawn by weight. At least one variant must have a *different target
form* from the original. A new clause for the same target still counts as instruction diversity,
so it may fill the remaining slots. Everything is a hash of (`seed`, `sample_id`, QA index), so a
config plus the corpus fully determines the output. Per-source overrides cover recognition sources
(1 variant) and long free-text sources (no sentence/json), and give TAT-QA more weight on
`reasoned`.

Resulting format mix over the corpus: original 84,507 · short 40,195 · json 24,057 · sentence
21,801 · reasoned 13,514 · number 9,310 · yes_no 2,640 · grounded 527 · mcq_index 485.

## 6. Usage

```bash
# full corpus straight from the Hub (column projection: the image column is never downloaded)
python scripts/derive_udd_reasoning.py --from-hub
# a local dump or parquet shard, optionally a subset of sources
python scripts/derive_udd_reasoning.py --rows train-00001-of-00005.parquet --sources tatqa cord
```

Outputs:

- `examples/udd_reasoning/rows.jsonl`: augmented rows, git-ignored. Contains repaired `answers`,
  filled `reasoning`, derived QAs (`task_detail = derived:<deriver>`), and the aligned
  `reasoning_provenance`, `evidence_box` and `reasoning_rejected` columns.
- `examples/udd_reasoning/variants.jsonl`: one Sample-shaped record per variant, git-ignored. Each
  has `question`, `answers`, `metric`, and `meta.{format, final_answers, reasoning_provenance, fold}`.
  It is keyed to its image by `udd_sample_id`.
- `docs/results/udd_reasoning_stats.json` and `docs/results/udd_reasoning_preview.md`.

**Caveats.**

- Variants are **training** targets. Held-out evaluation stays answer-only, as the schema
  requires. Score a `reasoned` prediction on its final answer (`meta.final_answers`).
- Images are not copied. Join on `udd_sample_id`, as `build_task_trainsets.py` does.
- Rows keep the HF column names but add three columns (`reasoning_provenance`, `evidence_box`,
  `reasoning_rejected`). Pushing them to the Hub would be a schema revision.

## 7. Suggested experiments (A2 on public data)

These runs isolate what each change buys. Train on the same images and steps, and evaluate on the
UDD held-out fold, answer-only:

1. **Repair only vs original golds** (TAT-QA, DVQA/PlotQA periods). This measures how much the
   corrupted golds were costing.
2. **`reasoned` vs answer-only** on the 17.6k rationale QAs, split by tier. Does `derived_exact`
   help as much as `source_verified`? Does `evidence_match` help at all?
3. **Format diversity on vs off** (`variants_per_qa` 0 / 2 / 4). Check instruction-following on
   unseen clause wordings, and whether diversity costs accuracy on the original prompts.
