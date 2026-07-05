# Repository conventions

## Language: English only

**Every document and code artefact in this repository is written in English.** This includes:

- all Markdown / reports / READMEs (under `docs/` and anywhere else),
- code comments and docstrings,
- commit messages, and any generated text artefacts.

Do **not** write Korean (or any non-English) prose, and do **not** add mixed-language glosses such
as `cheque / 수표`. Keep document-type names and labels in English only. When asked to add or edit
docs, produce English. Chat replies to the user may be in the user's language, but anything written
into the repo must be English.

## Where things live

- **Reports & results live under `docs/`** — `docs/report/` (technical report, taxonomies,
  figures, ablation plan, insights) and `docs/results/` (comparison table, matrices, probe
  signals). Generators write there (`build_insights`, `build_report`, `plot_*`, `run_matrix`,
  `analyze_*`, the CLI defaults). Do not recreate top-level `report/` or `results/`.
- **`docs/plan.md`** is the project's north-star narrative and reading order.

## UDD (Universal Document Dataset)

The public-data track lives in `src/docvlm_eval/unified/` (task-typed loader, HF layer, enrichment,
phash dedup) and builds to the Hub dataset `danelcsb/UDD` via `scripts/build_udd.py`. Schema
invariants that must hold (enforced by `validate_payload_shapes` inside every `safety_check` —
keep it that way): **one row per image**; `instructions: list[str]` index-paired with
`answers: list[list[str]]` (inner list = gold VARIANTS of one answer, never answers to different
questions); `len(instructions) == len(answers) >= 1`; fields' `key`/`value` are required strings;
boxes are always `[x1,y1,x2,y2,normalized]|null`. The in-memory DTO keeps the flat-XOR-grouped
rule: a `UnifiedSample` populates `instruction`+`answers` OR `qas`, never both. Regenerable build
outputs stay git-ignored (`data/udd*`); the narrative doc is `docs/report/unified_loader.md`, and
the public-data ablation arms are composed by `scripts/run_udd_ablation.py`
(see `docs/report/ablation_plan.md` §11b).

## Attention backend

Default is **eager** (`attn="auto"` → eager). flash_attention_2 is reference-only (needs Ampere+,
no win on T4); opt in explicitly if benchmarking. See `notebooks/flash_attention_benchmark.ipynb`.
