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

## Attention backend

Default is **eager** (`attn="auto"` → eager). flash_attention_2 is reference-only (needs Ampere+,
no win on T4); opt in explicitly if benchmarking. See `notebooks/flash_attention_benchmark.ipynb`.
