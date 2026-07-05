# UDD merge value — what merging buys, measured without training

Corpus: 9389 rows, 33 sources, 9389 distinct images.

| signal | best single source | merged corpus | gain |
|---|---|---|---|
| task coverage | 1 / 7 tasks | **7 tasks** | complete |
| language coverage | 3 | **6** (ar, en, id, ko, und, zh) | ×2.0 |
| payload types (fields/boxes/table/full-text) | 2 / 4 | **4** | complete |
| visual diversity (mean pairwise dhash Hamming) | 26.8 avg within-source (max 31.9) | **30.1** | +12% vs avg source |
| vocabulary | 12,951 tokens (largest source) | **72,761** | ×5.6 |
| cross-source redundancy | — | **1.1%** of images have a strict near-dup in another source | merging adds ~99% new data |

![merge value](../report/figures/udd_merge_value.png)

## Verdict

Merging is worth it on dataset-level evidence alone:

1. **No single source spans the training surface.** The best source covers 1 of 7 tasks and 3 of 6 languages; a trainer needing KIE boxes + layout localization + Korean text has no single-source option.
2. **The merged input distribution is measurably wider** — pairwise visual diversity 30.1 vs 26.8 within an average source — without collapsing into duplicates (near-linear vocabulary growth; sources contribute complementary content).
3. **Redundancy is negligible** (1.1% strict near-dups, see `udd_duplicates.md`), so each merged source adds data, not copies — the audit still matters for train/val hygiene (chartqa↔mathvista).

What this analysis **cannot** show is whether the wider distribution transfers to model capability at a fixed budget — that is exactly the GPU task-value ablation (`run_task_value.py`) and the A1/A4 hypothesis runs (`build_task_trainsets.py --group-by task|language`).
