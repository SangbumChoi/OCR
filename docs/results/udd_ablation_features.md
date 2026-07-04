# UDD — additional ablation-usable features

Corpus: 9389 image-rows. Beyond the wired dimensions (task / language / fold / grounding / derived rationales), these column-derived dimensions have enough bucket support to ablate (equal-N buckets are the constraint):

- **resolution (A7 preprocessing)** — small <0.3MP: **3860**, medium 0.3-0.7MP: **3234**, large >0.7MP: **2295**
- **aspect ratio (doc form factor)** — wide <0.8: **4313**, portrait 1.3-2: **2441**, square 0.8-1.3: **2370**, tall >2 (receipts/screens): **265**
- **QAs per image (packing)** — 1: **7334**, 5+: **1463**, 2-4: **592**
- **answer length (output style)** — short <=15: **5395**, long >60 (abstractive): **2525**, medium 16-60: **1469**
- **answer modality** — textual: **7806**, numeric: **1583**
- **question type** — other: **5450**, what: **2704**, yes/no: **639**, how-many/count: **596**
- **grounding box size (A1 curriculum)** — small <1% page area: **9890**, larger: **7753**
- **license (compliance filter)** — unspecified: **4809**, permissive/tagged: **3380**, other: **900**, non-commercial (cc-nc): **300**

![distributions](../report/figures/udd_ablation_features.png)

## Proposed new arms

| arm | bucket recipe (column filter) | hypothesis |
|---|---|---|
| U-A7 resolution strata | filter by `image_width*image_height` buckets | training on high-res pages moves small-text NED more than equal-N low-res |
| U-A8 QA packing | images with 5+ QAs (`len(instructions)`) vs 1-QA images at equal QA budget | many-QAs-per-image amortizes vision compute AND teaches multi-field reading — beats 1 QA/image |
| U-A9 output style | short (<=15 chars) vs long (>60) answer rows | long-answer training degrades exact/anls on short-answer eval (verbosity bias) — measure the interference |
| U-A10 numeric reasoning | rows whose gold is numeric (16% of corpus) | numeric-only training moves chart/relaxed_acc without touching text extraction |
| U-A11 grounding difficulty | region rows split by box area (55% of boxes <1% page) | curriculum large->small boxes beats mixed-size grounding at equal N (A1 refinement) |
| U-A12 form factor | aspect-ratio buckets (tall receipts/screens vs wide pages) | form-factor-matched training transfers within factor, weakly across |
| license filter | drop cc-by-nc rows (200) from training | compliance-safe training costs nothing measurable (only MTVQA is NC) |

All recipes are pure column filters on the live schema — no new data collection; `build_task_trainsets.py`-style equal-N subsampling + `run_ablation --arm public` run them unchanged. Support caveats: `when/where` question types are too thin (41/20 rows) to ablate; `tall` aspect has 190 rows — pair it with a lowered --count.
