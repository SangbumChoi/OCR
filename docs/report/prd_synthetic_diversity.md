# PRD — Synthetic data **diversity** for understanding-not-memorization fine-tuning

**Status:** living document (iterate in-session). **Owner:** docvlm_eval synth.
**Related:** [`synthetic_data_dto.md`](synthetic_data_dto.md) · [`ablation_plan.md`](ablation_plan.md) ·
[`synth_generation_survey.md`](synth_generation_survey.md) (open-source technique survey) ·
generator `scripts/make_realistic_cases.py` + `src/docvlm_eval/synth/`.

## 1. Problem & goal

The generator can produce **infinite** labelled document images, which is its strength *and* its
risk: a model can **memorize templates** instead of **understanding documents**. The A0 experiment
(`ablation_plan.md`) shows the cure is **diversity**, not just count — held-out generalization rises
with diversity, while count alone drives memorization (train→1.0, held-out flat).

**Goal:** maximize, measurably and controllably, the diversity of the synthetic corpus along the two
axes that make a document hard, so fine-tuning teaches *reading + reasoning* that transfers to unseen
documents.

## 2. The two axes (and their sub-dimensions)

### Axis A — Visual diversity (the *image* distribution)
| Sub-dimension | Method | Knob |
| --- | --- | --- |
| document kind | 21 case templates (invoice…UI, audit packet, investment dossier, and five hard families) | `--only` / weights |
| acquisition / lighting | Augraphy presets (scan/photo/fax/historical/screenshot) | `degrade_presets`, `degrade_prob` |
| photographed geometry | same-frame perspective warp with exact homography box transforms | `perspective_prob`, `perspective_max_inset_fraction`, `perspective_min_area_ratio` |
| degradation validity | local evidence visibility + clean/degraded crop correlation | `validate_degraded_evidence`, `degraded_min_structure_correlation`, `degrade_max_attempts` |
| paper colour / tint | per-doc background palette | `jitter` |
| accent / brand colour | per-doc accent for headings/totals/links | `jitter` |
| typography | per-doc font family (sans/serif/mono mix) | `jitter`, `fonts` |
| layout geometry | per-doc margin + table row-count + field-count variety | `jitter` |
| hard-document structure | three semantic-preserving layouts per hard family | `hard_layout_families`, `--hard-layout` |
| authored document marks | evidence-safe handwriting, rectangular stamps, and circular seals | `overlay_prob`, `overlay_types`, `overlay_max_count` |
| resolution / optics | DPI + resize + tiling | `dpi`, `target_long_side`, `tiling_n_max` |
| script / language | locale-aware content (en/es/ko/ja/zh/ar) | `languages`, `language_weights` |

### Axis B — Annotation difficulty (the *label* distribution)
| Sub-dimension | Example | Where |
| --- | --- | --- |
| extraction (easy) | read a field | `field`/`qa` |
| localisation | where is word / table | `ask_where`/`ask_region` |
| counting | how many `$` / messages / panels | `ask_count` |
| arithmetic | sum of line items | `ask_aggregate sum` |
| **multi-step / accounting** | total **+10% tax**; **diff** of extremes | `ask_aggregate diff`, H-accounting |
| **strict field-parse** | **surname only** from MRZ | H-extract-strict |
| affordance / next-action | what should the user do next | H-action |
| comprehension | what is the conversation about | H-comprehension |
| consistency / abstain | dual-amount agree? redacted value? | probes |

### Reasoning heuristics per design (model-free, infinite)
Because the generator authors every value, we can derive **reasoning** GT by heuristic — no model, gold
by construction — and attach a rationale. Each design gets questions natural to its content, so scaling
the data scales the reasoning variety, not just templates:

| Design | Heuristic-derived reasoning (examples) |
| --- | --- |
| invoice | #line-items; which row is largest (compare Amount col); sum / first+last / max−min; +10% tax |
| bank statement | #deposits vs #withdrawals; largest |transaction|; max−min balance |
| webtoon | bubbles on **left vs right** (alternating sides) + which side has more |
| mobile chat | #incoming vs #outgoing; who sent more; topic; next action |
| checkbox form | #checked vs #unchecked; which options selected |
| id / passport | surname-only parse from the MRZ encoding |
| RTL / ancient | reading direction; **which language/script** |
| (any table) | row/col counts, per-column sum/mean, row comparison — large combinatorial space |

Principle: for each design enumerate the *facts we already know* (counts, extremes, positions, language,
relations) and emit a Q+A+rationale for each. Tables are the richest source (the combinatorics of
rows × columns × reductions). The rationale doubles as the A2 chain-of-thought target.

## 3. Non-goals
- Photorealistic GAN imagery (we keep GT-exact HTML/PDF rendering).
- Real PII / real documents (synthetic only).
- Replacing public benchmarks (this *augments* them for fine-tuning).

## 4. Diversity (richness) metrics — how we know it's "rich enough"

Computed by `scripts/measure_diversity.py` over a generated corpus → `synthetic_diversity_report.md`:

1. **Doc-type coverage / entropy** — all kinds present, roughly balanced.
2. **Visual spread** — variance of mean brightness, colour (RGB), aspect ratio, image size across docs.
3. **Layout spread** — variance of #fields, table row-counts, page sizes.
4. **Task-type distribution** — count per `answer_type` (extraction vs localisation vs reasoning …);
   target: reasoning/understanding ≥ ~40% of QAs.
5. **Language / script distribution** — share per language.
6. **Near-duplicate rate** — fraction of images with near-identical perceptual hash (lower = richer).
7. **Unique-content rate** — distinct field values / GT answers ÷ total.

**Acceptance (v1 "rich"):** ≥14 doc types; near-duplicate rate < 5% at count≥20; brightness/colour
CoV > 0.1; ≥6 distinct `answer_type` families with reasoning ≥40%; ≥1 non-Latin script when enabled.

## 5. Iteration plan (so the session can keep enhancing)
- **v1 (this pass):** per-doc visual jitter (paper/accent/font/margin) + table row-count variety +
  the difficulty tasks above; richness report + acceptance check. ← implemented now. Two correctness
  bugs surfaced and were fixed while driving the memorization metric to 0 (both are exactly the
  "memorize vs understand" failure mode the PRD targets):
  - **Cross-variant RNG aliasing.** The generator seeded the global RNG once per *variant*, so cases
    late in the list (e.g. `website`) inherited a cumulative RNG state that aliased across variants —
    only ~4 distinct renders + identical gold labels over 20 variants (true-dup ≈ 0.08). Fix: reseed
    per `(seed, variant, case)` with a stable hash → every case gets an independent, reproducible
    stream → true-dup **0.0** at count=20.
  - **ID/passport MRZ lost off-page.** A page-tall card with the MRZ in `position:absolute; bottom:0`
    spilled the strip onto a 2nd PDF page (WeasyPrint), so the rasterised page-0 image *and* the MRZ
    region GT were empty. Fix: normal-flow compact layout that fits one page (the `_FIXED_LAYOUT`
    cases also opt out of @page-margin/font-size jitter so they never overflow). MRZ now renders and
    its `extract-strict` + `region` GT resolve.
- **v2a (implemented):** executable semantic graphs; difficulty levels 1–5; graph-authored hard
  table, chart, investment, and scientific-paper families; multi-box evidence for SFT/RLVR; content
  and template fingerprints; deterministic split assignment and a cross-split leakage validator.
- **v2b (multilingual projection, photographed geometry, and hard-layout diversity implemented):**
  the four executable hard families now render English, Spanish, Korean, Japanese, and Simplified
  Chinese titles, tables, body text, questions, text answers, and rationales from one validated
  locale catalog. Each family has classic, compact, and report layouts that preserve the latent
  graph and answers while changing page geometry, section order, and spatial grouping.
  Photo-style cases can warp the post-resize raster and every spatial target through the same
  deterministic homography; counterfactual pairs share both layout and warp. Evidence-safe
  handwriting, stamp, and seal overlays are authored before perspective, carry grounded
  recognition QAs, update full-page OCR targets, and remain active with `--no-degrade`.
- **v2c (degradation evidence gate implemented):** final-resolution clean boxes are checked against
  local background pixels; each degraded candidate must preserve both visibility and padded-crop
  structure. A 17-family by 5-preset by 3-seed calibration observed 1,018 valid box crops; the
  minimum valid correlation was 0.269, fixing the conservative default at 0.25. Eleven Augraphy
  runtime failures in 255 candidates motivated bounded deterministic retries with accepted
  seed/attempt provenance.
- **v3 (multi-page, cross-document, and composition curriculum implemented):** a three-page procurement packet composes order,
  receiving, and payment records with six-box cross-page reconciliation, quantity, and consistency
  targets. The renderer can preserve every PDF page in exact-offset vertical or compute-aware grid
  canvases; grid is the default for small-model resolution efficiency. A separate investment
  dossier composes an audited filing, exchange snapshot, and external analyst memo as independent
  source documents. Its executable graph supervises valuation, growth, claim discrepancy,
  source reliability, and next action with exact cross-document evidence. A programmatic scientific
  workflow adds directed topology, edge reading, path products, parallel-path aggregation, and
  expected-count questions across three semantic-preserving layouts. Scientific result pages now
  add grounded quantitative figure marks and balanced correct/incorrect Results claims whose
  consistency is recomputed from four table values plus the visible claim. Exact page/document
  counts now survive the UDD bridge and drive a secondary, task-preserving optimizer-step
  curriculum from single pages through multi-page packets to cross-document dossiers.
Each vN: add knob → regenerate at scale → `measure_diversity` → A0 held-out check → keep if held-out
generalization improves.

## 6. Risks
- More diversity can dilute any single skill at fixed count → control via A0 + per-axis eval.
- Font availability (CJK/Arabic/handwriting) — installed in the notebooks/sweep; report flags tofu.
- Box validity under visual jitter — jitter is photometric/typographic only (no geometry move), so
  GT boxes stay exact (re-resolved from the render each time). Perspective is the explicit
  exception: every supported spatial view is transformed by the sampled homography and stored as a
  clipped axis-aligned envelope before the clean and degraded pixel gates run.
