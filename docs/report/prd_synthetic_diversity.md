# PRD — Synthetic data **diversity** for understanding-not-memorization fine-tuning

**Status:** living document (iterate in-session). **Owner:** docvlm_eval synth.
**Related:** [`synthetic_data_dto.md`](synthetic_data_dto.md) · [`ablation_plan.md`](ablation_plan.md) ·
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
| document kind | 14 case templates (invoice…webtoon…UI) | `--only` / weights |
| acquisition / lighting | Augraphy presets (scan/photo/fax/historical/screenshot) | `degrade_presets`, `degrade_prob` |
| paper colour / tint | per-doc background palette | `jitter` |
| accent / brand colour | per-doc accent for headings/totals/links | `jitter` |
| typography | per-doc font family (sans/serif/mono mix) | `jitter`, `fonts` |
| layout geometry | per-doc margin + table row-count + field-count variety | `jitter` |
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
- **v2:** layout templates per case (2–3 variants each), handwriting/stamp/seal overlays, more
  languages, distractor fields, photographed-perspective warp (kept geometry-safe for boxes).
- **v3:** compositional multi-page docs; cross-document reasoning; programmatic charts/diagrams with
  next-action GT; curriculum scheduling of difficulty.
Each vN: add knob → regenerate at scale → `measure_diversity` → A0 held-out check → keep if held-out
generalization improves.

## 6. Risks
- More diversity can dilute any single skill at fixed count → control via A0 + per-axis eval.
- Font availability (CJK/Arabic/handwriting) — installed in the notebooks/sweep; report flags tofu.
- Box validity under visual jitter — jitter is photometric/typographic only (no geometry move), so
  GT boxes stay exact (re-resolved from the render each time).
