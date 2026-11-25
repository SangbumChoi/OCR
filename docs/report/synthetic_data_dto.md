# Synthetic data: the ground-truth DTO and config-driven ablation control

This document explains how we generate fine-tuning data whose **distribution matches reality** and
whose **ground truth carries every factor the ablations want to vary**
([`ablation_plan.md`](ablation_plan.md)). One config file fully determines a dataset
variant, so each ablation arm is reproducible and differs from its control in exactly one factor.

Reading order: [`document_type_taxonomy.md`](document_type_taxonomy.md) (what documents exist and
their stressors) → this file (how we synthesise them with built-in GT) →
[`ablation_plan.md`](ablation_plan.md) (how the GT factors are switched per experiment).

## 0. The two quality axes we scale (and the memorization guard)

A synthetic corpus is only useful for fine-tuning if we can grow it along the two axes that make a
document *hard*, while proving the model **understands** rather than **memorizes** the templates:

- **Visual diversity** — the *image* distribution: document kinds (18 current families),
  lighting, colour cast, contrast, resolution, script. Knobs: `degrade_presets`/`degrade_prob`
  (acquisition + lighting; arm `D_visual_diverse`), `languages` (script), `dpi`/`target_long_side`.
- **Annotation difficulty** — the *label* distribution: not just extraction but *understanding* —
  accountant-style multi-step calc (sum → +tax), MRZ field-parse (surname only), table-extremes
  *difference*, next-action/affordance, conversation comprehension/turn-count. These live in the
  model-free understanding layer (`derive`) + the harder per-case QAs; toggle via `emit_understanding`.

**Memorization vs understanding (A0, the prerequisite).** Because the generator is *infinite*, we
first verify learning generalises: train at increasing scale and evaluate on a **held-out split
generated with a different seed** (`run_ablation.py --heldout-seed`). Understanding ⇒ held-out keeps
improving with scale and the train/held-out gap stays small; memorization ⇒ train → ~1.0 while
held-out plateaus. A0's outcome trades the two axes (more *diversity* vs more *count*) and fixes the
data scale used by the downstream ablations ([`ablation_plan.md`](ablation_plan.md)).

## 1. Why synthetic, and how we match the real distribution

The generator's premise (`src/docvlm_eval/synth/`): **render the document from a single source so
the image and its labels can never drift.** We author HTML/CSS, render it to a **digital-native
PDF** (WeasyPrint), rasterise the selected page composition (PyMuPDF), and read text positions
**straight out of every rendered PDF page**.
If its text layer misses a required CJK, RTL, letter-spaced, or wrapped span, an optional
layout-neutral color-probe render recovers the occurrence-aware pixel box. Faker (seeded) fills
realistic field content.

Matching the real-world *and* image distribution is done on three axes, all config-controlled:

| Real-world axis | How we match it | Knob |
| --- | --- | --- |
| **Acquisition modality** (the taxonomy's PDF-native / scan / phone-photo / screenshot split) | evidence-safe authored marks, deterministic same-frame perspective for photo-style cases, then a photometric Augraphy preset: `scan`/`photo`/`fax`/`historical`/`screenshot` | `overlay_*`, `perspective_*`, `degrade_prob`, `degrade_presets`, `degrade_severity` |
| **Resolution / capture optics** | rasterise at a chosen DPI; optionally resize the longest side and (de)preserve aspect to mimic a model's preprocessor; record a small-text slice | `dpi`, `target_long_side`, `keep_aspect`, `tiling_n_max`, `small_text_px` |
| **Language / script mix** | choose each document's language from a weighted mix; generate content in that Faker locale; record language + writing system | `languages`, `language_weights` |
| **Spatial-label resolver** | PDF text positions first; model-free color-probe render only for required misses | `color_probe_fallback` |
| **Spatial-label visibility** | final-resolution box geometry and local-background pixel contrast | `validate_evidence_pixels`, `evidence_min_*` |

Boxes first come from the digital-native render. After A7 resizing, an eligible photo-style sample
may transform the raster, spotting boxes, grounding answers, rationale coordinates, and evidence
boxes through one exact homography. Standard xyxy consumers receive clipped axis-aligned envelopes
of the transformed quadrilaterals. The resulting warped raster is the clean coordinate frame;
Augraphy then changes pixels without changing that frame, so the degraded copy reuses its geometry.
Reuse is not assumed sufficient: every
candidate must retain visible local evidence and a minimum clean/degraded padded-crop structure
correlation before it is written. Spotting-off ablations run the same checks against a private
pre-ablation view, but serialize only a coordinate-free quality summary.

## 2. The ground-truth DTO (`docvlm_eval.synth.dto`)

`DocSample` is the typed record serialised to each `gt.json`. Its sub-objects exist specifically so
the ablation factors are *stored as data*, not implied:

```
DocSample
├─ doc_id, doc_type, domain, acquisition, stressors, anchor_metric, languages, source
├─ fields:  [ Field(key, value, role, bbox, language, script, font_px, is_small, reading_index) ]
├─ qa:      [ QAItem(question, answers, answer_type, metric, rationale, answer_bbox,
│                    languages, graph_query_id) ]
├─ table_html, selection, redacted, reading_order, probes
├─ semantic_graph: executable facts, relations, queries, answers, and semantic fingerprints
├─ difficulty: level, reasoning hops, distractors, density, cross-region flag, skills
├─ split: synthetic|train|validation|heldout
├─ render:  RenderSpec(source, dpi, size_px, page_size, page_count,
│                      rendered_page_count, page_mode, page_gap_px,
│                      page_origins_px, page_sizes_px,
│                      page_document_indices, page_document_ids,
│                      document_count, document_mode, document_gap_px,
│                      document_ids, document_origins_px, document_sizes_px, documents,
│                      target_long_side, keep_aspect, tiling, aspect_ratio,
│                      layout_family, layout_fingerprint, box_resolver,
│                      overlay_seed, overlay_fingerprint, overlays,
│                      color_probe_fallback_count, geometry, evidence_quality)
├─ degradation:  Degradation(preset, severity, seed, attempts, geometry_preserved,
│                            evidence_quality)
├─ gen_config:   the GenConfig that produced this sample (provenance)
└─ ablation_support: AblationSupport(spotting, rationale, multilingual, small_text,
                                     table, abstain, reading_order)   # computed from content
```

`to_dict()` is a **backward-compatible superset**: it writes the structured view
(`fields_detailed`, `qa_detailed`, `render`, `ablation_support`, …) **and** mirrors the legacy flat
keys (`type`, `fields`, `spotting`, `qa`, `table_html`, …) that `to_samples.py` and the eval
pipeline already read. Existing loaders keep working unchanged.

`ablation_support` lets a sampler filter the corpus to documents that *can* exercise a given factor
(e.g. "only docs with a rationale" for an A2 arm), so arms stay balanced.

Hard queries may cite more than one region. `QAItem.evidence_keys` links a graph query to rendered
fields or cells; after PDF box resolution, `evidence_bboxes` carries every exact pixel box into the
structured SFT/RLVR target. See
[`hard_synthetic_pipeline.md`](hard_synthetic_pipeline.md) for the executable graph and split gates.

Hard-document variants are emitted in adjacent factual/edited pairs when
`emit_counterfactual_pairs` is enabled. A pair shares its document family, locale, graph program,
and visual-theme seed while its latent values are independently regenerated. The benchmark loader
retains a reasoning pair only when both roles exist and their gold answers differ. It records the
stable pair plus graph-query ID in `counterfactual_group`, allowing evaluation to test value
sensitivity instead of template memorization. Counts above one must be even so no authored pair is
silently orphaned.

Hard families also select one of three structural layouts from `hard_layout_families`. The layout
draw uses pair-level deterministic provenance, so factual/edited members share visual structure
without coupling the content RNG. `template_fingerprint` tracks the semantic operation topology;
`RenderSpec.layout_fingerprint` tracks visual structure. This permits `split_group_by: layout` and
the validator's strict `--require-layout-isolation` gate without conflating pixels with semantics.

Authored handwriting, stamp, and seal marks are stored under `RenderSpec.overlays` with mark type,
text, final box, angle, and opacity. Each mark also emits an answer-box-linked recognition QA.
The legacy QA mirror preserves that box, so `case_to_samples()` carries it into `Sample.meta`
instead of silently reducing the task to answer-only supervision. `overlay_seed` and
`overlay_fingerprint` support exact resume and corpus auditing.

Multi-page PDF records distinguish the source `page_count` from `rendered_page_count`. Vertical and
grid modes store each page's canvas origin and pixel size, so text-search boxes, overlays, resize,
and evidence-page attribution share one coordinate contract. Cross-page samples carry the sorted
`evidence_pages` list and a `cross_page_evidence` flag into training and evaluation metadata.

Cross-document bundles add a second provenance level without changing the one-image student
contract. Each independently rendered source receives a stable document ID and exact canvas
origin/size; source keys are namespaced as `<document_id>.<field>`. Flattened page arrays retain
page-aware compatibility, while `page_document_indices`, `page_document_ids`, and `documents`
recover source identity. Cross-document samples carry sorted `evidence_documents` and a
`cross_document_evidence` flag.

Every hard document also contains a locale-matched absent-field question. The converted sample
sets `abstain_expected: true`, includes the localized absence form among valid answers, and feeds
the same locale forms to the calibrated-abstention reward. This makes the hallucination slice part
of the generated held-out benchmark rather than an optional hand-authored add-on.

## 2b. The model-free understanding layer (the part that isn't OCR)

The OCR ground truth (what string is where) falls out of the render for free. The harder, more
valuable GT is **non-OCR understanding** — *where is this word / table?*, *how many times does it
appear?*, *what is the total?* — and the **reasoning** behind each. `docvlm_eval.synth.derive`
produces all of it **with no external model**, from PDF text positions or their color-probe
fallback + the values the generator already knows. Each derivation is one `DocBuilder.ask_*` call,
resolved against the open render at build time, and emits a QA with answer **and** rationale:

| Primitive | GT type | Question it answers | Derived from | Metric |
| --- | --- | --- | --- | --- |
| `ask_where(text)` | `L1-locate` | *Where is this word?* | the text's exact box | grounding (IoU) |
| `ask_region(label, texts)` | `L1-region` | *Where is the table/region?* | union of the member strings' boxes | grounding (IoU) |
| `ask_count(text)` | `H-count` | *How many times does X appear?* | count of the word's hits | exact |
| `ask_aggregate(label, values, op)` | `H1-aggregate` | *What is the total/…?* | arithmetic over known values | relaxed-acc |

Because it is geometry + arithmetic only, every answer is **gold by construction**; derivers
**validate** (warn + skip when a requested word is absent, so GT is never silently wrong) and the
whole layer is one config switch (`emit_understanding`, ablation `U_understanding_on/off`). Under an
A7 resize, derived boxes (and the coordinates inside their rationales) are rescaled with the image so
they stay exact. An optional perspective transform then updates the same coordinate-bearing views
with its homography before DTO conversion and quality auditing. Browse it interactively in
[`notebooks/synthetic_data_design.ipynb`](../../notebooks/synthetic_data_design.ipynb), which shows
each case's GT image with box overlays and the derived *question → answer → reasoning*.

## 3. Ablation factor → DTO/config mapping

| Ablation | What varies | Where it lives in the GT | Config knob |
| --- | --- | --- | --- |
| **A1 spotting** | answer-only vs `value + [x1,y1,x2,y2]` | `Field.bbox`, `QAItem.answer_bbox` | `emit_spotting` |
| **A2 reasoning** | answer vs `rationale → answer` | `QAItem.rationale` | `emit_rationale` |
| **A3 spot+reason** | the four corners of {spot}×{reason} | both of the above | both flags (see `A3_*` overrides) |
| **A4 multilingual** | single vs mixed languages; which pairs transfer | `Field.language`/`script`, `DocSample.languages` | `languages`, `language_weights` |
| **A7 preprocessing** | resolution / tiling / aspect; small-text legibility | `RenderSpec.*`, `Field.is_small` | `dpi`, `target_long_side`, `keep_aspect`, `tiling_n_max`, `small_text_px` |
| **Hard curriculum** | lookup → aggregation → cross-region/multi-path | `semantic_graph`, `difficulty` | `difficulty_level` |
| **Counterfactual reliability** | factual/edited latent values + absent field | `counterfactual`, `graph_query_id`, probe metadata | `emit_counterfactual_pairs` |
| **Hard-layout diversity** | classic vs compact vs report structure | `RenderSpec.layout_family`, `layout_fingerprint` | `hard_layout_families` |
| **Document-mark robustness** | none vs handwriting/stamp/seal mixtures | `RenderSpec.overlays`, grounded mark QA | `overlay_prob`, `overlay_types`, `overlay_max_count` |
| **Multi-page composition** | vertical strip vs compute-aware page grid | page origins/sizes, `evidence_pages`, `page_count` | `multipage_mode` |
| **Multi-document composition** | independent-source strip vs document grid | document IDs/origins/sizes, `evidence_documents`, `document_count` | `multidocument_mode` |

`rendered_page_count` and `document_count` are also persisted as typed UDD columns. The native
student maps them to `single_page`, `multi_page`, and `cross_document` tiers for the exact
task-preserving schedule in
[`student_composition_curriculum.md`](student_composition_curriculum.md).
| **Box resolver robustness** | native PDF lookup vs native plus fallback | `RenderSpec.box_resolver`, fallback count | `color_probe_fallback` |
| **Evidence quality gate** | required-key coverage, geometry, and raster visibility | `render.evidence_quality` | `validate_evidence_pixels`, `evidence_min_*` |
| **Degradation retention gate** | degraded visibility plus clean/degraded crop structure | `degradation.evidence_quality` | `validate_degraded_evidence`, `degraded_min_structure_correlation`, `degrade_max_attempts` |

(*A5 LoRA-placement* and *A6 HPO* are training-side — they consume this GT but need no generator
knob.)

## 4. One config = one dataset variant

`configs/synth_data.yaml` has a `base:` block (the defaults) and an `ablation_overrides:` block (the
per-experiment deltas, deep-merged onto `base`). `GenConfig.from_yaml(path, ablation=…)` returns the
fully-resolved config, so each arm changes exactly one factor family vs. its control:

```bash
# baseline dataset
python scripts/make_realistic_cases.py --config configs/synth_data.yaml

# A1: spotting ON arm vs OFF control (only Field.bbox / answer_bbox differ)
python scripts/make_realistic_cases.py --config configs/synth_data.yaml --ablation A1_spotting_on
python scripts/make_realistic_cases.py --config configs/synth_data.yaml --ablation A1_spotting_off

# A4: language mix (real per-language content via the locale); A7: high-res + dynamic tiling
python scripts/make_realistic_cases.py --config configs/synth_data.yaml --ablation A4_ko_en
python scripts/make_realistic_cases.py --config configs/synth_data.yaml --ablation A7_dynamic_tiling
```

`--count N` fans each case out into `…/<NNNN>/` with reseeded content. Counterfactual pair members
share one language draw; independent pairs realize the configured corpus mix. The resolved config is also written to
`gen_config.json` next to the data for provenance, and embedded per-sample under `gen_config`.
The production base uses a weighted English/Spanish/Korean/Japanese/Chinese mix; A4 overrides clear
that weight map and sample uniformly from only the languages named by each arm.

## 5. Guarantees

- **No label drift:** every value is declared once and the box is read from the render.
- **No invisible spatial labels:** requested boxes must resolve, remain inside the final raster, and
  contain sufficient foreground contrast; the full audit is persisted under
  `render.evidence_quality`.
- **No false multilingual labels:** hard-document render text, questions, text answers, rationales,
  fields, and graph locale must agree; unsupported locale projections and missing CJK fonts fail.
- **Boxes survive degradation and resize:** the A7 resize rescales all boxes by the same factor.
  Degradation must preserve image geometry, local evidence visibility, and padded-crop structure;
  runtime or quality failures use a bounded deterministic retry sequence and then fail closed.
- **One factor at a time:** an ablation override touches only its knob family; everything else
  inherits from `base`, so a measured Δ is attributable (the staircase in `ablation_plan.md`).
- **Reproducible:** seeded Faker + recorded `gen_config` → byte-stable regeneration.
