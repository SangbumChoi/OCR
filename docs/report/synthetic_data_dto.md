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
PDF** (WeasyPrint), rasterise page 0 (PyMuPDF), and read text positions **straight out of the PDF**
— so spotting boxes are pixel-exact by construction. Faker (seeded) fills realistic field content.

Matching the real-world *and* image distribution is done on three axes, all config-controlled:

| Real-world axis | How we match it | Knob |
| --- | --- | --- |
| **Acquisition modality** (the taxonomy's PDF-native / scan / phone-photo / screenshot split) | a photometric Augraphy preset applied to a *copy* (no geometry change → boxes stay valid): `scan`/`photo`/`fax`/`historical`/`screenshot` | `degrade_prob`, `degrade_presets`, `degrade_severity` |
| **Resolution / capture optics** | rasterise at a chosen DPI; optionally resize the longest side and (de)preserve aspect to mimic a model's preprocessor; record a small-text slice | `dpi`, `target_long_side`, `keep_aspect`, `tiling_n_max`, `small_text_px` |
| **Language / script mix** | choose each document's language from a weighted mix; generate content in that Faker locale; record language + writing system | `languages`, `language_weights` |

Because the boxes come from the clean digital-native render and degradation is photometric only,
**a degraded copy reuses the clean GT** — the heuristic that makes free, exact labels possible.

## 2. The ground-truth DTO (`docvlm_eval.synth.dto`)

`DocSample` is the typed record serialised to each `gt.json`. Its sub-objects exist specifically so
the ablation factors are *stored as data*, not implied:

```
DocSample
├─ doc_id, doc_type, domain, acquisition, stressors, anchor_metric, languages, source
├─ fields:  [ Field(key, value, role, bbox, language, script, font_px, is_small, reading_index) ]
├─ qa:      [ QAItem(question, answers, answer_type, metric, rationale, answer_bbox, languages) ]
├─ table_html, selection, redacted, reading_order, probes
├─ semantic_graph: executable facts, relations, queries, answers, and semantic fingerprints
├─ difficulty: level, reasoning hops, distractors, density, cross-region flag, skills
├─ split: synthetic|train|validation|heldout
├─ render:  RenderSpec(source, dpi, size_px, page_size, page_count,
│                      target_long_side, keep_aspect, tiling, aspect_ratio)
├─ degradation:  Degradation(preset, severity, seed, geometry_preserved)
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

## 2b. The model-free understanding layer (the part that isn't OCR)

The OCR ground truth (what string is where) falls out of the render for free. The harder, more
valuable GT is **non-OCR understanding** — *where is this word / table?*, *how many times does it
appear?*, *what is the total?* — and the **reasoning** behind each. `docvlm_eval.synth.derive`
produces all of it **with no external model**, from the rendered PDF's exact text positions
(PyMuPDF) + the values the generator already knows. Each derivation is one `DocBuilder.ask_*` call,
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
they stay exact. Browse it interactively in
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

`--count N` fans each case out into `…/<NNNN>/` with reseeded content (and a fresh language draw per
doc, so the mix is realised across the corpus). The resolved config is also written to
`gen_config.json` next to the data for provenance, and embedded per-sample under `gen_config`.
The production base uses a weighted English/Spanish/Korean/Japanese/Chinese mix; A4 overrides clear
that weight map and sample uniformly from only the languages named by each arm.

## 5. Guarantees

- **No label drift:** every value is declared once and the box is read from the render.
- **No false multilingual labels:** hard-document render text, questions, text answers, rationales,
  fields, and graph locale must agree; unsupported locale projections and missing CJK fonts fail.
- **Boxes survive degradation and resize:** degradation is photometric; the A7 resize rescales all
  boxes by the same factor (`scripts/make_realistic_cases.py::_resize_with_boxes`).
- **One factor at a time:** an ablation override touches only its knob family; everything else
  inherits from `base`, so a measured Δ is attributable (the staircase in `ablation_plan.md`).
- **Reproducible:** seeded Faker + recorded `gen_config` → byte-stable regeneration.
