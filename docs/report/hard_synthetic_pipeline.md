# Executable hard-document generation

The synthetic pipeline now has a semantic layer between random content and rendered pixels. It
targets the document skills that are difficult to supervise from public OCR data alone:
multi-cell tables, exact chart arithmetic, multi-hop investment ownership, and quantitative
scientific-paper verification.

## Contract

`docvlm_eval.synth.latent.LatentDocumentGraph` is the source of truth for hard reasoning:

1. `GraphNode` stores rendered and latent facts, including the `field_key` used to recover visual
   evidence after rendering.
2. `GraphEdge` stores typed relations and optional weights, such as direct ownership fractions.
3. `GraphQuery` declares a deterministic operation, input IDs, answer format, metric, task slice,
   and evidence nodes.
4. `resolve()` recomputes the answer and concise rationale from the graph. Validation fails on
   duplicate IDs, dangling relations, unsupported operations, invalid numeric values, or a stale
   authored `expected` answer.
5. `DocBuilder` renders the same values and resolves every evidence key through the PDF text layer,
   with an occurrence-aware color-probe fallback for required misses.
6. A final raster audit deduplicates all field, answer, and multi-box evidence coordinates after
   resize, optional photo perspective, and supervision projection. It rejects unresolved requested
   keys, clipped geometry, and crops without sufficient contrast against their local background.
7. Photo-style samples use one deterministic homography for pixels and all spatial supervision.
   Counterfactual factual/edited variants share that homography so geometry cannot reveal the role.

The operation registry currently covers direct lookup, sum, mean, difference, ratio, percent
change, relative reduction, extrema, weighted sum, path products, and sums of independent path
products. This is deliberately small and auditable: no teacher model invents hard labels.

## Multilingual projection

Every hard family has executable locale projections for English, Spanish, Korean, Japanese, and
Simplified Chinese. The latent node IDs, edge programs, numeric values, evidence keys, and canonical
machine units remain language-independent. Titles, table headers, explanatory body text, questions,
text-valued answers, concise-answer instructions, and operation rationales are projected from one
validated message catalog.

The default generation mix is weighted toward English while retaining all five languages:
`en=0.35`, `es=0.15`, `ko=0.20`, `ja=0.15`, and `zh=0.15`. A generated hard record is rejected
unless:

- every detailed field and QA carries the selected locale;
- the semantic graph records the same locale;
- every family produces a non-empty full-page text-layer target;
- Korean, Japanese, and Chinese content contains the expected script in both rendered text and
  questions;
- Spanish questions contain the locale-specific interrogative form;
- a CJK-capable font is available before generation starts.

This replaces the previous behavior that could render English hard documents and relabel only their
metadata as non-English. Template fingerprints remain language-independent because translated
display text never changes graph topology or canonical units.

The end-to-end experiment plan content-addresses `configs/synth_data.yaml`. Locale catalog or mix
changes therefore invalidate synthesis and all dependent training stages under exact resume.

Each `gt.json` records:

- the complete `semantic_graph`;
- resolved query answers, rationales, and evidence keys;
- content and template fingerprints;
- the visual layout family and layout fingerprint;
- a machine-readable `difficulty` profile;
- explicit split provenance and a deterministic suggested split.
- the spatial resolver contract and number of color-probe fallbacks used.
- optional perspective seed, destination corners, homography, area ratio, and box enclosure policy.
- the pixel-level evidence audit, thresholds, source coverage, and per-box visibility statistics.

## Hard families

| Family | Rendered evidence | Gold program | Level-5 capability |
| --- | --- | --- | --- |
| `hard_table` | dense regional operating table plus an external budget field | lookup, sums, profit, argmax, cross-region budget subtraction | eleven-box evidence for a table-to-summary calculation |
| `hard_chart` | labelled temporal bar chart | lookup, percent change, argmax, multi-year mean | exact temporal chart aggregation |
| `hard_investment` | direct beneficial-ownership schedule | relation lookup, path product, sum of path products | effective ownership across two independent paths |
| `hard_science` | paper title, abstract, equation, result table, caption | lookup, argmin, control-relative reduction, treatment comparison | quantitative claim verification against the stated equation |

The `difficulty_level` knob is an integer from 1 to 5. Level 1 emits direct visual lookup. Higher
levels add aggregation or relational paths. Level 5 enables multi-path or cross-region reasoning
and the largest distractor budget. The profile records reasoning hops, distractor count, visual
density, cross-region status, and required skills, so curriculum sampling does not infer
difficulty from task names.

## Semantic-preserving layouts

Every hard family has three structural renderings: `classic-v1`, `compact-v1`, and `report-v1`.
They change page orientation or size, section order, and grouping into columns, sidebars, or report
panels. They do not change latent values, graph operations, answers, evidence keys, or semantic
fingerprints. Layout selection uses a separate deterministic stream, so enabling layout diversity
does not perturb authored content. Adjacent factual/edited counterfactual variants deliberately
share one layout.

`semantic_graph.template_fingerprint` identifies the language-independent operation topology.
`render.layout_fingerprint` separately identifies the visual structure and is derived from the
document family plus layout family. This separation supports both value generalization under a
known visual form and strict visual-layout holdout. The enabled set is controlled by
`hard_layout_families`; `--hard-layout` forces one family for diagnosis. The
`D_hard_layout_classic` and `D_hard_layout_diverse` config arms provide matched ablations.

## Authored document marks

Handwritten margin notes, rectangular approval stamps, and circular validity seals are applied
after resize and before optional perspective. A deterministic low-ink placement search rejects
every candidate intersecting an expanded authored evidence box. Perspective then transforms both
the raster mark and its provenance box in the same homography. Each accepted mark adds a language
label, a grounded recognition QA, a stressor tag, and exact render provenance. Full-page
OCR targets append the mark text so the visible raster and transcript do not disagree.

`overlay_prob`, `overlay_types`, and `overlay_max_count` control the mixture.
`D_overlays_off/on` provide probability-zero and probability-one arms, while `--overlay-prob` and
`--overlay-type` support CLI diagnosis. Counterfactual pairs share the overlay seed and mark-type
signature. Spotting-off projection removes mark and QA boxes while retaining coordinate-free mark
provenance. Clean and degraded evidence gates audit the mark QA boxes exactly like authored text.

## Grounded post-training

Hard questions can cite multiple evidence cells. `QAItem.evidence_keys` is resolved after rendering
to `evidence_bboxes`, then `case_to_samples()` places all boxes in `Sample.meta["boxes"]`.
`RewardContext.from_sample()` normalizes every box, and `StructuredPostTrainingDataset` emits the
strict target:

```json
{"answer":"...","evidence":[[0.1,0.2,0.3,0.4]],"rationale":"..."}
```

This means answer correctness, evidence IoU, and grounded-rationale reward components operate on
the same authored graph query. Evidence is removed automatically in spotting-off ablations because
the boxes are resolved only from emitted spotting ground truth.

Control arms are fail-closed. Spotting-off removes grounding QAs, coordinate answers, and evidence
links, not just the top-level box dictionary. Rationale-off removes the executable node/edge/query
payload and retains only graph fingerprints and family provenance, so the operation program cannot
become a hidden reasoning target.

## Generation and leakage gates

```bash
# Curriculum endpoints
python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science \
  --difficulty-level 1 --split-name train --seed 7 --count 100 \
  --out data/generated/hard_train_l1

python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science \
  --difficulty-level 5 --split-name heldout --seed 7007 --count 100 \
  --out data/generated/hard_heldout_l5

# Mandatory exact-content leakage check
python scripts/validate_synth_splits.py \
  --split train=data/generated/hard_train_l1 \
  --split heldout=data/generated/hard_heldout_l5 \
  --output docs/results/hard_split_audit.json

# Strict visual-layout holdout
python scripts/validate_synth_splits.py \
  --split train=data/generated/hard_train_layouts \
  --split heldout=data/generated/hard_heldout_layout \
  --require-layout-isolation
```

Generation itself is fail-closed before any sample files are written. The clean pixel gate is configured
by `validate_evidence_pixels`, `evidence_min_contrast`,
`evidence_min_foreground_fraction`, and `evidence_min_foreground_pixels` in
`configs/synth_data.yaml`. A degraded candidate then has to preserve image size, independently
visible evidence, and a minimum clean/degraded structure correlation for every box. Augraphy
runtime failures and rejected candidates use a bounded deterministic retry sequence controlled by
`degrade_max_attempts`; the accepted seed, attempt count, and per-box retention statistics are
stored under `degradation.evidence_quality`.
Photo perspective is controlled by `perspective_prob`, `perspective_max_inset_fraction`, and
`perspective_min_area_ratio`. It is sampled only for `photo` preset/acquisition cases and only when
degraded generation is enabled. The transform runs before both pixel gates and Augraphy, and
`D_perspective_off/on` provide probability-zero and probability-one ablation arms.
Spotting-off controls are audited against a private pre-ablation box view; only the status,
thresholds, and aggregate structure statistics are serialized, so the gate cannot become a hidden
coordinate-supervision channel.

The split validator rejects the same semantic content fingerprint across splits while reporting
template overlap. `--require-template-isolation` additionally rejects the same graph program
topology across splits, which is useful for a strict template-generalization evaluation. Template
overlap is otherwise allowed so the standard heldout set can measure new values under known
programs, separately from the stricter topology holdout. `split_group_by: layout` assigns all
records with one layout fingerprint to the same split, and `--require-layout-isolation` rejects
cross-split visual-layout overlap. Missing layout provenance fails closed when that gate is enabled.

## Verified smoke path

The implementation is covered by graph, curriculum, fingerprint, split-leakage, DTO, sample bridge,
and reward-context tests. A real four-family render at difficulty 5 produced twelve hard reasoning
samples; every sample retained a non-empty rationale and between two and eleven normalized evidence
boxes in the structured post-training target. A second-seed heldout render produced eight unique
semantic contents across train and heldout, with template overlap reported separately.

A multilingual CLI smoke generated 40 clean difficulty-5 documents across all four hard families
and all five supported languages. All 40 GT/image pairs passed locale validation and had unique
content fingerprints. A direct 20-cell family-by-language render verified searchable text, exact
program answers, localized text answers, and language-independent template fingerprints.

A forced-perspective smoke covered ID, webtoon, LCD, and hard-chart photo cases. Every transformed
box remained inside its raster and all clean/degraded evidence audits passed. A hard-chart
counterfactual pair shared its seed, destination corners, and homography while retaining distinct
content fingerprints. Across 256 base seeds, the default 0.35 probability selected 335 of 1,024
eligible document decisions (32.71%); non-photo families never received geometry.

A 12-cell family-by-layout render covered every hard family in all three layouts at 96 DPI. Every
render stayed on one page, produced a distinct raster within its family, and passed required-box
visibility auditing while retaining identical content and template fingerprints across layouts.
A separate 24-document degraded CLI smoke passed all clean/degraded gates; each adjacent
counterfactual pair shared its layout. A high-value compact-chart regression also verifies that
normalized bars and labels remain inside the landscape page.

A forced-overlay smoke covered all 18 document families and produced all three mark types while
every clean evidence audit passed. A second 24-document hard-family smoke combined overlays,
degradation, and sampled perspective: all clean and degraded gates passed, every pair shared its
mark-type signature, and the mix contained 14 seals, 12 handwritten notes, and 6 stamps. Across
256 base seeds and all 18 families, the default 0.35 probability selected 1,596 of 4,608 decisions
(34.64%); conditional type counts were balanced at 788 stamps, 808 handwritten notes, and 777 seals.
