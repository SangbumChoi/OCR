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
| `investment_dossier` | audited filing, exchange snapshot, and external analyst memo | weighted valuation, percent change, discrepancy checks, decision lookup | three-source claim verification and next action |
| `hard_diagram` | directed parallel-assay workflow with exact edge labels | lookup, topology, path product, sum of paths, weighted expected count | six- and seven-box process reasoning |

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

## Multi-page composition

`RenderResult` no longer has to silently reduce a PDF to page 0. Its explicit `page_mode` is
`first`, `vertical`, or `grid`. The all-page modes rasterize every PDF page, compose them over a
gray gutter, offset every PDF text-search box into the shared canvas, and concatenate full-text
targets in page order. `RenderSpec` records PDF and rendered page counts, page origins, page sizes,
gutter width, and composition mode. Resize scales those values with every supervision box.

The `audit_packet` family is a three-page procurement program: purchase order, receiving
inspection, and payment authorization. Its reconciliation target requires six evidence boxes
across all three pages; quantity and consistency targets span pages 1–2 and 1–3. The page-aware
sample bridge records `evidence_pages`, `cross_page_evidence`, and `page_count`, and native-student
evaluation exposes `page_count` as a matched robustness axis.

`multipage_mode` controls `vertical` versus `grid`; `D_multipage_vertical/grid` and
`--multipage-mode` provide matched controls. Grid is the small-model default. For three A5 pages
at a 896px long-side budget, it raises effective page width from about 209px to 313px versus one
vertical strip, while retaining exact page identity and ordering.

## Cross-document composition

`BundleDocument` represents an independently rendered source with its own image, GT, and stable
ID. `compose_document_bundle()` packs those sources into a `grid` or `vertical` canvas and
namespaces every field and spotting key as `<document_id>.<key>`. It preserves both levels of
provenance: flattened page origins remain compatible with page-aware consumers, while document
IDs, origins, sizes, page ownership, and per-source records distinguish independent evidence.
Resize and perspective transform both region levels with the authored boxes.

The `investment_dossier` family composes three independent sources:

1. an audited filing with revenue, cash, debt, and diluted shares;
2. an exchange snapshot with dated share price;
3. an unaudited analyst memo containing deliberately inconsistent valuation and growth claims.

Its executable graph recomputes enterprise value as `shares x price + debt - cash`, audited revenue
growth, valuation overstatement, and growth overstatement. A fifth query converts those verified
discrepancies into an exact next action: escalate the memo for review. Four of five reasoning
queries cross document boundaries; the most difficult two cite all three sources. An absent 2026
guidance probe remains the abstention control.

`multidocument_mode`, `D_multidocument_vertical/grid`, and `--multidocument-mode` provide matched
packing controls. The sample bridge records `evidence_documents`, `cross_document_evidence`, and
`document_count`; native evaluation exposes `document_count` as a seventh canonical robustness
axis. This preserves the production student's one-image API rather than relying on an unimplemented
multi-image path.

## Programmatic diagrams

`hard_diagram` is drawn directly with PIL rather than rasterizing a decorative figure. Every stage
label and edge-rate label has an authored pixel box, and every arrow weight is also an executable
`GraphEdge`. The graph represents sample intake, a quality gate, two routed assay branches, fusion
review, and release decision. Routing fractions sum to one; branch retention and final release
rates are independently sampled.

The five-level curriculum is structural:

| Level | Added supervision |
| --- | --- |
| 1 | direct edge-rate reading |
| 2 | identify the stage where two branches merge |
| 3 | multiply the complete Assay A path |
| 4 | sum both path products and compare branch yields |
| 5 | combine the submitted batch size with total release yield |

`classic-v1` is left-to-right, `compact-v1` is top-to-bottom, and `report-v1` uses a mixed report
composition. All three retain identical semantic and template fingerprints for the same RNG state
while changing image size, node coordinates, and layout fingerprint. The family is English-only
until a diagram-specific locale catalog is authored; its metadata is fixed to `en` rather than
mislabeling English pixels as another language. Existing difficulty, layout, spotting, rationale,
overlay, degradation, resize, split, and evidence-quality controls apply without a new special
ablation.

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
  --only hard_table hard_chart hard_investment hard_science hard_diagram \
  --difficulty-level 1 --split-name train --seed 7 --count 100 \
  --out data/generated/hard_train_l1

python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science hard_diagram \
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

A 12-cell family-by-layout render covered the four original hard families in all three layouts at
96 DPI. Every
render stayed on one page, produced a distinct raster within its family, and passed required-box
visibility auditing while retaining identical content and template fingerprints across layouts.
A separate 24-document degraded CLI smoke passed all clean/degraded gates; each adjacent
counterfactual pair shared its layout. A high-value compact-chart regression also verifies that
normalized bars and labels remain inside the landscape page.

A forced-overlay smoke covered the 19 families present at that calibration point and produced all
three mark types while
every clean evidence audit passed. A second 24-document hard-family smoke combined overlays,
degradation, and sampled perspective: all clean and degraded gates passed, every pair shared its
mark-type signature, and the mix contained 14 seals, 12 handwritten notes, and 6 stamps. Across
256 base seeds and all 19 families, the default 0.35 probability selected 1,691 of 4,864 decisions
(34.77%); conditional type counts were balanced at 831 stamps, 857 handwritten notes, and 824 seals.

A three-variant grid smoke produced 1,768x2,500 three-page packets with degradation and document
marks. Every clean/degraded audit passed; the minimum observed structure correlation was 0.817.
All reconciliation and consistency samples carried evidence pages `[0,1,2]`, and quantity samples
carried `[0,1]`. A vertical 875x3,759 control and a resized 543x768 grid render also passed exact
box and page-provenance checks.

A real three-source investment-dossier render produced a 1,768x2,500 document grid with eleven
visible field boxes and 66 deduplicated field/evidence references. Its valuation sample cited
documents `[0,1]`; valuation discrepancy and review-action samples cited `[0,1,2]`; the local
audited-growth control cited `[0]`. Three forced-overlay degraded variants passed every clean and
degraded gate on the first attempt, with minimum structure correlation 0.646. A 178x768 vertical
low-resolution control also preserved all document origins, evidence boxes, and quality checks,
while exposing the expected severe text-resolution cost of strip packing. A subsequent
forced-overlay clean smoke covered all 21 current case families and every evidence gate passed.

A three-layout level-5 diagram smoke produced distinct 1,200x760, 900x900, and 1,200x900 rasters
with identical semantic content fingerprints. Visual inspection caught and corrected a compact
header/node overlap. Three forced-overlay degraded variants passed every clean/degraded gate on
their first attempt; minimum structure correlation was 0.819. A representative graph emitted six
executable reasoning queries with evidence counts 1, 1, 4, 6, 6, and 7, and all thirteen required
stage/edge/audit spotting keys passed the clean raster gate.
