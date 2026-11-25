# Survey — open-source synthetic-data generators (textual & spatial diversity)

**Purpose.** Deepen our dataset-generation methodology by surveying four open-source synthetic-data
projects and extracting techniques to enrich the **textual** and **spatial** diversity of our
generator (`scripts/make_realistic_cases.py` + `src/docvlm_eval/synth/`).
**Hard constraint for this task:** we use **simulation-based generation only** (no LLM gateway at
generation time). Projects that mandate an LLM are catalogued and architected as *future-optional*
plug-ins (see §4), never as a runtime dependency.

Related: [`prd_synthetic_diversity.md`](prd_synthetic_diversity.md) ·
[`synthetic_data_dto.md`](synthetic_data_dto.md) · [`ablation_plan.md`](ablation_plan.md).

## 1. The four projects at a glance

| Project | Output | LLM gateway? | Stack | License |
| --- | --- | --- | --- | --- |
| [microsoft/genalog](https://github.com/microsoft/genalog) | scanned-document **images** from HTML | **No** | WeasyPrint (Pango/Cairo) + font randomisation + degradation; optional Azure OCR | MIT |
| [Travvy88/DocumentGenerator_DoGe](https://github.com/Travvy88/DocumentGenerator_DoGe) | document **images + word boxes** | **No** | Wikipedia text → python-docx → (unoserver) PDF → pdf2image; OpenCV box detection; Augraphy | (unspecified) |
| [argilla-io/synthetic-data-generator](https://github.com/argilla-io/synthetic-data-generator) | text-classification / SFT-chat / RAG **text** datasets | **Yes (mandatory)** | distilabel + HF Inference / OpenAI / Ollama / vLLM; Argilla curation | Apache-2.0 |
| [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit) | QA / CoT / summary / multimodal-QA for fine-tuning | **Yes (mandatory)** for `create`/`curate` | vLLM or API backend; Lance ingest; judge-based curation | MIT |

**Takeaway.** The two **image** generators (genalog, DoGe) are simulation-based and map directly onto
our HTML→PDF→raster→degrade pipeline. The two **text** generators (Argilla, Meta) are LLM-first and
are out of scope for runtime use here; we keep their *ideas* (iterate-on-samples, judge-based
quality curation) as a future-optional layer.

## 2. Simulation-based techniques we adopt now (no LLM)

### From genalog (image realism via templating + degradation)
| Technique | genalog | Our generator |
| --- | --- | --- |
| HTML/CSS templates → WeasyPrint render | core | already core (`synth/render.py`, `patterns.DocBuilder`) |
| **Font-family randomisation** | primary diversity lever | **adopted/expanded** — `_BODY_FONTS` broadened to sans/serif/mono families (was 4 → 8) |
| Synthetic scan degradation (noise/blur/bleed) | core | already core via **Augraphy** presets (`scan/photo/fax/historical/screenshot`) |
| Text supplied externally (no LLM) | yes | yes — content from Faker/locale + curated pools |

### From DoGe (per-document layout parameters + model-free boxes)
| Technique | DoGe | Our generator |
| --- | --- | --- |
| **Per-doc layout params** (font-size, alignment, line-spacing, columns) | `docx_config.json` | **adopted/expanded** — `_theme_css` now jitters `font-size`, **`line-height`**, **`letter-spacing`**, page-margin, heading alignment per doc (spatial spread) |
| **Color-coded word → raster-mask bbox** (model-free spotting GT) | core trick | **implemented as a fallback resolver** for PDF text-search misses or fragmented spans; see §3 |
| Real-corpus text (Wikipedia) for textual variety | crawl | analog: locale-aware Faker + curated content pools (offline, deterministic); real-corpus is a future-optional `TextSource` (§4) |
| Augraphy final augmentation | yes | yes |
| Multi-stage render (docx→pdf→png) | yes | we render HTML→pdf→png directly (one fewer lossy hop, keeps exact boxes) |

These two adoptions are implemented in this pass: broadened font pool and DoGe-style per-document
layout jitter (`_theme_css` structural branch). Effect is measured by
`scripts/measure_diversity.py` — the *same-template visual-similarity* descriptor (the PRD v2
layout-variety lever) is the target signal.

## 3. Model-free bounding boxes: our approach vs DoGe's color-coding

We currently derive boxes by **searching the rendered PDF's text layer** (`render.search_boxes` →
PyMuPDF `search_for`), which is exact and cheap for Latin text. DoGe instead **renders each word in a
unique RGB colour, then recovers boxes with OpenCV connected components** — this is OCR-free *and*
text-search-free, so it is robust where `search_for` fails:

- wrapped / letter-spaced strings (the ID **MRZ** we just fixed),
- complex scripts where the PDF text layer is reordered (Arabic RTL, CJK vertical),
- decorative fonts whose glyphs don't round-trip to searchable text.

**Implemented fallback:** `DocBuilder.build()` first queries the PDF text layer. When it returns
fewer occurrences than a spotting or spatial-derivation request requires, the builder wraps the
visible target text nodes in layout-neutral, uniquely colored spans and renders one probe copy.
The exact raster-color mask recovers one occurrence-aware union box per span. Overlapping target
strings that cannot share a batch receive an isolated probe pass.

The fallback is simulation-only and model-free. It is controlled by
`GenConfig.color_probe_fallback`, shared by spotting plus locate/count/region derivations, and only
runs for native lookup misses. `RenderSpec.box_resolver` and
`color_probe_fallback_count` make the resolver contract and actual use auditable per sample.

## 4. LLM-first projects → future-optional architecture (NOT a runtime dependency)

Argilla's `synthetic-data-generator` and Meta's `synthetic-data-kit` both **require** an LLM
backend. We do **not** depend on them. To keep them usable *later* without contaminating the
simulation-only path, we define seams:

- **`TextSource` seam (textual diversity).** Today every case draws text from an offline source
  (Faker + curated pools). A `text_source` config knob (default `"offline"`) reserves space for
  future `"corpus"` (DoGe-style real-text, e.g. a local Wikipedia dump — offline, no gateway) and
  `"llm"` (Argilla/Meta-style generation behind an explicit, opt-in gateway). The non-offline modes
  raise a clear `NotImplementedError` pointing here until a user explicitly enables a backend, so the
  default build never touches a network/LLM.
- **Judge-based curation seam (quality).** Meta's `curate` step filters generated samples with an LLM
  judge. Our analog that needs **no LLM** is already shipping: `scripts/measure_diversity.py`
  (content-aware true-duplicate rate, reasoning share, visual CoV) is a *heuristic curator*. A future
  `"llm"` judge would be an optional second pass, never replacing the heuristic gate.
- **Iterate-on-samples loop.** Argilla's UX (preview a few samples → adjust → scale) maps onto our
  `--count 1` preview → `measure_diversity` → scale workflow; no code dependency required.

**Acceptance for "future usage":** the LLM seams exist as documented config values + explicit
opt-in errors, so enabling them later is a localized change, while the present generator is provably
LLM-free (default `text_source="offline"`, no network at generation time).

## 5. What changed in this pass (simulation-only)
1. `_BODY_FONTS` broadened (genalog font-variety lever): 4 → 8 sans/serif/mono families.
2. `_theme_css` structural jitter expanded (DoGe per-doc layout params): `line-height`,
   `letter-spacing`, wider `font-size` and page-`margin` ranges → larger spatial spread.
3. Occurrence-aware color-probe box recovery for PDF text-layer misses, with an explicit control
   arm and per-sample provenance.
4. This survey + the future-optional `TextSource`/judge seams documented (no LLM wired).

Measured effect and acceptance: see [`../results/synthetic_diversity_report.md`](../results/synthetic_diversity_report.md).

## 6. Backlog (still simulation-based, future passes)
- Local real-text corpus `TextSource="corpus"` (offline Wikipedia/Gutenberg dump) for richer prose.
- Per-case layout **templates** (2–3 skeletons each) — the PRD v2 lever for the residual
  same-template similarity that photometric/typographic jitter alone cannot move.
