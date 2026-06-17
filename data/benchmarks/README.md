# Benchmark catalog & sample previews

Every benchmark across the **10 capability categories** of
[`../../report/benchmark_taxonomy.md`](../../report/benchmark_taxonomy.md), annotated with **what each
one measures** (`purpose`). Source of truth: [`../../configs/benchmark_catalog.yaml`](../../configs/benchmark_catalog.yaml).

- 🖼️ **sample** = image + `sample.json` (GT + metric + purpose) in `<key>/` — fetched via
  `scripts/fetch_benchmark_samples.py` (HF streaming) or attached via `scripts/make_synthetic_samples.py`.
- 📝 **documented** = not cleanly streamable from HF (script-based / gated / eval-server); catalogued with purpose + source.

**Coverage: 24 image samples across 42 catalogued benchmarks, all 10 categories.**

### 1. Text recognition (full-page / line / word)

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`iam`](iam/) | 🖼️ sample (HF) | CER / WER / NED | Recognize handwritten English text lines; the classic handwriting-recognition fidelity test. |
| [`recognition_fullpage`](recognition_fullpage/) | 🖼️ sample (synthetic) | CER / WER / NED | Transcribe a full printed page exactly; measures end-to-end OCR fidelity / typo rate. |
| [`iiit5k`](iiit5k/) | 📝 documented | word accuracy / 1-NED | Cropped-word recognition; isolates pure recognition from detection. |

### 2. Scene-text detection & recognition

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`scenetext`](scenetext/) | 🖼️ sample (synthetic) | H-mean / word-acc / 1-NED | Detect + read text 'in the wild' (signs, products); irregular fonts/orientations. |
| [`icdar2015`](icdar2015/) | 📝 documented | detection H-mean | Detect/recognize incidental scene text under blur/perspective; standard detection F-measure. |
| [`totaltext`](totaltext/) | 📝 documented | detection H-mean | Curved/multi-oriented scene text; stresses non-horizontal layouts. |

### 3. Document / scene-text / diagram VQA

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`docvqa`](docvqa/) | 🖼️ sample (HF) | anls | Answer questions about scanned business documents; reading + layout + key-value extraction. |
| [`infovqa`](infovqa/) | 🖼️ sample (HF) | anls | QA over infographics; joint layout + graphics + text + basic arithmetic reasoning. |
| [`textvqa`](textvqa/) | 🖼️ sample (HF) | exact | Answer questions that require reading text in natural images. |
| [`stvqa`](stvqa/) | 🖼️ sample (HF) | anls | VQA requiring reading scene text; origin of the ANLS metric. |
| [`ocrvqa`](ocrvqa/) | 🖼️ sample (HF) | exact | QA over book-cover images; reading titles/authors/edition from text-heavy images. |
| [`ai2d`](ai2d/) | 🖼️ sample (HF) | exact | Multiple-choice QA over grade-school science diagrams; arrows/labels/structure reasoning. |
| [`visualmrc`](visualmrc/) | 📝 documented | BLEU/METEOR/CIDEr | Long abstractive answers over webpage screenshots; reading comprehension + generation. |

### 4. Key Information Extraction (KIE)

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`funsd`](funsd/) | 🖼️ sample (HF) | entity F1 | Extract entities (question/answer/header) AND key->value links from noisy scanned forms. |
| [`cord`](cord/) | 🖼️ sample (HF) | entity F1 | Extract ~30 fine-grained fields + hierarchy from receipts. |
| [`sroie`](sroie/) | 🖼️ sample (HF) | field F1 | Extract 4 fields (company/date/address/total) from receipts; exact field match. |
| [`docile`](docile/) | 📝 documented | F1 / AP | Localize+extract key info (KILE) and line items (LIR) from business documents; 55 field types. |
| [`xfund`](xfund/) | 📝 documented | entity / relation F1 | Multilingual (7 languages) form understanding; entity + relation extraction. |
| [`wildreceipt`](wildreceipt/) | 📝 documented | entity F1 | 26-key receipt KIE in realistic photos; harder capture conditions than SROIE. |

### 5. Table recognition & structure

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`pubtabnet`](pubtabnet/) | 🖼️ sample (HF) | TEDS / TEDS-S | Reconstruct table structure + cell content (HTML) from images; cell-relationship fidelity. |
| [`pubtables1m`](pubtables1m/) | 📄 label only | GriTS / mAP | Large-scale table detection + structure recognition; GriTS over cell grids (topology). |
| [`fintabnet`](fintabnet/) | 📝 documented | TEDS | Complex financial tables (merged cells, dense numerics); structure + content extraction. |
| [`scitsr`](scitsr/) | 📝 documented | cell-adjacency F1 | Table structure from scientific papers; neighbour-relationship correctness. |

### 6. Chart / plot / figure reasoning

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`chartqa`](chartqa/) | 🖼️ sample (HF) | relaxed_acc | Read + reason (compare/arithmetic) over bar/line/pie charts; relaxed numeric accuracy. |
| [`mathvista`](mathvista/) | 🖼️ sample (HF) | exact | Visual mathematical reasoning over figures/plots/diagrams; multi-step quantitative reasoning. |
| [`plotqa`](plotqa/) | 📝 documented | relaxed_acc | QA over scientific plots with out-of-vocabulary numeric answers; data extraction + reasoning. |
| [`dvqa`](dvqa/) | 📝 documented | exact | Bar-chart QA testing structure/element reading and value comparison. |

### 7. Formula / math-expression recognition

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`im2latex`](im2latex/) | 🖼️ sample (HF) | edit dist / BLEU | Image -> LaTeX for printed formulas; token accuracy / normalized edit distance. |
| [`latexocr`](latexocr/) | 🖼️ sample (HF) | edit dist / BLEU / exact | Image -> LaTeX rendering; exact-match + edit-distance on the token sequence. |
| [`crohme`](crohme/) | 📝 documented | expression rec. rate | Handwritten math expression recognition; expression-level + symbol-level accuracy. |

### 8. Comprehensive LMM OCR & figure suites

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`ocrbench`](ocrbench/) | 🖼️ sample (HF) | ocrbench | 1000-item OCR capability probe across 5 sub-skills (recog / scene-VQA / doc-VQA / KIE / HME). |
| [`ocrbench_v2`](ocrbench_v2/) | 🖼️ sample (HF) | per-task acc | Extended OCR suite (~10k items, 31 scenarios) incl. fine-grained spotting & structure. |
| [`charxiv`](charxiv/) | 🖼️ sample (HF) | descriptive/reasoning acc | Realistic scientific-figure understanding (arXiv charts); descriptive + reasoning questions. |
| [`seedbench2plus`](seedbench2plus/) | 📝 documented | accuracy | MCQ over text-rich images (charts/maps/webs); broad text-comprehension coverage. |

### 9. End-to-end page parsing & layout

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`omnidocbench`](omnidocbench/) | 🖼️ sample (HF) | edit dist / TEDS / CDM | Whole-page -> structured output across text/formula/table/reading-order; per-element scoring. |
| [`doclaynet`](doclaynet/) | 📝 documented | COCO mAP | Document layout detection (11 classes) across 6 doc types; element localization. |
| [`fox`](fox/) | 📝 documented | edit dist / F1 | Region/colour-guided OCR + cross-page understanding on dense multi-page docs. |
| [`readoc`](readoc/) | 📝 documented | edit-sim / TEDS | Holistic PDF -> semantic Markdown (text/headings/tables/reading-order). |

### 10. Reliability: robustness / calibration / hallucination

| Benchmark | Status | Metric | Purpose (what it measures) |
|---|---|---|---|
| [`robustness`](robustness/) | 🖼️ sample (synthetic) | ANLS retention + ECE | Stability under capture degradations + terminology paraphrase; confidence calibration. |
| [`pope`](pope/) | 🖼️ sample (HF) | F1 / accuracy | Object-existence polling (yes/no) to quantify hallucination; precision/recall/F1. |
| [`hallusionbench`](hallusionbench/) | 🖼️ sample (HF) | acc + consistency | Visual-illusion vs knowledge-hallucination yes/no pairs; consistency-aware scoring. |
| [`kie_hvqa`](kie_hvqa/) | 📝 documented | hallucination-free acc | Penalize confident wrong field values on degraded ID/invoice docs; reward correct-or-abstain. |

