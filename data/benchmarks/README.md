# Benchmark catalog & sample previews

Every benchmark across the capability families of
[`../../docs/report/benchmark_taxonomy.md`](../../docs/report/benchmark_taxonomy.md) and
[`../../docs/report/capability_axes.md`](../../docs/report/capability_axes.md), annotated with **what
each one measures** (`purpose`). Source of truth:
[`../../configs/benchmark_catalog.yaml`](../../configs/benchmark_catalog.yaml).

- 🖼️ **sample** = image + `sample.json` (GT + metric + purpose) in `<key>/`.
- 📝 **documented** = not cleanly streamable from HF; catalogued with purpose + source.

**Coverage: 29 image samples across 46 catalogued benchmarks.**

### A. Recognition / transcription (full-page · scene-text · end-to-end parsing)

| Code | Benchmark                                       | Status                | Metric                    | Purpose (what it measures)                                                                    |
| ---- | ----------------------------------------------- | --------------------- | ------------------------- | --------------------------------------------------------------------------------------------- |
| A1   | [`iam`](iam/)                                   | 🖼️ sample (HF)        | CER / WER / NED           | Recognize handwritten English text lines; the classic handwriting-recognition fidelity test.  |
| A1   | [`recognition_fullpage`](recognition_fullpage/) | 🖼️ sample (synthetic) | CER / WER / NED           | Transcribe a full printed page exactly; measures end-to-end OCR fidelity / typo rate.         |
| A1   | [`iiit5k`](iiit5k/)                             | 📝 documented          | word accuracy / 1-NED     | Cropped-word recognition; isolates pure recognition from detection.                           |
| A2   | [`scenetext`](scenetext/)                       | 🖼️ sample (synthetic) | H-mean / word-acc / 1-NED | Detect + read text 'in the wild' (signs, products); irregular fonts/orientations.             |
| A2   | [`icdar2015`](icdar2015/)                       | 📝 documented          | detection H-mean          | Detect/recognize incidental scene text under blur/perspective; standard detection F-measure.  |
| A2   | [`totaltext`](totaltext/)                       | 📝 documented          | detection H-mean          | Curved/multi-oriented scene text; stresses non-horizontal layouts.                            |
| A3   | [`omnidocbench`](omnidocbench/)                 | 🖼️ sample (HF)        | edit dist / TEDS / CDM    | Whole-page -> structured output across text/formula/table/reading-order; per-element scoring. |
| A3   | [`doclaynet`](doclaynet/)                       | 📝 documented          | COCO mAP                  | Document layout detection (11 classes) across 6 doc types; element localization.              |
| A3   | [`fox`](fox/)                                   | 📝 documented          | edit dist / F1            | Region/colour-guided OCR + cross-page understanding on dense multi-page docs.                 |
| A3   | [`readoc`](readoc/)                             | 📝 documented          | edit-sim / TEDS           | Holistic PDF -> semantic Markdown (text/headings/tables/reading-order).                       |

### B. Question answering & extraction (VQA · KIE · chart)

| Code | Benchmark                     | Status         | Metric               | Purpose (what it measures)                                                                     |
| ---- | ----------------------------- | -------------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| B1   | [`docvqa`](docvqa/)           | 🖼️ sample (HF) | anls                 | Answer questions about scanned business documents; reading + layout + key-value extraction.    |
| B1   | [`infovqa`](infovqa/)         | 🖼️ sample (HF) | anls                 | QA over infographics; joint layout + graphics + text + basic arithmetic reasoning.             |
| B1   | [`textvqa`](textvqa/)         | 🖼️ sample (HF) | exact                | Answer questions that require reading text in natural images.                                  |
| B1   | [`stvqa`](stvqa/)             | 🖼️ sample (HF) | anls                 | VQA requiring reading scene text; origin of the ANLS metric.                                   |
| B1   | [`ocrvqa`](ocrvqa/)           | 🖼️ sample (HF) | exact                | QA over book-cover images; reading titles/authors/edition from text-heavy images.              |
| B1   | [`ai2d`](ai2d/)               | 🖼️ sample (HF) | exact                | Multiple-choice QA over grade-school science diagrams; arrows/labels/structure reasoning.      |
| B1   | [`visualmrc`](visualmrc/)     | 📝 documented   | BLEU/METEOR/CIDEr    | Long abstractive answers over webpage screenshots; reading comprehension + generation.         |
| B2   | [`funsd`](funsd/)             | 🖼️ sample (HF) | entity F1            | Extract entities (question/answer/header) AND key->value links from noisy scanned forms.       |
| B2   | [`cord`](cord/)               | 🖼️ sample (HF) | entity F1            | Extract ~30 fine-grained fields + hierarchy from receipts.                                     |
| B2   | [`sroie`](sroie/)             | 🖼️ sample (HF) | field F1             | Extract 4 fields (company/date/address/total) from receipts; exact field match.                |
| B2   | [`docile`](docile/)           | 📝 documented   | F1 / AP              | Localize+extract key info (KILE) and line items (LIR) from business documents; 55 field types. |
| B2   | [`xfund`](xfund/)             | 📝 documented   | entity / relation F1 | Multilingual (7 languages) form understanding; entity + relation extraction.                   |
| B2   | [`wildreceipt`](wildreceipt/) | 📝 documented   | entity F1            | 26-key receipt KIE in realistic photos; harder capture conditions than SROIE.                  |
| B3   | [`chartqa`](chartqa/)         | 🖼️ sample (HF) | relaxed_acc          | Read + reason (compare/arithmetic) over bar/line/pie charts; relaxed numeric accuracy.         |
| B3   | [`mathvista`](mathvista/)     | 🖼️ sample (HF) | exact                | Visual mathematical reasoning over figures/plots/diagrams; multi-step quantitative reasoning.  |
| B3   | [`plotqa`](plotqa/)           | 📝 documented   | relaxed_acc          | QA over scientific plots with out-of-vocabulary numeric answers; data extraction + reasoning.  |
| B3   | [`dvqa`](dvqa/)               | 📝 documented   | exact                | Bar-chart QA testing structure/element reading and value comparison.                           |

### C. Structure recovery (tables · formulas)

| Code | Benchmark                     | Status         | Metric                   | Purpose (what it measures)                                                                 |
| ---- | ----------------------------- | -------------- | ------------------------ | ------------------------------------------------------------------------------------------ |
| C1   | [`pubtabnet`](pubtabnet/)     | 🖼️ sample (HF) | TEDS / TEDS-S            | Reconstruct table structure + cell content (HTML) from images; cell-relationship fidelity. |
| C1   | [`pubtables1m`](pubtables1m/) | 📄 label only   | GriTS / mAP              | Large-scale table detection + structure recognition; GriTS over cell grids (topology).     |
| C1   | [`fintabnet`](fintabnet/)     | 📝 documented   | TEDS                     | Complex financial tables (merged cells, dense numerics); structure + content extraction.   |
| C1   | [`scitsr`](scitsr/)           | 📝 documented   | cell-adjacency F1        | Table structure from scientific papers; neighbour-relationship correctness.                |
| C2   | [`im2latex`](im2latex/)       | 🖼️ sample (HF) | edit dist / BLEU         | Image -> LaTeX for printed formulas; token accuracy / normalized edit distance.            |
| C2   | [`latexocr`](latexocr/)       | 🖼️ sample (HF) | edit dist / BLEU / exact | Image -> LaTeX rendering; exact-match + edit-distance on the token sequence.               |
| C2   | [`crohme`](crohme/)           | 📝 documented   | expression rec. rate     | Handwritten math expression recognition; expression-level + symbol-level accuracy.         |

### D. Umbrella OCR & figure suites

| Code | Benchmark                           | Status         | Metric                    | Purpose (what it measures)                                                                    |
| ---- | ----------------------------------- | -------------- | ------------------------- | --------------------------------------------------------------------------------------------- |
| D1   | [`ocrbench`](ocrbench/)             | 🖼️ sample (HF) | ocrbench                  | 1000-item OCR capability probe across 5 sub-skills (recog / scene-VQA / doc-VQA / KIE / HME). |
| D1   | [`ocrbench_v2`](ocrbench_v2/)       | 🖼️ sample (HF) | per-task acc              | Extended OCR suite (~10k items, 31 scenarios) incl. fine-grained spotting & structure.        |
| D1   | [`charxiv`](charxiv/)               | 🖼️ sample (HF) | descriptive/reasoning acc | Realistic scientific-figure understanding (arXiv charts); descriptive + reasoning questions.  |
| D1   | [`seedbench2plus`](seedbench2plus/) | 📝 documented   | accuracy                  | MCQ over text-rich images (charts/maps/webs); broad text-comprehension coverage.              |

### E. Reliability: robustness / calibration / hallucination

| Code | Benchmark                           | Status                | Metric                 | Purpose (what it measures)                                                                    |
| ---- | ----------------------------------- | --------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| E1   | [`robustness`](robustness/)         | 🖼️ sample (synthetic) | ANLS retention + ECE   | Stability under capture degradations + terminology paraphrase; confidence calibration.        |
| E1   | [`pope`](pope/)                     | 🖼️ sample (HF)        | F1 / accuracy          | Object-existence polling (yes/no) to quantify hallucination; precision/recall/F1.             |
| E1   | [`hallusionbench`](hallusionbench/) | 🖼️ sample (HF)        | acc + consistency      | Visual-illusion vs knowledge-hallucination yes/no pairs; consistency-aware scoring.           |
| E1   | [`kie_hvqa`](kie_hvqa/)             | 📝 documented          | hallucination-free acc | Penalize confident wrong field values on degraded ID/invoice docs; reward correct-or-abstain. |

### F. Custom capability axes (our probes)

| Code | Benchmark                               | Status                | Metric                                       | Purpose (what it measures)                                                                                                                                                                                                                                                                                                                  |
| ---- | --------------------------------------- | --------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1   | [`capability_probe`](capability_probe/) | 🖼️ sample (synthetic) | anls / relaxed_acc / exact / grounding       | Isolate the document-VLM capability axes on controlled renders: text recognition, localized KIE, integrative reasoning (sum & relations), chart reading, and spatial grounding (bounding box) — built by scripts/make_capability_probe.py with exact GT.                                                                                    |
| F1   | [`custom_eval`](custom_eval/)           | 🖼️ sample (synthetic) | ned / teds / exact / relaxed_acc / grounding | Our proposed real-world eval format: per content-class (text/table/formula/chart/ qr/barcode/stamp/logo), per-language (en/ko/ja/zh), rotation robustness (0/15/90/180/ 270), reading-direction (vertical CJK), and spotting (basis-of-extraction), each scored with a class-appropriate metric. See data/benchmarks/custom_eval/README.md. |
| F1   | [`oov_probe`](oov_probe/)               | 🖼️ sample (synthetic) | ned / exact                                  | Un-tokenizable glyphs (invented / runic / 7-segment): measure the FALLBACK pattern (abstain / transliterate / hallucinate / copy) and whether an in-image legend enables decoding by in-context visual reasoning.                                                                                                                           |
| F1   | [`webui_probe`](webui_probe/)           | 🖼️ sample (synthetic) | grounding / exact / ned                      | Web-agent UI understanding: locate interactive elements (button/search/cart), identify the primary CTA, read the nav, and reason about affordances (what to click to act).                                                                                                                                                                  |

