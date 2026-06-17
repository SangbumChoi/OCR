# Benchmark sample previews

One representative sample (`sample.png` + `sample.json` with ground-truth label, metric, and
source) per document-understanding capability, organised by the taxonomy in
[`../../report/benchmark_taxonomy.md`](../../report/benchmark_taxonomy.md).

- Real benchmark samples are fetched via `python scripts/fetch_benchmark_samples.py` (HF streaming).
- Categories not cleanly available on HF (full-page recognition, scene text) and a reliability
  example are **attached locally** via `python scripts/make_synthetic_samples.py` and clearly
  marked as synthetic/derived in their `sample.json` (they render the task, not the official set).

| Folder | Capability category | Metric | Source dataset | Origin |
|---|---|---|---|---|
| [`recognition_fullpage/`](recognition_fullpage/) | 1. Full-page / printed text recognition | CER / WER / NED | `(attached)` | synthetic/derived |
| [`scenetext/`](scenetext/) | 2. Scene-text detection & recognition | detection H-mean / word accuracy / 1-NED | `(attached)` | synthetic/derived |
| [`docvqa/`](docvqa/) | 3. Document VQA | ANLS | `lmms-lab/DocVQA` | HF download |
| [`infovqa/`](infovqa/) | 3. Document VQA (infographics) | ANLS | `lmms-lab/DocVQA` | HF download |
| [`textvqa/`](textvqa/) | 3. Scene-text VQA | VQA accuracy | `lmms-lab/textvqa` | HF download |
| [`cord/`](cord/) | 4. Key Information Extraction (receipts) | entity-level F1 | `naver-clova-ix/cord-v2` | HF download |
| [`funsd/`](funsd/) | 4. Key Information Extraction (forms) | entity-level F1 | `nielsr/funsd-layoutlmv3` | HF download |
| [`sroie/`](sroie/) | 4. Key Information Extraction (receipts) | field-level F1 | `priyank-m/SROIE_2019_text_recognition` | HF download |
| [`pubtabnet/`](pubtabnet/) | 5. Table recognition | TEDS / GriTS | `apoidea/pubtabnet-html` | HF download |
| [`chartqa/`](chartqa/) | 6. Chart understanding | relaxed_acc | `lmms-lab/ChartQA` | HF download |
| [`im2latex/`](im2latex/) | 7. Formula recognition | edit distance / BLEU / exact | `OleehyO/latex-formulas` | HF download |
| [`ocrbench/`](ocrbench/) | 8. LMM OCR capability suite | OCRBench (/1000) | `echo840/OCRBench` | HF download |
| [`omnidocbench/`](omnidocbench/) | 9. End-to-end page parsing | edit distance / TEDS / CDM | `opendatalab/OmniDocBench` | HF download |
| [`robustness/`](robustness/) | 10. Reliability / robustness / calibration | ANLS retention + ECE | `(attached)` | synthetic/derived |

`sample.json` carries the ground truth (question/answers/fields/transcript), the metric +
a one-line scoring note, and the source/origin — enough to understand what each benchmark
asks and how it is scored, without downloading the full sets.
