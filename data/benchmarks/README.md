# Benchmark sample previews

One representative sample (`sample.png` + `sample.json` with ground-truth label, metric, and
source) per document-understanding benchmark, organised by the capability taxonomy in
[`../../report/benchmark_taxonomy.md`](../../report/benchmark_taxonomy.md). Fetched with
`python scripts/fetch_benchmark_samples.py` (HF `datasets` streaming, one example each).

| Folder | Capability category | Metric | HF dataset | Source |
|---|---|---|---|---|
| [`chartqa/`](chartqa/) | 6. Chart understanding | relaxed_acc | `lmms-lab/ChartQA` | [link](https://arxiv.org/abs/2203.10244) |
| [`cord/`](cord/) | 4. Key Information Extraction (receipts) | entity-level F1 | `naver-clova-ix/cord-v2` | [link](https://github.com/clovaai/cord) |
| [`docvqa/`](docvqa/) | 3. Document VQA | ANLS | `lmms-lab/DocVQA` | [link](https://arxiv.org/abs/2007.00398) |
| [`funsd/`](funsd/) | 4. Key Information Extraction (forms) | entity-level F1 | `nielsr/funsd-layoutlmv3` | [link](https://arxiv.org/abs/1905.13538) |
| [`im2latex/`](im2latex/) | 7. Formula recognition | edit distance / BLEU / exact | `OleehyO/latex-formulas` | [link](https://github.com/OleehyO/TexTeller) |
| [`infovqa/`](infovqa/) | 3. Document VQA (infographics) | ANLS | `lmms-lab/DocVQA` | [link](https://arxiv.org/abs/2104.12756) |
| [`ocrbench/`](ocrbench/) | 8. LMM OCR capability suite | OCRBench (/1000) | `echo840/OCRBench` | [link](https://arxiv.org/abs/2305.07895) |
| [`omnidocbench/`](omnidocbench/) | 9. End-to-end page parsing | edit distance / TEDS / CDM | `opendatalab/OmniDocBench` | [link](https://arxiv.org/abs/2412.07626) |
| [`pubtabnet/`](pubtabnet/) | 5. Table recognition | TEDS / GriTS | `apoidea/pubtabnet-html` | [link](https://github.com/ibm-aur-nlp/PubTabNet) |
| [`sroie/`](sroie/) | 4. Key Information Extraction (receipts) | field-level F1 | `priyank-m/SROIE_2019_text_recognition` | [link](https://rrc.cvc.uab.es/?ch=13) |
| [`textvqa/`](textvqa/) | 3. Scene-text VQA | VQA accuracy | `lmms-lab/textvqa` | [link](https://arxiv.org/abs/1904.08920) |

`sample.json` carries the full record's ground truth (question/answers/fields), the metric
name + a one-line scoring note, and the dataset split/source — enough to understand what
each benchmark asks and how it is scored, without downloading the full sets.
