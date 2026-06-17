# Small VLMs for Document Understanding — Evaluation & Improvement PoC

A reproducible **proof-of-concept** for the task *"Adapting Small Vision-Language Models
(<1B) for Document Understanding"*:

- **Part 1 — Evaluation:** survey sub-1B VLMs, run them on a document-understanding benchmark
  suite, and compare them with metrics that go **beyond accuracy** (calibration + robustness).
- **Part 2 — Improvement strategy:** turn the gap analysis into a concrete, literature-grounded
  fine-tuning plan, backed by the LoRA scaffold in this repo.

> 📄 **Technical report:** [`report/technical_report.pdf`](report/technical_report.pdf)
> (source: [`report/technical_report.md`](report/technical_report.md))
> 🧭 **Benchmark & metric taxonomy:** [`report/benchmark_taxonomy.md`](report/benchmark_taxonomy.md)
> 📊 **Comparison table:** [`results/comparison_table.md`](results/comparison_table.md)

---

## What's here

```
src/docvlm_eval/        # the evaluation harness (Part 1)
  models/               #   adapter per VLM + registry  (add a model = 1 small file + 1 line)
  benchmarks/           #   HF dataset builders + custom robustness probe
  metrics/              #   ANLS / relaxed-acc / OCRBench + ECE calibration + aggregation
  pipeline.py           #   model x benchmark -> predictions + scores
scripts/
  evaluate.py           #   ⭐ single entrypoint: load ANY model, run benchmark, emit scores
  build_benchmarks.py   #   DocVQA / InfoVQA / ChartQA / OCRBench -> normalised JSONL
  build_robustness_set.py  # paired clean/perturbed probe (capture quality + terminology)
  make_comparison_table.py # all runs -> comparison table (md/csv/json)
  run_all.sh            #   full reproduction (Colab/Kaggle T4)
  build_report.py       #   technical_report.md -> PDF
  finetune_lora.py ...  #   Part-2 LoRA fine-tuning scaffold (improvement PoC)
configs/                # models.yaml, benchmarks.yaml (documented choices)
tests/                  # metric unit tests (the numbers must be exactly right)
report/ results/ data/  # report, taxonomy, comparison table, benchmark/probe data
```

**Candidate models** (`scripts/evaluate.py --list-models`): `internvl2_5-1b`, `internvl3-1b`,
`smolvlm-256m`, `smolvlm-500m`, `llava-ov-0.5b`, `got-ocr2`, `florence2-large`, `paddleocr-vl`
(+ `dummy-echo` for CPU smoke tests). See the report Appendix A for profiles.

**Benchmark suite:** DocVQA, InfoVQA, ChartQA, OCRBench (from VLMEvalKit/HF) + a custom
robustness probe. The full landscape of OCR/document benchmark *types and metrics* is in
[`report/benchmark_taxonomy.md`](report/benchmark_taxonomy.md).

**Inspect the benchmarks at a glance.** One representative sample (image + ground-truth label
+ metric note) per benchmark — across all taxonomy categories (VQA, KIE, tables, charts,
formulas, end-to-end parsing) — lives under
[`data/benchmarks/`](data/benchmarks/README.md), fetched with:
```bash
python scripts/fetch_benchmark_samples.py        # real samples via HF streaming
python scripts/make_synthetic_samples.py         # attach samples for categories not on HF
                                                 # (full-page recognition, scene text, robustness)
```

---

## Quick start

### 0) Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
```

### 1) Smoke test (CPU, no weights, ~seconds)
Proves the whole pipeline works end-to-end before spending GPU time:
```bash
python scripts/evaluate.py --model dummy-echo \
  --benchmark data/custom/custom.jsonl --benchmark-name custom \
  --out /tmp/custom --device cpu
python -m pytest tests/ -q          # metric unit tests
```

### 2) Full evaluation (free Colab/Kaggle T4)
```bash
# build benchmarks (use --limit for a fast subset)
python scripts/build_benchmarks.py --benchmark all --limit 300
python scripts/build_robustness_set.py --base data/benchmarks/docvqa.jsonl \
  --out-dir data/robustness/docvqa --limit 100

# evaluate one model on one benchmark
python scripts/evaluate.py --model internvl2_5-1b \
  --benchmark data/benchmarks/docvqa.jsonl --benchmark-name docvqa \
  --out results/internvl2_5-1b/docvqa --limit 300

# ...or run everything and build the comparison table
bash scripts/run_all.sh                  # LIMIT=20 bash scripts/run_all.sh  for a quick pass
python scripts/make_comparison_table.py  # -> results/comparison_table.{md,csv,json}
```

### 3) Add a new model (the "minimal modification" requirement)
Create `src/docvlm_eval/models/mymodel.py`:
```python
from dataclasses import dataclass
from .base import ModelAdapter
from .registry import register

@register("my-model")
@dataclass
class MyModel(ModelAdapter):
    hf_id: str = "org/my-model"
    param_count_m: float = 800.0
    def load(self):
        ...  # build self.model / self.processor
    def generate(self, image_path, question):
        ...  # return (answer_text, confidence_or_None)
```
…then add it to the lazy imports in `registry.build_model` (one line). No change to
`evaluate.py`.

### 4) Rebuild the report PDF
```bash
python scripts/build_report.py     # report/technical_report.md -> .pdf
```

---

## Design notes

- **One sample schema** (`docvlm_eval.schema.Sample`) normalises every benchmark, so the same
  loop evaluates any model on any dataset.
- **Beyond accuracy.** Besides the official ANLS / relaxed-accuracy / OCRBench scores, the
  pipeline computes **ECE calibration** (does the model know when it's wrong?) and
  **robustness retention** on a paired clean/perturbed probe (does it survive phone-photo /
  fax / jargon conditions?) — the columns public leaderboards omit. See report §I.3.
- **Reproducibility.** Greedy decoding, fixed seeds, pinned versions, cached
  `predictions.jsonl` (resumable + re-scorable without re-running the model).
- **Why a custom harness alongside VLMEvalKit:** VLMEvalKit is the standard for headline
  accuracy (and we mirror its dataset/metric choices), but it does not report calibration or
  our robustness probe — which are central to the document *deployment* story.

---

## Part 2 — Fine-tuning scaffold (improvement PoC)

The improvement strategy (report §II.2) is backed by a LoRA fine-tuning scaffold under
`src/ocr_ft` + `scripts/finetune_lora.py | eval.py | compare.py | merge_lora.py`. It expects
JSONL of `{"image_path", "text"}` and supports LoRA(PEFT) SFT, CER/WER eval, vanilla-vs-tuned
comparison, and adapter merge — the machinery for Steps 1–4 of the plan. Full scaffold docs:
[`docs/finetune_scaffold.md`](docs/finetune_scaffold.md).
