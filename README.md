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
> 🗺️ **Benchmark patterns & priority map:** [`report/benchmark_patterns.md`](report/benchmark_patterns.md)
> (what each benchmark collects · visual-class diversity · VQA answer-natures · grouping/priority)
> 🧩 **Capability axes & custom probe:** [`report/capability_axes.md`](report/capability_axes.md)
> (text vs location understanding · KIE/integrative/chart output natures · grounding fair-comparison)
> 🗂️ **Document-type taxonomy:** [`report/document_type_taxonomy.md`](report/document_type_taxonomy.md)
> (type × stressor matrix — webtoon/ID/historical/LCD … → which metric/axis each needs)
> 💡 **Cross-model insights (auto):** [`report/insights.md`](report/insights.md)
> (capability leaders · reasoning-emergence · grounding gap · efficiency frontier · OOV fallback)
> 🪜 **Part-2 ablation plan:** [`report/part2_ablation_plan.md`](report/part2_ablation_plan.md)
> (spotting/reasoning/multilingual/LoRA-placement/HPO/preprocessing ablations → cumulative staircase)
> 🔬 **Research novelty & open questions:** [`report/research_novelty.md`](report/research_novelty.md)
> (lit-grounded gaps the probes/ablations here can uniquely test at ≤1B)
> 🔎 **Results analysis & flaws:** [`report/results_analysis.md`](report/results_analysis.md)
> (real CPU runs · inference bugs fixed · scoring flaws · per-model capability vector)
> 📊 **Comparison table:** [`results/comparison_table.md`](results/comparison_table.md)

---

## What's here

One installable package, `docvlm_eval` (src layout):

```
src/docvlm_eval/        # the unified package  (pip install -e .)
  schema.py             #   Sample / Prediction
  pipeline.py           #   model x benchmark -> predictions + scores
  cli.py                #   console entry points (docvlm-eval / -fetch / -table / ...)
  comparison.py         #   runs -> comparison table (md/csv/json)
  models/               #   adapter per VLM + registry  (add a model = 1 small file + 1 line)
  benchmarks/           #   HF builders + catalog + custom robustness probe
  metrics/              #   ANLS / relaxed-acc / OCRBench + ECE calibration + aggregation
  finetune/             #   Part-2 LoRA fine-tuning subpackage (was src/ocr_ft)
scripts/                # thin shims over docvlm_eval.cli + run_all.sh / build_report.py /
                        #   fetch_benchmark_samples.py / make_synthetic_samples.py / plot_*
configs/                # models.yaml, benchmarks.yaml, benchmark_catalog.yaml
tests/                  # pytest suite (metrics, schema, loaders, registry, robustness,
                        #   pipeline, catalog, comparison, cli, finetune)  -> 60+ tests
report/ results/ data/  # report+figures, comparison table, benchmark/probe samples
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

### 0) Install (pip-installable; pick the extras you need)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .                      # core: pipeline + metrics + catalog + dummy model + tests
pip install -e ".[models]"            # + torch/transformers/datasets to run real VLMs
pip install -e ".[models,finetune,report,dev]"   # everything (Part-2 LoRA, report, tests)
```
This installs console commands: `docvlm-eval`, `docvlm-build-bench`, `docvlm-fetch`,
`docvlm-robustness`, `docvlm-table`. (`scripts/*.py` remain as thin shims for `python scripts/…`.)

### 1) Smoke test (CPU, no weights, ~seconds)
Proves the whole pipeline works end-to-end before spending GPU time:
```bash
docvlm-eval --model dummy-echo \
  --benchmark data/custom/custom.jsonl --benchmark-name custom \
  --out /tmp/custom --device cpu
pytest -q                            # 60+ tests: metrics, registry, pipeline, robustness, …
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

### 2b) Full comparison on a GPU (Colab T4) — incl. PaddleOCR-VL 1.0/1.5/1.6 + efficiency
Open [`notebooks/colab_full_comparison.ipynb`](notebooks/colab_full_comparison.ipynb) in Colab
(GPU runtime). It only clones + installs + runs repo scripts (`scripts/run_full_comparison.sh`),
which handles the two transformers passes (4.49 for the chat VLMs, 4.57 for PaddleOCR-VL) and
runs every model on the capability + spatial/context probes. Each run records **score +
inference time + peak CPU/GPU memory** (measured by the model wrapper) into `summary.json`, and
`results/matrix_*.md` includes an **Efficiency** table (load / avg latency / p90 / peak CPU MB /
peak GPU MB) so models are compared on quality *and* cost.

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

The improvement strategy (report §II.2) is backed by a LoRA fine-tuning subpackage at
`src/docvlm_eval/finetune` + `scripts/finetune_lora.py | eval.py | compare.py | merge_lora.py`. It expects
JSONL of `{"image_path", "text"}` and supports LoRA(PEFT) SFT, CER/WER eval, vanilla-vs-tuned
comparison, and adapter merge — the machinery for Steps 1–4 of the plan. Full scaffold docs:
[`docs/finetune_scaffold.md`](docs/finetune_scaffold.md).
