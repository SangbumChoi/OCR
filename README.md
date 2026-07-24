# Small VLMs for Document Understanding — Evaluation & Improvement PoC

A reproducible **proof-of-concept** for the task *"Adapting Small Vision-Language Models
(<1B) for Document Understanding"*:

- **Part 1 — Evaluation:** survey sub-1B VLMs, run them on a document-understanding benchmark
  suite, and compare them with metrics that go **beyond accuracy** (calibration + robustness).
- **Part 2 — Improvement strategy:** turn the gap analysis into a concrete, literature-grounded
  fine-tuning plan, backed by the LoRA scaffold in this repo.

> 🧭 **Start here — project plan / reading order:** [`docs/plan.md`](docs/plan.md)
> (the north-star narrative: define "document" (incl. UX/screens) → survey variations → evaluations
> (data > models) → special cases → evaluate models → integrate-don't-pipeline → ablation staircase)
> 📄 **Technical report:** [`docs/report/technical_report.pdf`](docs/report/technical_report.pdf)
> (source: [`docs/report/technical_report.md`](docs/report/technical_report.md))
> 🧭 **Benchmark & metric taxonomy:** [`docs/report/benchmark_taxonomy.md`](docs/report/benchmark_taxonomy.md)
> 🗺️ **Benchmark patterns & priority map:** [`docs/report/benchmark_patterns.md`](docs/report/benchmark_patterns.md)
> (what each benchmark collects · visual-class diversity · VQA answer-natures · grouping/priority)
> 🧩 **Capability axes & custom probe:** [`docs/report/capability_axes.md`](docs/report/capability_axes.md)
> (text vs location understanding · KIE/integrative/chart output natures · grounding fair-comparison)
> 🗂️ **Document-type taxonomy:** [`docs/report/document_type_taxonomy.md`](docs/report/document_type_taxonomy.md)
> (type × stressor matrix — webtoon/ID/historical/LCD … → which metric/axis each needs)
> 💡 **Cross-model insights (auto):** [`docs/report/insights.md`](docs/report/insights.md)
> (capability leaders · reasoning-emergence · grounding gap · efficiency frontier · OOV fallback)
> 🪜 **Ablation plan:** [`docs/report/ablation_plan.md`](docs/report/ablation_plan.md)
> (spotting/reasoning/multilingual/LoRA-placement/HPO/preprocessing ablations → cumulative staircase)
> 🔬 **Research novelty & open questions:** [`docs/report/research_novelty.md`](docs/report/research_novelty.md)
> (lit-grounded gaps the probes/ablations here can uniquely test at ≤1B)
> 🔎 **Results analysis & flaws:** [`docs/report/results_analysis.md`](docs/report/results_analysis.md)
> (real CPU runs · inference bugs fixed · scoring flaws · per-model capability vector)
> 📊 **Comparison table:** [`docs/results/comparison_table.md`](docs/results/comparison_table.md)

---

## UDD — the Universal Document Dataset (public-data track)

The Part-2 ablations now also run on **real public data** via
[`danelcsb/UDD`](https://huggingface.co/datasets/danelcsb/UDD) — 32 public sources unified into
**one sharded dataset** (39,837 image-rows / 77,063 QAs / 7 tasks, ≤1,500 images/source), built by
`scripts/build_udd.py` from the task-typed loader in `docvlm_eval.unified`. What the pipeline
guarantees (full story: [`docs/report/unified_loader.md`](docs/report/unified_loader.md)):

- **One row per image, native QA lists** — `instructions: list[str]` paired index-wise with
  `answers: list[list[str]]` (inner list = surface variants of one answer); same-phash duplicate
  images are folded into the lists (77,063 QAs in 39,837 image-rows, zero QAs lost) and
  the pairing/DTO shape is **enforced** at build time (`validate_payload_shapes`).
- **Derived columns** for slicing and hygiene: heuristic `language`, `phash`, hosting-repo
  `license`, leakage-safe `fold` (train/heldout keyed by image identity), payload counts and
  image dims.
- **Data quality fixed structurally**: CORD answers keep every line item (nested `gt_parse`),
  case/punctuation duplicate golds are canonicalized, insertion-time hash-index dedup
  (it caught MathVista re-using a ChartQA image), duplicate/near-duplicate audit committed
  ([`docs/results/udd_duplicates.md`](docs/results/udd_duplicates.md)).
- **Ablations on public data**: full-pool per-task / per-language training sets with a public
  heldout fold (`scripts/build_task_trainsets.py --per-task -1`), 24 composed arm mixes for
  A1–A6 trained at each family's largest equal-N budget
  (`scripts/run_udd_ablation.py`, incl. geometry-derived A2 rationales with an answer-only
  control), a **metric bank** (SQuAD-F1 / DROP-EM / CER / layered `semantic_match`) with a
  measured tolerance matrix ([`docs/results/metric_tendency.md`](docs/results/metric_tendency.md)),
  and a GPU-free **mock multi-model eval** that sanity-checks the whole scoring path
  ([`docs/results/udd_mock_eval.md`](docs/results/udd_mock_eval.md)).
- **What merging buys, measured without training**:
  [`docs/results/udd_merge_value.md`](docs/results/udd_merge_value.md); further ablation-usable
  features mined from the corpus:
  [`docs/results/udd_ablation_features.md`](docs/results/udd_ablation_features.md).
- **Verify the ablation wiring end-to-end (no GPU):**
  [`notebooks/udd_ablation.ipynb`](notebooks/udd_ablation.ipynb).

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
docs/report/ docs/results/ data/  # report+figures, comparison table, benchmark/probe samples
```

**Candidate models** (`scripts/evaluate.py --list-models`): `internvl2_5-1b`, `internvl3-1b`,
`smolvlm-256m`, `smolvlm-500m`, `llava-ov-0.5b`, `got-ocr2`, `florence2-large`, `paddleocr-vl`
(+ `dummy-echo` for CPU smoke tests). See the report Appendix A for profiles.

**Benchmark suite:** DocVQA, InfoVQA, ChartQA, OCRBench (from VLMEvalKit/HF) + a custom
robustness probe. The full landscape of OCR/document benchmark *types and metrics* is in
[`docs/report/benchmark_taxonomy.md`](docs/report/benchmark_taxonomy.md).

**Inspect the benchmarks at a glance.** One representative sample (image + ground-truth label
+ metric note) per benchmark — across all taxonomy categories (VQA, KIE, tables, charts,
formulas, end-to-end parsing) — lives under
[`data/benchmarks/`](data/benchmarks/README.md), fetched with:
```bash
python scripts/fetch_benchmark_samples.py        # real samples via HF streaming
python scripts/make_synthetic_samples.py         # attach samples for categories not on HF
                                                 # (full-page recognition, scene text, robustness)
```

**Realistic synthetic special cases.** Genuinely document-*looking* renders of the hard document
types (ID/passport, cheque, prescription, redacted, RTL, webtoon, ancient, LCD …) — realistic
*and* GT-exact, usable as realistic eval **and** Part-2 fine-tuning data. Pipeline: HTML/CSS +
Faker → WeasyPrint → PDF → PyMuPDF (rasterize + exact spotting boxes) → Augraphy degradation.
See [`data/probes/realistic_cases/`](data/probes/realistic_cases/README.md):
```bash
pip install -e ".[synth]"
python scripts/make_realistic_cases.py           # 18 cases, each clean.png + degraded.png + gt.json
```

The generator also includes four executable-graph hard families: dense tables, labelled charts,
multi-path beneficial ownership, and quantitative scientific papers. Their answers, rationales,
and multi-box evidence are recomputed from the same graph that supplies the rendered values:

```bash
python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science \
  --difficulty-level 5 --split-name train --count 100 --out data/generated/hard_train

python scripts/validate_synth_splits.py \
  --split train=data/generated/hard_train \
  --split heldout=data/generated/hard_heldout
```

See [`docs/report/hard_synthetic_pipeline.md`](docs/report/hard_synthetic_pipeline.md) for the
curriculum, semantic fingerprints, leakage policy, and structured SFT/RLVR evidence contract.

The LFM spotting track also has a parameter-matched confirmatory LoRA placement experiment:

```bash
python scripts/run_lora_placement_sweep.py --dry-run
# after the six GPU jobs:
python scripts/analyze_lora_placement_sweep.py
```

It compares vision-only adaptation with the exact vision+connector target union over three paired
seeds. See
[`docs/report/student_lora_placement_interaction.md`](docs/report/student_lora_placement_interaction.md).

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
  --benchmark data/probes/custom_eval/custom_eval.jsonl --benchmark-name custom_eval \
  --out /tmp/custom_eval --device cpu
pytest -q                            # 60+ tests: metrics, registry, pipeline, robustness, …
```

### 2) Full evaluation (free Colab/Kaggle T4)
```bash
# OFFLINE 10-sample slice of every benchmark (no network — uses the committed previews):
python scripts/build_preview_eval.py        # -> data/benchmarks/preview_eval.jsonl (16 benchmarks)

# ...or build full benchmarks from HF (use --limit for a fast subset)
python scripts/build_benchmarks.py --benchmark all --limit 300
python scripts/build_robustness_set.py --base data/benchmarks/docvqa.jsonl \
  --out-dir data/robustness/docvqa --limit 100

# evaluate one model on one benchmark
python scripts/evaluate.py --model internvl2_5-1b \
  --benchmark data/benchmarks/docvqa.jsonl --benchmark-name docvqa \
  --out docs/results/internvl2_5-1b/docvqa --limit 300

# ...or run everything and build the comparison table
bash scripts/run_all.sh                  # LIMIT=20 bash scripts/run_all.sh  for a quick pass
python scripts/make_comparison_table.py  # -> docs/results/comparison_table.{md,csv,json}
```

### 2a) Resumable runs on a GPU-limited / ephemeral box (Colab)
The pipeline writes `predictions.jsonl` **per sample** and resumes by `sample_id`, so an
interrupted run never recomputes finished work. To survive the **container being reset when the
free-GPU limit hits**, checkpoint results to git and resume across sessions:
```bash
MODELS="smolvlm-256m smolvlm-500m internvl3-1b" \
BENCH=data/benchmarks/preview_eval.jsonl NAME=preview_eval DEVICE=cuda \
bash scripts/run_checkpointed.sh    # pull -> run each model -> commit+push its predictions/summary
```
Re-running after a reset `git pull`s the partial results back and continues from where it stopped.
(`docs/results/**/predictions.jsonl` and `summary.json` are git-tracked for exactly this; the
matrix is rebuilt from every committed `summary.json`.)

### 2b) Full comparison on a GPU (Colab T4) — incl. PaddleOCR-VL 1.0/1.5/1.6 + efficiency
Open [`notebooks/colab_full_comparison.ipynb`](notebooks/colab_full_comparison.ipynb) in Colab
(GPU runtime). It only clones + installs + runs repo scripts (`scripts/run_full_comparison.sh`),
which handles the two transformers passes (4.49 for the chat VLMs, 4.57 for PaddleOCR-VL) and
runs every model on the capability + spatial/context probes. Each run records **score +
inference time + peak CPU/GPU memory** (measured by the model wrapper) into `summary.json`, and
`docs/results/matrix_*.md` includes an **Efficiency** table (load / avg latency / p90 / peak CPU MB /
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
python scripts/build_report.py     # docs/report/technical_report.md -> .pdf
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

### Native sub-1B student

The near-random-initialization track has an executable approximately 800M model rather than only a
design document. It consists of a 12-layer ViT, a 64-token gated resampler, and a 23-layer GQA
decoder, with contrastive, orientation, and valid-box auxiliary heads:

```bash
pip install -e ".[student,student-gpu]"
python scripts/build_sub1b_student.py --device meta
# exact result: 799,919,884 parameters, without allocating the weights
```

`--init-arm I0_random|I1_vision|I2_language|I3_dual|I4_selective` controls initialization.
Exact-shape SigLIP/Llama-style blocks can be depth-mapped and copied; incompatible dimensions stay
random and are reserved for feature/logit distillation. See
[`docs/report/sub1b_architecture_blueprint.md`](docs/report/sub1b_architecture_blueprint.md).
The one-image production microbatch uses an exact-resumable local contrastive memory so the
alignment head sees real cross-document negatives; see
[`docs/report/student_contrastive_memory.md`](docs/report/student_contrastive_memory.md).
The default input path uses per-image packed patch sequences with stable 2D positions. Its paired
dense adaptive, aspect-bucketed, and fixed-square controls and visual-padding efficiency
measurements are documented in
[`docs/report/student_visual_canvas_sweep.md`](docs/report/student_visual_canvas_sweep.md).
The production recipe uses fail-closed bitsandbytes AdamW8bit in every native training stage;
CPU smoke plans resolve explicitly to standard AdamW. Runtime identity, version, state bytes, and
resume invariants are documented in
[`docs/report/student_optimizer_memory.md`](docs/report/student_optimizer_memory.md).

The native UDD input path is also executable. It lazily expands every image's QA list, derives
single-evidence grounding examples from structured elements, balances task/source/language groups,
trains a new NFC byte-level multilingual tokenizer, rotates boxes with images, masks prompt and
padding tokens, and prevents padded visual patches from entering ViT or resampler attention:

```bash
python scripts/train_student_tokenizer.py \
  --repo danelcsb/UDD \
  --output artifacts/student_tokenizer
```

The batch contract and its spatial invariants are documented in
[`docs/report/student_input_pipeline.md`](docs/report/student_input_pipeline.md).

The native pretraining runner adds same-tokenizer top-k logit and selected-feature distillation,
token-count warmup/cosine decay, mixed precision, distributed balanced sampling, held-out loss
slices, and atomic exact resume:

```bash
python scripts/pretrain_student.py \
  --repo danelcsb/UDD \
  --tokenizer artifacts/student_tokenizer \
  --output outputs/student_pretrain/I0_random
```

Use `torchrun --standalone --nproc-per-node=N` for data parallel training and `--resume latest`
after interruption. LFM and other cross-tokenizer teachers supply fingerprinted, metric-gated
offline sequence targets; native same-tokenizer teachers can additionally provide online top-k KL
and feature alignment. See
[`docs/report/student_pretraining_runner.md`](docs/report/student_pretraining_runner.md).

The second stage is also executable. It teaches a strict
`{"answer","evidence","rationale"}` contract with exhaustive SFT, then optionally runs
verifier-ranked DPO/IPO, single-update GRPO, or the sequential preference-to-GRPO path with an
immutable SFT reference. Rewards cover task-applicable exact, text, box, table, chart, formula,
grounding, and abstention signals:

```bash
python scripts/posttrain_student.py sft \
  --samples data/posttraining/train.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_pretrain/I0_random/checkpoints/step-00010000/student \
  --output outputs/student_sft/evidence_linked

python scripts/posttrain_student.py rlvr \
  --samples data/posttraining/rlvr.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_sft/evidence_linked/checkpoints/step-00002000/student \
  --output outputs/student_rlvr/full_reward
```

Preference optimization defaults to a verifier-ranked candidate set with one exact collated SFT
target. This prevents an early near-random model from finishing DPO/IPO with only tied malformed
responses and zero optimizer updates; a reference-only candidate source remains available as a
matched ablation.

RL rollout reuses one encoded visual prefix per image. When GRPO follows DPO/IPO, the preference
checkpoint starts the trainable policy while `--reference-checkpoint` pins the exact SFT model.
SFT supports `torchrun`; native RLVR is currently single-process and requires the full policy,
frozen reference, and optimizer state on one device. See
[`docs/report/student_posttraining_runner.md`](docs/report/student_posttraining_runner.md).

The complete native-student path is available as one validated, resumable experiment DAG:

```bash
python scripts/run_student_experiment.py --dry-run
python scripts/run_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml
```

The full configuration is `configs/sub1b_experiment.yaml`. Both configurations connect independent
hard-document train/heldout generation, leakage validation, weighted UDD mixing, cross-tokenizer
teacher generation and quality gating, tokenizer and student creation, pretraining, SFT, RLVR, and
split evaluation; preference optimization can optionally be inserted between SFT and RLVR. The full
DAG adds a separate validation split and converts matched
baseline-to-final learning progress plus residual structured failures into a content-addressed
next-batch synthesis plan; only validation-derived plans may replace train generation in the
following run. It also blocks initialization on target-GPU visual
parity/performance and a production-shaped full-model forward/backward/AdamW memory probe. See
[`docs/report/student_experiment_runner.md`](docs/report/student_experiment_runner.md).
The policy and leakage contract are documented in
[`docs/report/student_failure_driven_synthesis.md`](docs/report/student_failure_driven_synthesis.md).

After a run, produce and independently re-verify a full-hash evidence attestation:

```bash
python scripts/attest_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml
python scripts/attest_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml \
  --verify outputs/docvlm-tiny-smoke/evidence_attestation.json
```

The attestation separates pipeline execution from capability evidence, so a successful smoke run
cannot be mistaken for a quality or deployment pass. See
[`docs/report/student_experiment_evidence.md`](docs/report/student_experiment_evidence.md).

The full mixture now acquires the pinned public `danelcsb/UDD` train fold at 55% and combines it
with 45% authored hard documents. Hub revision, selection, schema, duplicate identity, decoded
image dimensions, and source distributions are recorded before mixing. See
[`docs/report/student_data_acquisition.md`](docs/report/student_data_acquisition.md).

Pinned SigLIP and Qwen checkpoints can now enter the same DAG as selective initialization sources.
Their revisions and cached files are validated before transfer. The five-arm baseline suite is
`configs/sub1b_initialization_sweep.yaml`; the focused exact-versus-structured SwiGLU test is
`configs/sub1b_structured_mlp_transfer_sweep.yaml`; and the strict attention-geometry interaction
test is `configs/sub1b_attention_geometry_transfer_factorial.yaml`. See
[`docs/report/student_initialization_runner.md`](docs/report/student_initialization_runner.md).

The 45-run `configs/sub1b_initialization_data_scale.yaml` factorial crosses those five arms with
three unique-data scales at fixed token budgets and fixed heldout documents. It reports paired
difference-in-differences and actual mixed rows. See
[`docs/report/student_factorial_runner.md`](docs/report/student_factorial_runner.md).

The 15-run `configs/sub1b_architecture_compute_sweep.yaml` compares five input-resolution and
visual-latent profiles under identical analytical student FLOP budgets across pretraining, SFT,
and RLVR. Runtime counters, compute-progress schedules, and a realized-budget overshoot gate make
the fixed-compute claim executable. See
[`docs/report/student_architecture_compute_sweep.md`](docs/report/student_architecture_compute_sweep.md).

The 15-run `configs/sub1b_pretraining_loss_sweep.yaml` removes each active native pretraining loss
one at a time while preserving offline LFM targets and paired controls. Checkpoints record the
resolved supervision contract, and online native-teacher losses now fail closed when no compatible
teacher is supplied. See
[`docs/report/student_pretraining_loss_sweep.md`](docs/report/student_pretraining_loss_sweep.md).

The six-run `configs/sub1b_token_relation_distillation_sweep.yaml` compares pointwise hidden
anchors with bounded token-relation KL under the same native teacher, logit-KL weight,
representation-loss weight, and three paired seed blocks. See
[`docs/report/student_token_relation_distillation_sweep.md`](docs/report/student_token_relation_distillation_sweep.md).

A nine-run SFT suite and fifteen-run RLVR suite separate target and reward effects. The SFT suite
compares answer-only, free-rationale, and evidence-linked checkpoints before RLVR. The RLVR suite
fixes evidence-linked SFT and compares SFT-only, correctness-only, no-rationale, and full
decomposed-reward arms. Additional paired preference sweeps isolate method, objective, and
gold-anchored versus reference-only candidate construction. See
[`docs/report/student_posttraining_sweeps.md`](docs/report/student_posttraining_sweeps.md).

The nine-run `configs/sub1b_sequence_teacher_sweep.yaml` compares gold-only, pinned LFM, and pinned
Qwen sequence supervision. It fixes a 4,096-request budget, an exact 400-target dose, and a
gold-only tokenizer corpus so teacher coverage and vocabulary drift cannot masquerade as teacher
quality. See
[`docs/report/student_sequence_teacher_sweep.md`](docs/report/student_sequence_teacher_sweep.md).

Compare train and heldout generation from any native checkpoint:

```bash
python scripts/eval_student.py \
  --split train=data/posttraining/train.jsonl \
  --split heldout=data/posttraining/heldout.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_rlvr/full_reward/checkpoints/step-00001000/student \
  --output outputs/student_eval/full_reward \
  --wandb-project docvlm-ablation
```

This writes standard benchmark scores and structured rewards separately, including
train-minus-heldout gaps and task/source/language slices. It also evaluates canonical
document-family, language, evidence-count, and degradation slices. W&B receives
`eval/<split>_<axis>`, `eval_by_axis/<axis>/<split>`, and
`eval_by_slice/<axis>/<value>/<split>` keys for paired panels.

To issue executable deployment-gate decisions, compare against prior and monolingual-control
evaluation roots:

```bash
python scripts/eval_student.py \
  --split train=data/posttraining/train.jsonl \
  --split heldout=data/posttraining/heldout.jsonl \
  --tokenizer artifacts/student_tokenizer \
  --checkpoint outputs/student_rlvr/candidate/checkpoints/step-00001000/student \
  --output outputs/student_eval/candidate \
  --baseline-evaluation outputs/student_eval/reference \
  --monolingual-control-evaluation outputs/student_eval/monolingual_controls
```

`gates.json` distinguishes `pass`, `fail`, and `insufficient_evidence`. It checks the actual
parameter count, matched held-out improvement and train gap, grounding versus extraction
retention, counterfactual reasoning, per-language retention, and selective risk plus hallucination
at fixed coverage.
