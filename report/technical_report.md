# Adapting Small Vision-Language Models (<1B) for Document Understanding

**A systematic evaluation and improvement strategy**

Author: Sangbum Choi · Date: 2026-06-17

---

## 0. Executive summary

Small VLMs (sub-1B parameters) are attractive for document understanding because documents
are high-volume, privacy-sensitive, and increasingly processed on-device or at the edge —
exactly where a 3B+ model is too expensive. But their document capabilities are
under-characterised: most public leaderboards focus on 3B–70B models.

This report (1) selects eight sub-1B VLMs spanning *generalists*, *edge models* and *OCR
specialists*; (2) defines a document-understanding benchmark suite anchored on **DocVQA,
InfoVQA, ChartQA and OCRBench**, plus a **custom robustness probe**; (3) evaluates them with
metrics that go beyond accuracy — **ANLS / relaxed accuracy / OCRBench score**, plus
**calibration (ECE)** and **robustness retention**; and (4) turns the resulting gap analysis
into a concrete, literature-grounded improvement plan.

**Headline finding (from published reference figures, to be reproduced with the included
pipeline):** the **InternVL family at ~0.9B (InternVL2.5-1B / InternVL3-1B)** is the
strongest document generalist in this size class — DocVQA ≈ 82–85 ANLS, OCRBench ≈ 785–790,
ChartQA ≈ 75 — yet it has a **clear, consistent weak spot on InfoVQA (≈ 54–56)**:
multi-element, layout-and-numeric reasoning over infographics. This — not raw text
recognition — is where the best small model falls short, and it is the target of the
improvement strategy in Part 2.

> **Reproducibility note.** This environment has **no GPU**, so the report does not present
> numbers I ran myself as if they were measured here. The **evaluation pipeline is complete,
> tested, and runnable** (CPU smoke-tested end-to-end; metric code unit-tested); the GPU runs
> are intended for a free Colab/Kaggle T4, for which exact commands are provided. Where I cite
> scores, they are **published reference figures** (model papers / cards) with explicit
> source and caveat, and the comparison table clearly separates "published reference" from
> "reproduced".

---

## Part 1 — Evaluation

### 1. Model selection criteria

I select for **architectural diversity within the <1B budget**, so the comparison explains
*why* one design wins, not just *which* wins. Three archetypes are represented:

| # | Model | Params | Vision encoder + LM | Archetype | Why included |
|---|-------|-------:|---------------------|-----------|--------------|
| 1 | **InternVL2.5-1B** | ~0.94B | InternViT-300M + Qwen2.5-0.5B, dynamic tiling | Document generalist | SoTA-for-size doc/OCR; first-class in VLMEvalKit |
| 2 | **InternVL3-1B** | ~0.94B | InternViT-300M + Qwen2.5-0.5B, native multimodal pretrain | Document generalist | Newer pretraining recipe; head-to-head with 2.5 |
| 3 | **SmolVLM-500M** | 0.5B | SigLIP + SmolLM2, Idefics3, heavy token compression | Edge | Edge-deployment anchor; trained incl. Docmatix |
| 4 | **SmolVLM-256M** | 0.26B | SigLIP + SmolLM2 | Extreme edge | Smallest serious VLM; floor of the size/quality curve |
| 5 | **LLaVA-OneVision-0.5B** | ~0.9B | SigLIP-SO400M + Qwen2-0.5B, AnyRes | General VLM | "General-purpose small VLM" baseline (not doc-specialised) |
| 6 | **GOT-OCR2.0** | ~0.58B | ~80M ViT + Qwen-0.5B decoder | OCR specialist | Pure transcription; tests "OCR ≠ document QA" |
| 7 | **Florence-2-large** | ~0.77B | DaViT + BART enc-dec, task tokens | OCR/detection specialist | Unified task-token model; no free-form VQA |
| 8 | **PaddleOCR-VL-0.9B** | ~0.9B | NaViT-style encoder + ERNIE-4.5-0.3B | Doc-parsing specialist | Newest, most document-specialised release |

**Rationale highlights**

- **Why InternVL ×2.** It is the consensus strongest small document model, and including
  both 2.5 and 3 isolates the effect of the *pretraining recipe* at fixed architecture/size —
  directly relevant to "pretraining data" in the selection criteria.
- **Why two SmolVLM sizes.** They trace the size/quality frontier (256M→500M) and represent
  the edge-deployment use case the task emphasises; SmolVLM's training mix includes
  **Docmatix**, a large document instruction set, making it the strongest *small-and-edge*
  document model and a fair foil to InternVL.
- **Why a general baseline (LLaVA-OV-0.5B).** To quantify the premium document-specialised
  pretraining buys over a strong but general small VLM at the same ~0.9B size.
- **Why two OCR specialists (GOT, Florence-2) and a parser (PaddleOCR-VL).** Document
  understanding is *not* OCR. GOT/Florence-2 transcribe but cannot answer questions; including
  them makes the **"recognition vs. reasoning" gap measurable** rather than asserted, and
  PaddleOCR-VL tests whether a parsing-first design transfers to QA.

> Architecture / pretraining / capability profiles for each model are in **Appendix A**.

### 2. Benchmark selection & design

Document understanding spans *recognition* (read the pixels), *layout* (where things are),
and *reasoning* (combine fields, do arithmetic). No single dataset covers all three, so I use
a **suite** plus a **custom probe**:

| Benchmark | Source (VLMEvalKit / HF) | Primary metric | What it probes | Why it's here |
|-----------|--------------------------|----------------|----------------|---------------|
| **DocVQA** | `lmms-lab/DocVQA` (DocVQA, val) | ANLS | Printed text + layout + KIE on scanned business docs | The canonical document-QA benchmark |
| **InfoVQA** | `lmms-lab/DocVQA` (InfographicVQA, val) | ANLS | Dense multi-element layout, text+chart fusion, numeric reasoning | Hardest layout/reasoning; complements DocVQA |
| **ChartQA** | `lmms-lab/ChartQA` (test; human/aug slices) | Relaxed acc | Reading + arithmetic over plotted data | Structured-visual reasoning |
| **OCRBench** | `echo840/OCRBench` | OCRBench (/1000) | Regular/irregular/handwritten/artistic text, KIE, handwritten math | Fine-grained *recognition* capability |
| **Robustness probe** | derived from DocVQA (ours) | ANLS retention | Capture-quality + terminology stability | The "beyond accuracy" deployment axis |

**Why validation splits.** DocVQA/InfoVQA **test** labels live behind an eval server; I use
**val** so results are self-contained and reproducible. (Published model cards often quote
*test* — a split mismatch I flag explicitly in the comparison table so numbers are not
silently conflated.)

**The custom robustness probe (design decisions).** Real documents are phone photos, faxes,
and re-compressed PDFs, and users phrase questions in domain jargon. Headline ANLS on pristine
scans hides this. The probe takes N DocVQA items and emits a **paired** set — a `clean`
baseline plus controlled perturbations — so failures are *attributable*:

- **downscale** (low-DPI / small photo → small-text legibility)
- **jpeg** quality 18 (heavy re-compression artifacts)
- **blur** (out-of-focus capture)
- **rotate** 5° (skew / handheld tilt)
- **noise** (sensor / photocopier speckle)
- **term_paraphrase** (image untouched; question rewritten with terse domain wording —
  "total"→"aggregate", "how much"→"what is the amount" — a deterministic, rule-based
  terminology stress test)

Robustness is reported as **retention = score(perturbed) / score(clean)** per family, which
separates "accurate" from "accurate *and stable*".

### 3. Evaluation metrics — beyond accuracy

| Metric | Definition | Why it matters for documents |
|--------|------------|------------------------------|
| **ANLS** | 1 − normalised edit distance to best gold; 0 below 0.5 | Official DocVQA/InfoVQA metric; tolerant of minor OCR/format noise (e.g. `$1,200` vs `1200`) without rewarding near-misses |
| **Relaxed accuracy** | Numeric within 5% rel. error, else exact match | Official ChartQA metric; the right notion of "correct" for read-off numbers |
| **OCRBench score** | Gold string ⊆ prediction, summed /1000 | Standard OCRBench scoring; isolates *recognition* from *reasoning* |
| **Calibration — ECE** | Σ |acc(bin) − conf(bin)|·\|bin\|/N over 10 conf. bins | A deployable reader must *know when it is unsure* so low-confidence fields route to human review. Two models with equal accuracy but different ECE are **not** equally useful. (Guo et al., ICML'17) |
| **Robustness retention** | score(perturbed)/score(clean) per family | Production inputs are degraded; retention predicts real-world accuracy better than clean ANLS |
| **Operational** | answer-rate, load time, avg latency/sample | Edge viability: a model that's accurate but 10× slower may be unusable |

Confidence for ECE is the **mean token probability** of the generated answer, read from HF
`generate(output_scores=True)`. Genuinely autoregressive backends (InternVL, SmolVLM,
LLaVA-OV, PaddleOCR-VL) expose it; transcription-only models (GOT, Florence-2) do not, and are
reported as `ECE = n/a` rather than with a fabricated number.

### 4. Experimental setup & reproducibility

- **Hardware.** Free **Google Colab / Kaggle T4 (16 GB)** — every model fits in fp16/bf16; no
  paid hardware required, as the task allows.
- **Decoding (held constant across all models for fairness).** Greedy (`do_sample=False`,
  `num_beams=1`), `max_new_tokens=256`, bf16. Greedy ⇒ deterministic ⇒ reproducible.
- **Preprocessing.** Each model uses **its own native, documented** image pipeline (InternVL
  dynamic tiling up to 12×448px; SmolVLM/LLaVA AnyRes; Florence/GOT task pipelines). Forcing a
  shared resize would unfairly cripple tiling-based models whose document edge *comes from*
  high-res tiling — so "consistency" here means *consistent decoding + consistent scoring*, not
  identical pixels. This choice is documented so it is auditable.
- **Determinism.** Fixed seeds; greedy decoding; pinned package versions (`requirements.txt`);
  predictions cached to `predictions.jsonl` so re-scoring never re-runs the model and runs are
  resumable.
- **Fairness controls.** Same prompt template per task; same sample subset (`--limit`) across
  models; identical metric code for all.

### 5. Software stack

| Tool | Role | Why |
|------|------|-----|
| **PyTorch + HF Transformers** | Model loading/inference | Every candidate ships HF weights; one API (`AutoModel*`, `generate`) covers them via thin adapters |
| **HF Datasets** | Benchmark sourcing | Canonical hosting for DocVQA/InfoVQA/ChartQA/OCRBench |
| **VLMEvalKit** | Cross-check / standard harness | The task's reference harness; we mirror its dataset choices & metrics, and provide a thin wrapper. We add a *self-contained* pipeline on top because VLMEvalKit reports accuracy/ANLS but **not calibration or our robustness retention** — the metrics that make this analysis "beyond accuracy" |
| **PEFT (LoRA)** | Part-2 fine-tuning PoC | Parameter-efficient adaptation fits the sub-1B/edge story and a T4 budget |
| **Pillow / NumPy / torchvision** | Image I/O + perturbations + tiling transforms | Lightweight, ubiquitous |

**Why a custom pipeline *and* VLMEvalKit.** VLMEvalKit is the right standard for headline
accuracy and is referenced for cross-checking. But calibration (ECE) and the paired robustness
probe are not in its metric set, and they are central to the document deployment story — hence
the small, tested `docvlm_eval` package, whose single `evaluate.py` "loads any model, runs the
benchmark, outputs per-model scores" exactly as the PoC requires.

---

## Part 2 — Knowledge gap & improvement strategy

### 1. Knowledge-gap analysis (evidence-based)

Using **published reference figures** for the sub-1B models (sources & caveats in the
comparison table) and the structure of the benchmark suite, three gaps emerge. The
improvement strategy targets the **best model, InternVL2.5-1B**, against its own weakest axis.

**Gap A — InfoVQA ≪ DocVQA (the primary gap).** Across *every* model the InfoVQA score sits
far below DocVQA (InternVL2.5-1B: **56.0 vs 84.8**; InternVL3-1B: 53.7 vs 81.9; LLaVA-OV-0.5B:
46.3 vs 73.7). InfoVQA is not harder to *read* — it is harder to *reason over*: dense
infographics demand (i) parsing many spatially-scattered elements, (ii) fusing text with
embedded charts/icons, and (iii) multi-step numeric reasoning. The 0.5B language model is the
bottleneck: recognition is solved, *layout-grounded reasoning* is not.

**Gap B — recognition ≠ reasoning.** OCR specialists (GOT, Florence-2) are built for
transcription and report **no** DocVQA/InfoVQA/OCRBench-VQA scores because they cannot answer
questions; PaddleOCR-VL is evaluated only on parsing (OmniDocBench edit-distance), not VQA.
This confirms that pushing OCR quality alone will *not* close the document-QA gap — the deficit
is in reasoning over recognised content, exactly where Gap A points.

**Gap C — reliability is unmeasured by leaderboards.** No public card reports **calibration**
or **degraded-input robustness** for these models, yet both decide deployability. Small LMs are
known to be over-confident; our pipeline measures ECE and per-perturbation retention to expose
it. (These are produced by running the provided pipeline; they are the columns leaderboards
omit.)

**Why InternVL2.5-1B is the target.** It is the strongest published small document model
(DocVQA 84.8, OCRBench 785, ChartQA 75.9, InfoVQA 56.0). Its DocVQA/OCRBench/ChartQA are
near-saturated for the size, so the **highest-leverage, lowest-risk** improvement is its worst
axis — **InfoVQA / layout-grounded numeric reasoning** (Gap A) — while hardening reliability
(Gap C).

### 2. Improvement strategy (concrete, justified)

A staged plan, each step tied to a gap and to literature, all runnable on a single T4 — and
backed by the **LoRA fine-tuning scaffold already in this repo** (`src/ocr_ft`, `scripts/
finetune_lora.py`).

**Step 1 — Targeted LoRA SFT on layout-reasoning data (attacks Gap A).**
Parameter-efficient **LoRA** (Hu et al., 2021) / **QLoRA** (Dettmers et al., 2023) on the LM +
projector, freezing InternViT. Train on a *reasoning-heavy* document mix: **InfographicVQA
train**, **DocVQA train**, **ChartQA**, **Docmatix** (the doc instruction set behind SmolVLM),
and **TabMWP/PlotQA-style** numeric items. Rationale: the recognition stack is already strong;
LoRA cheaply re-weights the *reasoning* pathway toward multi-element/numeric questions without
catastrophic forgetting, and fits a T4. *Keep dynamic high-res tiling on* — it is what lets a
0.5B LM see small infographic text.

**Step 2 — Reasoning distillation from a larger teacher (amplifies Gap A).**
Generate **chain-of-thought rationales** for InfoVQA/ChartQA train questions with a larger
teacher (InternVL-8B/26B or a frontier VLM) and fine-tune the 1B student on
(image, question → rationale → answer). Sequence-level KD / rationale distillation transfers
*reasoning procedure*, not just answers, and is well-suited to small students (Hinton et al.,
2015; Hsieh et al., "Distilling step-by-step", 2023). This directly addresses the multi-step
numeric reasoning that InfoVQA exposes.

**Step 3 — Robustness-aware augmentation (attacks Gap C / retention).**
Augment SFT with the **same perturbations as the robustness probe** (downscale/jpeg/blur/
rotate/noise) and **terminology paraphrases**. Standard augmentation theory + the paired probe
give a direct read on whether retention improves. Keep augmented and clean copies so clean
accuracy is preserved.

**Step 4 — Calibration (attacks Gap C / ECE).**
Post-hoc **temperature scaling** on a held-out doc set (Guo et al., 2017) — one parameter,
no accuracy change, large ECE reduction — optionally with **confidence-aware decoding** so the
deployed reader can abstain/route low-confidence fields. Re-measure ECE with the pipeline.

**Why this combination.** LoRA+QLoRA = T4-feasible and forgetting-safe; distillation injects
the reasoning the small LM lacks; augmentation + temperature scaling convert clean-set wins
into *deployable* wins. The plan is *surgical* — it spends capacity on the one axis the
evidence says is weak (InfoVQA/layout reasoning), not on already-saturated OCR.

### 3. Expected outcomes & measurement

| Axis | Baseline (published ref.) | Target after plan | Measured by |
|------|---------------------------:|------------------:|-------------|
| InfoVQA (val ANLS) | ~56 | **+6–10 ANLS** | `evaluate.py` on InfoVQA |
| ChartQA (relaxed) | ~76 | +2–4 | `evaluate.py` on ChartQA |
| DocVQA (val ANLS) | ~85 | **no regression** (±1) | `evaluate.py` on DocVQA |
| Robustness worst-case retention | TBD (pipeline) | **+0.1–0.2** | robustness probe retention |
| Calibration (ECE) | TBD (pipeline) | **halved** | ECE in `summary.json` |

Expected magnitudes are deliberately modest and InfoVQA-focused because that is where the
headroom is; the success criterion is **InfoVQA up materially with DocVQA/OCRBench held**,
plus measurable reliability gains. Everything is verified by re-running the *same* pipeline,
so before/after is apples-to-apples.

---

## Appendix A — Model profiles

- **InternVL2.5-1B / InternVL3-1B** (`OpenGVLab/InternVL2_5-1B`, `OpenGVLab/InternVL3-1B`,
  MIT). InternViT-300M vision encoder + Qwen2.5-0.5B LM via an MLP pixel-shuffle projector,
  with **dynamic high-resolution tiling** (up to 12× 448px crops + thumbnail). Trained on
  large multimodal corpora with heavy OCR/document/chart data; v3 adds a native multimodal
  pretraining recipe. Strong OCR/doc/chart for the size.
- **SmolVLM-256M / 500M** (`HuggingFaceTB/SmolVLM-*-Instruct`, Apache-2.0). SigLIP vision
  encoder + SmolLM2 LM in the Idefics3 framework, with aggressive pixel-shuffle token
  compression for on-device use; training mix includes **Docmatix** (document instructions).
- **LLaVA-OneVision-0.5B** (`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, Apache-2.0).
  SigLIP-SO400M + Qwen2-0.5B with AnyRes tiling; strong general single-image/multi-image/video
  VLM, not document-specialised.
- **GOT-OCR2.0** (`stepfun-ai/GOT-OCR-2.0-hf`, Apache-2.0). ~80M ViT encoder + Qwen-0.5B
  decoder trained end-to-end for **transcription** (plain/formatted/markdown). No free-form
  question interface → OCR-specialist baseline.
- **Florence-2-large** (`microsoft/Florence-2-large`, MIT). DaViT encoder + BART-style
  enc-dec trained on FLD-5B with a **task-token** interface (`<OCR>`, `<OD>`, …). No `<DocVQA>`
  token → driven with `<OCR>`; specialist, not conversational.
- **PaddleOCR-VL-0.9B** (`PaddlePaddle/PaddleOCR-VL`). NaViT-style dynamic-resolution encoder
  + ERNIE-4.5-0.3B LM, purpose-built for full-page **document parsing** (text/table/formula/
  chart, reading order) across 100+ languages; evaluated by its authors on parsing
  (OmniDocBench edit-distance), not VQA.

## Appendix B — Selected references

- Mathew et al. *DocVQA: A Dataset for VQA on Document Images.* WACV 2021.
- Mathew et al. *InfographicVQA.* WACV 2022.
- Masry et al. *ChartQA.* ACL Findings 2022.
- Liu et al. *OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models.* 2023.
- Chen et al. *InternVL* / *InternVL 2.5* (arXiv:2412.05271) / *InternVL3* (arXiv:2504.10479).
- Marafioti et al. *SmolVLM.* arXiv:2504.05299.
- Li et al. *LLaVA-OneVision.* arXiv:2408.03326.
- Wei et al. *General OCR Theory (GOT-OCR2.0).* arXiv:2409.01704.
- Xiao et al. *Florence-2.* arXiv:2311.06242 (CVPR 2024).
- *PaddleOCR-VL Technical Report.* arXiv:2510.14528.
- Hu et al. *LoRA.* arXiv:2106.09685. Dettmers et al. *QLoRA.* arXiv:2305.14314.
- Hinton et al. *Distilling the Knowledge in a Neural Network.* 2015. Hsieh et al.
  *Distilling Step-by-Step.* ACL 2023.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.

*Published scores quoted in this report are self-reported by model authors (papers/cards),
evaluated via VLMEvalKit/OpenCompass; split conventions (val vs test) and scale conventions
(OCRBench /1000 vs /100) differ between sources and are flagged in the comparison table.*
