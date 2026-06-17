# Results analysis — what we loaded, what ran, and the flaws it exposed

This interprets the **stored** results (`results/matrix_*.{md,json}`,
`results/<model>/<bench>/`) produced by running models across the benchmark suite, and lists
the flaws ("흠") found — split into **inference bugs**, **harness/scoring flaws**, and
**model-capability flaws** — plus what was fixed.

## What actually ran (and what couldn't)

This environment has **no GPU**. To get *real* numbers (and to surface real inference bugs) we
installed CPU torch/transformers and ran the smallest models for real:

| Model | all_preview (16) | capability probe (6) | Note |
|---|:---:|:---:|---|
| dummy-echo | ✅ | ✅ | plumbing baseline (scores 0) |
| **SmolVLM-256M** | ✅ real | ✅ real | runs on CPU |
| **SmolVLM-500M** | ✅ real | ✅ real | runs on CPU |
| Florence-2-base | ❌ | ❌ | **inference blocked** (see below) |
| InternVL2/2.5/3-1B, Ovis2-1B, H2OVL, LLaVA-OV, GOT, PaddleOCR-VL(1.0/1.5), SmolDocling, Florence-2-large | ⏳ GPU | ⏳ GPU | too slow/large for this CPU box; run via `scripts/run_all.sh` on Colab/Kaggle |

The matrix runner stores per-model results and records a **run-status per model**, so failures
are captured as data instead of aborting the sweep.

## A. Inference bugs (the "인퍼런스 안 되는 부분" — found via real runs)

1. **`AutoModelForVision2Seq` removed in transformers 5.x** → SmolVLM/SmolDocling failed to
   load. **Fixed**: adapter now falls back to `AutoModelForImageTextToText`. (SmolVLM then ran
   16/16.)
2. **Florence-2 remote code incompatible with transformers 5.x**:
   `Florence2LanguageConfig object has no attribute 'forced_bos_token_id'` at generate.
   **Not yet fixed** — it is upstream remote-code drift; options: pin `transformers<5` for
   Florence, or patch the generation config. Recorded as a known-blocked model.
3. **`run_matrix` matrix keyed by `answer_type` not `sample_id`** → NaN cells on the capability
   probe (where axis≠id). **Fixed** (key by `sample_id`; aggregation now also cumulative across
   separate runs).

## B. Harness / scoring flaws (zeros that were NOT the model's fault)

Real SmolVLM outputs exposed several *evaluation* flaws — important because they would make a
capable model look bad:

| Flaw | Evidence | Status |
|---|---|---|
| **Verbosity vs strict match** | InfoVQA pred *"Pinterest has a heavy female audience."* vs gold *"pinterest"* → ANLS 0, though correct | **Fixed**: short-answer prompts append *"Answer concisely with only the value."* (as VLMEvalKit does) |
| **AI2D gold is an option *index*** | gold `"1"` but answer text is `options[1]="D"` → always wrong | **Fixed**: map index→option text in `build_preview_benchmark` |
| **Transcription preamble / markdown** | full-page pred starts *"### Markdown Format: …"* → ANLS penalised despite correct content | **Documented**: needs preamble stripping or a CER-with-normalisation metric |
| **OCRBench tiny-crop hedging** | pred *"There is text in the image."* on a 104×27 crop → 0 | **Model behaviour**, but prompt could force a literal read |
| **POPE yes/no** | pred *"Yes, there is a snowboarder…"* vs gold *"yes"* | concise prompt mitigates; a yes/no normaliser would fully fix |

Take-away: on these tiny models, **a big fraction of "0.00" cells are metric/prompt artefacts,
not capability gaps** — exactly why the controlled capability probe (below) is more trustworthy
than a 1-sample VQA preview.

## C. Model-capability flaws (the real ones, from the capability probe)

The controlled probe (`results/matrix_capability.md`) gives a clean **capability vector**:

| axis | SmolVLM-256M | SmolVLM-500M | reading |
|---|:---:|:---:|---|
| text-recognition | 0.93 | 0.93 | solved at 256M |
| kie-localized | 0.94 | 0.94 | solved at 256M |
| integrative-sum | **0.00** | **1.00** | arithmetic **emerges 256M→500M** |
| integrative-rel | **0.00** | **0.00** | cross-region comparison **fails at ≤0.5B** |
| chart-dependent | 1.00 | 1.00 | clean-chart read works |
| location-grounding | **0.00** | **0.00** | general small VLMs have **no usable box head** |

**Flaws / limits of the best small models tested:**
- **Relational reasoning** ("which is largest?") fails — both answer the *first* item. The
  bottleneck is multi-region comparison, not OCR.
- **Numeric aggregation** is fragile: 256M miscomputes a 3-number sum; only 500M is reliable.
- **No localisation**: neither can return a bounding box → for field-localisation / parsing,
  a **spotting-capable** model (PaddleOCR-VL, Florence-2, GOT) is required (see
  `report/capability_axes.md` §5).

## D. Honest limitations of this run

- The **all_preview** matrix is **one sample per benchmark** → a *sanity/plumbing* matrix, not
  leaderboard accuracy; treat trends, not absolute numbers.
- **11 of 14 models** still need a GPU; their adapters are written + import-clean on
  transformers 5.x but their *runtime* correctness is verified only for the SmolVLM family here.
  Full numbers come from `scripts/run_all.sh` on free GPU.
- Florence-2 inference is blocked on transformers 5.x remote code (item A2).

## How to reproduce / extend
```bash
python scripts/build_preview_benchmark.py            # cross-benchmark preview set
python scripts/make_capability_probe.py              # controlled capability probe
python scripts/run_matrix.py --models smolvlm-256m smolvlm-500m \
    --benchmark data/benchmarks/capability_probe/capability.jsonl --device cpu --dtype float32
python scripts/run_matrix.py --all --device cuda     # full sweep on a GPU
```
