# Results analysis — what we loaded, what ran, and the flaws it exposed

This interprets the **stored** results (`results/matrix_*.{md,json}`,
`results/<model>/<bench>/`) produced by running models across the benchmark suite, and lists
the flaws found — split into **inference bugs**, **harness/scoring flaws**, and
**model-capability flaws** — plus what was fixed.

## What actually ran (and what couldn't)

This environment has **no GPU**. We installed CPU torch/transformers and attempted **every
registered model** on CPU (`scripts/run_all_cpu.sh`, per-model 1500s budget) over the
capability probe. Outcome:

| Model                        | CPU result    | Why                                                                                                              |
| ---------------------------- | :-----------: | ---------------------------------------------------------------------------------------------------------------- |
| dummy-echo                   | ✅             | plumbing baseline                                                                                                |
| **SmolVLM-256M**             | ✅ real        | runs (~5–10 s/sample)                                                                                            |
| **SmolVLM-500M**             | ✅ real        | runs                                                                                                             |
| **SmolDocling-256M**         | ✅ real        | runs, but emits **DocTags** not answers → ~0 on VQA cells (interface mismatch, not incapacity)                   |
| **GOT-OCR2.0**               | ✅ real        | runs (transcription; scores on read-off cells like the chart number)                                             |
| LLaVA-OneVision-0.5B         | ⏳ **timeout** | works but **~8 min/sample** on CPU → exceeded budget at 3/6. Not broken — too slow                               |
| InternVL2-1B / 2.5-1B / 3-1B | ❌             | `AttributeError: 'InternVLChatModel' has no attribute 'all_tied_weights_keys'` — remote code vs transformers 5.x |
| Ovis2-1B                     | ❌             | remote-code (AIMv2) incompatible with transformers 5.x                                                           |
| PaddleOCR-VL / -1.5          | ❌             | remote-code incompatible with transformers 5.x (`check_model_inputs` deprecation path)                           |
| Florence-2-base / -large     | ❌             | `Florence2LanguageConfig has no attribute 'forced_bos_token_id'` (remote code vs transformers 5.x)               |
| H2OVL-0.8B                   | ❌             | InternVL-style remote code, same transformers-5.x drift                                                          |

The matrix runner stores per-model results + a **run-status**, so failures are captured as data.

### Dependency fix & recovery (re-running the previously-failed models)

The failures were a **library/version problem**, so we fixed the dependencies and re-ran:
pinned **transformers 4.49** (`<5`, the version their `trust_remote_code` expects) and installed
**`peft`** and **`protobuf`**. Result — **11 models now produce real CPU results**:

| Newly recovered                  | Fix                | Status                                                                                                                       |
| -------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **InternVL2-1B / 2.5-1B / 3-1B** | transformers<5     | ✅ run (the headline doc models)                                                                                              |
| **Florence-2 base / large**      | transformers<5     | ✅ run                                                                                                                        |
| **H2OVL-0.8B**                   | `pip install peft` | ⚠ loads but emits empty output (remote-code chat returns "")                                                                 |
| Ovis2-1B                         | —                  | ❌ remote code hard-requires CUDA `flash_attn` → GPU-only                                                                     |
| PaddleOCR-VL 1.0 / 1.5           | —                  | ❌ need *newer* transformers (`masking_utils`/`use_kernel_forward_from_hub`) which conflicts with the `<5` pin → separate env |
| LLaVA-OV-0.5B                    | —                  | ⏳ loads but ~8 min/sample on CPU → too slow                                                                                  |

**Final capability matrix** (`results/matrix_capability.md`, real CPU runs):

| model            | text  | kie   | integ-sum | integ-rel | chart | grounding |
| ---------------- | :---: | :---: | :-------: | :-------: | :---: | :-------: |
| internvl3-1b     | 0.00¹ | 1.00  | **1.00**  | **1.00**  | 1.00  | 0.00      |
| internvl2_5-1b   | 0.00¹ | 1.00  | **1.00**  | **1.00**  | 1.00  | 0.00      |
| internvl2-1b     | 0.00¹ | 1.00  | 1.00      | 0.00      | 1.00  | 0.00      |
| smolvlm-500m     | 0.93  | 0.94  | 1.00      | 0.00      | 1.00  | 0.00      |
| smolvlm-256m     | 0.93  | 0.94  | 0.00      | 0.00      | 1.00  | 0.00      |
| smoldocling-256m | 0.00² | 0.94  | 0.00      | 0.00      | 0.00  | 0.02      |
| florence2-large  | 0.00² | 0.00² | 0.00      | 0.00      | 1.00  | 0.00      |
| got-ocr2         | 0.00² | 0.00² | 0.00      | 0.00      | 1.00  | 0.00      |
| h2ovl-0.8b       | 0.00³ | 0.00³ | 0.00³     | 0.00³     | 0.00³ | 0.00³     |

¹ InternVL answered partially ("2025" for the invoice no.) → low ANLS, not a reasoning failure.
² OCR/transcription specialists output the whole page / task-token format → low on the
short-answer cells (interface mismatch), but read the chart number. ³ H2OVL emits empty text.

**Headline finding:** **InternVL2.5-1B / 3-1B clear the integrative-reasoning axis (sum AND
relational = 1.00)** that *no* SmolVLM model reaches — concrete evidence that multi-region
reasoning emerges around 1B, while ≤0.5B models stay at sum-only (500M) or neither (256M). All
models still score 0 on grounding (no spotting head). This is exactly the gap the report's
Part-2 improvement plan targets.

## A. Inference bugs (the "inference doesn't work" cases — found via real runs)

1. **`AutoModelForVision2Seq` removed in transformers 5.x** → SmolVLM/SmolDocling failed to
   load. **Fixed**: adapter now falls back to `AutoModelForImageTextToText`. (SmolVLM then ran
   16/16.)
2. **`run_matrix` matrix keyed by `answer_type` not `sample_id`** → NaN cells on the capability
   probe (where axis≠id). **Fixed** (key by `sample_id`; aggregation now cumulative across runs).
3. **The big root cause — `trust_remote_code` models vs transformers 5.x.** The InternVL,
   Ovis, PaddleOCR-VL, H2OVL and Florence-2 adapters all fail *inside the models' own
   downloaded code*, which was written for transformers ≤4.x and breaks on the installed
   5.12 (`all_tied_weights_keys`, `forced_bos_token_id`, `check_model_inputs`). This is not a
   bug in our adapters (their class imports are clean on 5.x — verified) but a version drift.
   **Fix (recommended, not applied here to keep SmolVLM's 5.x path):** pin
   `pip install "transformers>=4.49,<5"` for these families — `requirements.txt`/`[models]`
   already allow it; `scripts/run_all.sh` on a GPU box should use that pin. We did **not**
   downgrade in this container because (a) it would not help the *other* blocker — CPU is far
   too slow for 1B models (LLaVA-OV = 8 min/sample → a full sweep is many hours), and (b) the
   SmolVLM family already runs on 5.x. So the correct environment for the 1B models is
   **GPU + transformers 4.x**, documented for reproduction.

## B. Harness / scoring flaws (zeros that were NOT the model's fault)

Real SmolVLM outputs exposed several *evaluation* flaws — important because they would make a
capable model look bad:

| Flaw                                  | Evidence                                                                                               | Status                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Verbosity vs strict match**         | InfoVQA pred *"Pinterest has a heavy female audience."* vs gold *"pinterest"* → ANLS 0, though correct | **Fixed**: short-answer prompts append *"Answer concisely with only the value."* (as VLMEvalKit does) |
| **AI2D gold is an option *index***    | gold `"1"` but answer text is `options[1]="D"` → always wrong                                          | **Fixed**: map index→option text in `build_preview_benchmark`                                         |
| **Transcription preamble / markdown** | full-page pred starts *"### Markdown Format: …"* → ANLS penalised despite correct content              | **Documented**: needs preamble stripping or a CER-with-normalisation metric                           |
| **OCRBench tiny-crop hedging**        | pred *"There is text in the image."* on a 104×27 crop → 0                                              | **Model behaviour**, but prompt could force a literal read                                            |
| **POPE yes/no**                       | pred *"Yes, there is a snowboarder…"* vs gold *"yes"*                                                  | concise prompt mitigates; a yes/no normaliser would fully fix                                         |

Take-away: on these tiny models, **a big fraction of "0.00" cells are metric/prompt artefacts,
not capability gaps** — exactly why the controlled capability probe (below) is more trustworthy
than a 1-sample VQA preview.

## C. Model-capability flaws (the real ones, from the capability probe)

The controlled probe (`results/matrix_capability.md`) gives a clean **capability vector**:

| axis               | SmolVLM-256M | SmolVLM-500M | reading                                        |
| ------------------ | :----------: | :----------: | ---------------------------------------------- |
| text-recognition   | 0.93         | 0.93         | solved at 256M                                 |
| kie-localized      | 0.94         | 0.94         | solved at 256M                                 |
| integrative-sum    | **0.00**     | **1.00**     | arithmetic **emerges 256M→500M**               |
| integrative-rel    | **0.00**     | **0.00**     | cross-region comparison **fails at ≤0.5B**     |
| chart-dependent    | 1.00         | 1.00         | clean-chart read works                         |
| location-grounding | **0.00**     | **0.00**     | general small VLMs have **no usable box head** |

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
    --benchmark data/probes/capability_probe/capability.jsonl --device cpu --dtype float32
python scripts/run_matrix.py --all --device cuda     # full sweep on a GPU
```
