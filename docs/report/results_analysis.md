# Results analysis — what we loaded, what ran, and the flaws it exposed

This interprets the **stored** results (`docs/results/matrix_*.{md,json}`,
`docs/results/<model>/<bench>/`) produced by running models across the benchmark suite, and
lists the flaws found — split into **inference bugs**, **harness/scoring flaws**, and
**model-capability flaws** — plus what was fixed.

> **Taxonomy note.** Scores below are reported against the capability-axis catalogue in
> [`capability_axes.md`](capability_axes.md): **T** (text) = T1 text-recognition, T2 KIE-localized;
> **H** (hybrid reasoning) = H1 content-reasoning *sum*, H2 content-reasoning *compare*, H3
> chart-value; **L** (location) = L1 grounding. The spatial/context signals (L2–L4 / H4–H7) come
> from the separate spatial-context probe (§E). When the taxonomy was renamed, every model's
> **cached `predictions.jsonl` was re-scored against the new codes without re-running the model**
> (`scripts/run_matrix.py --rescore`), so the numbers here reflect the current taxonomy exactly.

## What actually ran (and what couldn't)

This environment has **no GPU**. We installed CPU torch + **transformers 4.49** (`<5`, the version
the models' `trust_remote_code` expects) plus `peft`/`protobuf`, and attempted **every registered
model** on CPU over the capability probe. The matrix runner stores per-model results + a
**run-status**, so failures are captured as data.

**11 models have cached, committed `predictions.jsonl` for the capability probe** and are therefore
fully re-scoreable against the current taxonomy **without a GPU**:

| Model            | params(M) | capability run                                                              |
| ---------------- | --------: | --------------------------------------------------------------------------- |
| SmolVLM-256M     |       256 | ✅ real (~5–10 s/sample)                                                     |
| SmolVLM-500M     |       500 | ✅ real                                                                      |
| SmolDocling-256M |       256 | ✅ real — emits **DocTags**, so short-answer cells read ~0 (interface, not incapacity) |
| GOT-OCR2.0       |       580 | ✅ real (transcription; scores on read-off cells like the chart number)      |
| Florence-2-base  |       230 | ✅ real                                                                      |
| Florence-2-large |       770 | ✅ real                                                                      |
| InternVL2-1B     |       938 | ✅ real                                                                      |
| InternVL2.5-1B   |       938 | ✅ real                                                                      |
| InternVL3-1B     |       938 | ✅ real                                                                      |
| H2OVL-0.8B       |       800 | ⚠ loads but emits empty output (remote-code chat returns "")                |
| LLaVA-OV-0.5B    |       894 | ⏳ **partial** — ~8 min/sample on CPU → only **3/6** samples (T1/T2/H1) ran   |

**3 models still need a fresh GPU extraction** — they have *no* cached predictions, so they are
absent from the matrix and must be run before they appear:

| Model           | params(M) | why it needs GPU                                                                                    |
| --------------- | --------: | --------------------------------------------------------------------------------------------------- |
| Ovis2-1B        |      1000 | remote code hard-requires CUDA `flash_attn` → GPU-only                                              |
| PaddleOCR-VL    |       900 | needs *newer* transformers (`masking_utils`/`use_kernel_forward_from_hub`) conflicting with the `<5` pin → separate env |
| PaddleOCR-VL-1.5 |       900 | same as PaddleOCR-VL                                                                                |

To extract them, run the checkpointed sweep on a GPU box (it pulls, runs each model, commits +
pushes predictions/summary so a GPU-limit interruption can resume):

```bash
bash scripts/run_checkpointed.sh ovis2-1b paddleocr-vl paddleocr-vl-1.5
```

**Capability matrix** (`docs/results/matrix_capability.md`, real CPU runs, re-scored to T/L/H):

| model            | params(M) |  T1   |  T2   |  H1   |  H2   |  H3   |  L1   |
| ---------------- | --------: | :---: | :---: | :---: | :---: | :---: | :---: |
| internvl3-1b     |       938 | 0.00¹ | 1.00  | **1.00** | **1.00** | 1.00 | 0.00  |
| internvl2_5-1b   |       938 | 0.00¹ | 1.00  | **1.00** | **1.00** | 1.00 | 0.00  |
| internvl2-1b     |       938 | 0.00¹ | 1.00  | 1.00  | 0.00  | 1.00  | 0.00  |
| smolvlm-500m     |       500 | 0.93  | 0.94  | 1.00  | 0.00  | 1.00  | 0.00  |
| smolvlm-256m     |       256 | 0.93  | 0.94  | 0.00  | 0.00  | 1.00  | 0.00  |
| llava-ov-0.5b    |       894 | 0.93  | 0.94  | 1.00  |  —⁴   |  —⁴   |  —⁴   |
| florence2-large  |       770 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| got-ocr2         |       580 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| smoldocling-256m |       256 | 0.00² | 0.94  | 0.00  | 0.00  | 0.00  | 0.02  |
| h2ovl-0.8b       |       800 | 0.00³ | 0.00³ | 0.00³ | 0.00³ | 0.00³ | 0.00³ |
| florence2-base   |       230 | 0.00² | 0.00² | 0.00  | 0.00  | 0.00  | 0.00  |
| **ovis2-1b**     |      1000 | pending fresh GPU extraction — see `run_checkpointed.sh` above |||||
| **paddleocr-vl** |       900 | pending fresh GPU extraction |||||
| **paddleocr-vl-1.5** |   900 | pending fresh GPU extraction |||||

¹ InternVL answered partially ("2025" for the invoice no.) → low ANLS on T1, not a reasoning
failure. ² OCR/transcription specialists output the whole page / task-token format → low on the
short-answer cells (interface mismatch), but read the chart number. ³ H2OVL emits empty text.
⁴ LLaVA-OV ran only 3/6 samples on CPU (timeout); H2/H3/L1 were never executed (shown `—`, not 0).

**Headline finding:** **InternVL2.5-1B / 3-1B clear the hybrid-reasoning axes (H1 sum AND H2
compare = 1.00)** that *no* SmolVLM reaches — concrete evidence that multi-region content
reasoning emerges around 1B, while ≤0.5B models stay at sum-only (500M) or neither (256M). **All
models still score ~0 on L1 grounding** (no spotting head). This is exactly the gap the report's
Part-2 improvement plan targets.

## A. Inference bugs (the "inference doesn't work" cases — found via real runs)

1. **`AutoModelForVision2Seq` removed in transformers 5.x** → SmolVLM/SmolDocling failed to
   load. **Fixed**: adapter now falls back to `AutoModelForImageTextToText`.
2. **`run_matrix` matrix keyed by `answer_type` not `sample_id`** → NaN cells on the capability
   probe (where axis≠id). **Fixed** (key by `sample_id`; aggregation is cumulative across runs).
3. **The big root cause — `trust_remote_code` models vs transformers 5.x.** The InternVL, Ovis,
   PaddleOCR-VL, H2OVL and Florence-2 adapters all failed *inside the models' own downloaded
   code*, written for transformers ≤4.x (`all_tied_weights_keys`, `forced_bos_token_id`,
   `check_model_inputs`). Not a bug in our adapters — a version drift. **Fix:** pin
   `transformers>=4.49,<5` for these families (done in this container, recovering 9 of them on
   CPU). PaddleOCR-VL needs the opposite (newer transformers) → a separate env, hence GPU-only.

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
not capability gaps** — exactly why the controlled capability probe (above) is more trustworthy
than a 1-sample VQA preview.

## C. Model-capability flaws (the real ones, from the capability probe)

The controlled probe gives a clean **capability vector** (T/L/H codes from `capability_axes.md`):

| axis                          | SmolVLM-256M | SmolVLM-500M | reading                                        |
| ----------------------------- | :----------: | :----------: | ---------------------------------------------- |
| T1 text-recognition           |     0.93     |     0.93     | solved at 256M                                 |
| T2 kie-localized              |     0.94     |     0.94     | solved at 256M                                 |
| H1 content-reasoning (sum)    |   **0.00**   |   **1.00**   | arithmetic **emerges 256M→500M**               |
| H2 content-reasoning (compare)|   **0.00**   |   **0.00**   | cross-region comparison **fails at ≤0.5B**     |
| H3 chart-value                |     1.00     |     1.00     | clean-chart read works                         |
| L1 location-grounding         |   **0.00**   |   **0.00**   | general small VLMs have **no usable box head** |

**Flaws / limits of the best small models tested:**
- **Relational reasoning** (H2, "which is largest?") fails — both answer the *first* item. The
  bottleneck is multi-region comparison, not OCR.
- **Numeric aggregation** (H1) is fragile: 256M miscomputes a 3-number sum; only 500M is reliable.
- **No localisation** (L1): neither returns a usable bounding box → for field-localisation /
  parsing, a **spotting-capable** model (PaddleOCR-VL, Florence-2, GOT) is required (see
  `capability_axes.md`).

## D. Cross-model insights

Auto-synthesised in [`insights.md`](insights.md) (`scripts/build_insights.py`): capability
leaders per axis, reasoning-emergence-vs-size, the grounding gap, and the efficiency frontier.
Regenerate after any re-score or fresh sweep.

## E. Spatial & context signals (the L2–L4 / H4–H7 probe)

The spatial-context probe is scored with **shortcut-robust criteria** (`analyze_probe_signals.py`
→ `docs/results/probe_signals.md`): a model must clear a control pair (counterfactual / distractor
/ position-bias), not just hit raw accuracy. Cached predictions exist for the two SmolVLM sizes:

| model        | L2 (abs) | L3 (rel) | L4 (box) | H4 (consist.) | H5 (absence) | H6 (distractor) | H7 (xref) |
| ------------ | :------: | :------: | :------: | :-----------: | :----------: | :-------------: | :-------: |
| smolvlm-256m |   FAIL   |   FAIL   |   FAIL   |     FAIL      |     FAIL     |     **PASS**    |   FAIL    |
| smolvlm-500m | **PASS** |   FAIL   |   FAIL   |     FAIL      |     FAIL     |     **PASS**    |   FAIL    |

Reading: even at 500M the only robust signals are coarse **absolute quadrant** (L2) and
**distractor disambiguation** (H6). Relative position (L3), box-tracking (L4), consistency (H4),
anti-hallucination on absent fields (H5) and cross-reference (H7) all fail their controls — these
are the spatial/context capabilities Part-2 must inject. The remaining models need a GPU run on
this probe to fill the table.

## F. Honest limitations of this run

- The **all_preview** matrix is **one sample per benchmark** → a *sanity/plumbing* matrix, not
  leaderboard accuracy; treat trends, not absolute numbers. The capability probe has one sample
  per axis too — directional, not a leaderboard.
- **Capability numbers are CPU runs re-scored to the current taxonomy.** No model was re-run for
  the recode; the predictions are frozen and only the scorer changed.
- **3 models (Ovis2-1B, PaddleOCR-VL, PaddleOCR-VL-1.5) and the full spatial/context table for the
  9 non-SmolVLM models still need a GPU.** Their adapters are import-clean; only runtime numbers
  are missing. Use `scripts/run_checkpointed.sh` on free GPU.

## How to reproduce / extend
```bash
python scripts/make_capability_probe.py               # controlled capability probe (+ images)
python scripts/make_spatial_context_probe.py          # spatial/context probe (+ images)

# Re-score every model's CACHED predictions against the current taxonomy (no GPU, no model load):
python scripts/run_matrix.py --rescore \
    --benchmark data/probes/capability_probe/capability.jsonl \
    --models florence2-base florence2-large got-ocr2 h2ovl-0.8b internvl2-1b internvl2_5-1b \
             internvl3-1b llava-ov-0.5b smoldocling-256m smolvlm-256m smolvlm-500m
python scripts/run_matrix.py --rescore \
    --benchmark data/probes/spatial_context_probe/probe.jsonl --models smolvlm-256m smolvlm-500m
python scripts/analyze_probe_signals.py --probe probe   # -> docs/results/probe_signals.md
python scripts/build_insights.py                        # -> docs/report/insights.md

# Fresh extraction of the GPU-only models (checkpointed, resumable):
bash scripts/run_checkpointed.sh ovis2-1b paddleocr-vl paddleocr-vl-1.5
```
