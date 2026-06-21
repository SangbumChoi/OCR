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

## What actually ran (full GPU sweep)

The full comparison was run on a **free T4** (`scripts/run_full_comparison.sh`, captured in
[`../../notebooks/colab_full_comparison.ipynb`](../../notebooks/colab_full_comparison.ipynb)):
**all 19 registered models** produced committed results across the capability probe, the
spatial/context probe and the proposed custom-eval, scored against the T/L/H taxonomy. Chat VLMs
ran at transformers 4.49 and PaddleOCR-VL at 4.57; predictions are cached so re-scoring never
re-runs a model. (An earlier interim run on this CPU-only container covered 11 models and is
superseded by the GPU numbers below.)

**Capability matrix** (`docs/results/matrix_capability.md`, measured on a T4, T1=cap_text,
T2=cap_kie, H1=cap_integ_sum, H2=cap_integ_rel, H3=cap_chart, L1=cap_ground):

| model            | params(M) |  T1   |  T2   |  H1   |  H2   |  H3   |  L1   |
| ---------------- | --------: | :---: | :---: | :---: | :---: | :---: | :---: |
| lfm2_5-vl-1.6b   |      1597 | 1.00  | 1.00  | **1.00** | **1.00** | 1.00 | 0.00  |
| minicpm-v-4_6    |      1300 | 0.93  | 1.00  | **1.00** | **1.00** | 1.00 | 0.00  |
| qwen3_5-0.8b     |       873 | 1.00  | 1.00  | 1.00  | 0.00  | 1.00  | 0.00  |
| internvl2-1b     |       938 | 0.93  | 1.00  | 1.00  | 0.00  | 1.00  | 0.01  |
| internvl2_5-1b   |       938 | 0.69  | 1.00  | 1.00  | 0.00  | 1.00  | 0.00  |
| internvl3-1b     |       938 | 0.93  | 1.00  | 0.00  | 0.00  | 1.00  | 0.00  |
| smolvlm-500m     |       500 | 0.93  | 0.94  | 0.00  | **1.00** | 1.00 | 0.00  |
| smolvlm-256m     |       256 | 0.59  | 0.94  | 0.00  | 0.00  | 1.00  | 0.00  |
| llava-ov-0.5b    |       894 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| got-ocr2         |       580 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| florence2-large  |       770 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| smoldocling-256m |       256 | 0.93  | 0.00² | 0.00  | 0.00  | 0.00  | 0.02  |
| lightonocr-1b    |      1161 | 0.00² | 0.00² | 0.00  | 0.00  | 0.00  | 0.01  |
| paddleocr-vl-1.5 |       900 | 0.00² | 0.00² | 0.00  | 0.00  | 1.00  | 0.00  |
| paddleocr-vl     |       900 | 0.00² | 0.00² | 0.00  | 0.00  | 0.00  | 0.00  |
| paddleocr-vl-1.6 |       900 | 0.00² | 0.00² | 0.00  | 0.00  | 0.00  | 0.00  |
| h2ovl-0.8b       |       800 | 0.00³ | 0.00³ | 0.00³ | 0.00³ | 0.00³ | 0.00³ |
| florence2-base   |       230 | 0.00² | 0.00² | 0.00  | 0.00  | 0.00  | 0.00  |

² OCR/transcription/parsing specialists output the whole page / task-token / DocTags format → low
on the short-answer cells (interface mismatch), even when they read correctly. ³ H2OVL emits empty
text. Full 19×6 grid + efficiency in [`../results/matrix_capability.md`](../results/matrix_capability.md).

**Headline finding (corrected by the GPU run):** **only LFM2.5-VL-1.6B and MiniCPM-V-4.6 clear
*both* hybrid-reasoning axes** (numeric-sum H1 *and* relational-compare H2 = 1.00). Notably the
interim CPU re-score had credited InternVL2.5/3-1B with both; the measured GPU run does **not**
(internvl2_5 H2 = 0.00, internvl3 H1 = 0.00) — a correction applied here. The best **strictly
sub-1B** generalist is **Qwen3.5-0.8B** (T1/T2/H1/H3 = 1.0, H2 = 0). **L1 grounding ≈ 0 on the
capability probe for every model**, but on the proposed custom-eval **LFM reaches a usable
spot-IoU of 0.229** (all others ≤ 0.04) and is **180°-rotation robust** — so grounding,
box-tracking and relational reasoning are the real gaps the Part-2 plan targets.

### Efficiency frontier (measured, T4)

LFM2.5-VL-1.6B (avg **0.98 s/sample**) is the only *capable* fast option; the InternVL-1B family
is 6–8 s, **Qwen3.5-0.8B ~13.9 s** (full-attention prefill), and the PaddleOCR-VL parsers 80–115 s
(full-page parsing). This is why Part-2 fine-tuning uses LFM as the base — Qwen3.5-VL at ~0.05 it/s
is infeasible to iterate on a free T4. The per-layer reason is dissected in
[`../../notebooks/latency_profile.ipynb`](../../notebooks/latency_profile.ipynb).

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
/ position-bias), not just hit raw accuracy. The full GPU sweep populates all 19 models; the
strongest profiles:

| model          | L2 (abs) | L3 (rel) | L4 (box) | H4 (consist.) | H5 (absence) | H6 (distractor) | H7 (xref) |
| -------------- | :------: | :------: | :------: | :-----------: | :----------: | :-------------: | :-------: |
| lfm2_5-vl-1.6b | **PASS** | **PASS** |   FAIL   |     FAIL      |   **PASS**   |     **PASS**    | **PASS**  |
| minicpm-v-4_6  | **PASS** | **PASS** |   FAIL   |     FAIL      |   **PASS**   |     **PASS**    | **PASS**  |
| internvl2_5-1b | **PASS** |   FAIL   |   FAIL   |    **PASS**   |   **PASS**   |     **PASS**    | **PASS**  |
| qwen3_5-0.8b   | **PASS** |   FAIL   |   FAIL   |     FAIL      |   **PASS**   |     **PASS**    |   FAIL    |
| smolvlm-500m   | **PASS** |   FAIL   |   FAIL   |     FAIL      |     FAIL     |     **PASS**    |   FAIL    |

Reading: **box-tracking (L4) is unsolved by every model** — the single hardest spatial axis and a
prime Part-2 target. The most spatially-robust models are **LFM2.5-VL-1.6B and MiniCPM-V-4.6**
(clear L2/L3/H5/H6/H7); the best **sub-1B** is **Qwen3.5-0.8B** (L2/H5/H6). Full 19-model table:
[`../results/probe_signals.md`](../results/probe_signals.md).

## F. Proposed custom-eval — by class / language / rotation / spotting

The proposed evaluation format (`docs/results/custom_eval_breakdown.md`) slices the score by the axes
it was built around. The decision-relevant rows (measured, T4):

| Axis (custom-eval)            | Leader(s)                                  | Reading |
| ----------------------------- | ------------------------------------------ | ------- |
| **text** (content class)      | **LFM 0.829**, Qwen3.5 0.777, InternVL2-1b 0.729 | recognition is broadly solved at the top |
| **stamp**                     | **LFM 0.551** (only non-zero)              | a niche class only LFM reads |
| **direction** (reading dir.)  | LFM / Qwen3.5 / MiniCPM = **1.0**          | most others ≈ 0.33 |
| **language — en**             | LFM / Qwen3.5 / SmolVLM-500M ≈ **0.77**    | English recognition saturates |
| **language — ko / ja / zh**   | PaddleOCR-VL-1.5/1.6 = **1.0**; LFM ko 0.875 | parsing specialists lead CJK |
| **rotation-180 retention**    | LFM / MiniCPM / PaddleOCR / LightOnOCR = **1.0**; InternVL ≈ 0.06–0.10, SmolVLM ≈ 0.10–0.13 | orientation robustness is bimodal |
| **reading-direction acc**     | LFM / Qwen3.5 / MiniCPM = **1.0**          | the rest 0.0–0.33 |
| **spotting IoU**              | **LFM 0.229**; InternVL2.5 0.042, others ≤ 0.02 | grounding is the systemic gap — only LFM is usable |

**PaddleOCR-VL version ablation (custom-eval text):** v1.0 = 0.2097, v1.5 = 0.2652, v1.6 = 0.2652
(avg latency ~5.4–6.3 s, peak GPU ~2.1 GB) — v1.5 improves on v1.0 and v1.6 matches v1.5. Full grids:
[`../results/custom_eval_breakdown.md`](../results/custom_eval_breakdown.md).

## G. Honest limitations of this run

- The **all_preview** matrix is **one sample per benchmark** → a *sanity/plumbing* matrix, not
  leaderboard accuracy; treat trends, not absolute numbers. The capability/spatial probes have one
  sample per axis too — directional, not a leaderboard. Scale to many samples for tighter numbers.
- **Numbers are a single T4 sweep** with cached predictions; latencies vary with cold/warm load
  (two runs in the notebook differ by ~2× on load time). Treat efficiency as order-of-magnitude.
- **Ovis2.5-2B is excluded** from the default sweep (custom interface, well over the <1B budget);
  opt in with `--models ovis2_5-2b`. The **fine-tuning staircase (Part 2) is not yet run** — its
  machinery is implemented and smoke-tested; projected gains are flagged as expected.

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
