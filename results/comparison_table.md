# Comparison Table — sub-1B VLMs on document understanding

This file has **two** parts:

1. **Published reference figures** — collected from model papers/cards for orientation. They
   are *self-reported* and use *mixed split/scale conventions* (flagged below). They are **not**
   numbers produced by this pipeline.
2. **Reproduced results** — filled automatically by `scripts/make_comparison_table.py` after
   you run `scripts/evaluate.py` on a GPU (Colab/Kaggle). This includes the calibration (ECE)
   and robustness columns that no public leaderboard reports.

---

## 1. Published reference figures (self-reported; see caveats)

| Model | Params | DocVQA (ANLS) | InfoVQA (ANLS) | OCRBench | ChartQA (relaxed) | Source / caveat |
|-------|------:|:-------------:|:--------------:|:--------:|:-----------------:|-----------------|
| **InternVL2.5-1B** | ~0.94B | **84.8** (test) | **56.0** (test) | **785** | **75.9** (test) | InternVL2.5 paper / card (arXiv:2412.05271); test splits |
| **InternVL3-1B** | ~0.94B | 81.9 (test) | 53.7 (test) | **790** | 75.3 (test) | InternVL3 paper Table 3 (arXiv:2504.10479); test splits |
| **LLaVA-OneVision-0.5B** | ~0.9B | 73.7 | 46.3 | n/r | 61.4 | HF card (self-reported; split unlabeled) |
| **SmolVLM-500M** | 0.5B | 70.5 (val) | n/r | 61.0 (=~610) | 62.8 (test) | SmolVLM paper (arXiv:2504.05299); OCRBench on /100 scale |
| **SmolVLM-256M** | 0.26B | 58.3 (val) | n/r | 52.6 (=~526) | 55.6 (test) | SmolVLM paper; OCRBench on /100 scale |
| **Florence-2-large** | ~0.77B | in paper, value unconfirmed | n/r | n/r | n/r | CVPR'24 paper reports DocVQA; HF card omits it. OCR-specialist |
| **GOT-OCR2.0** | ~0.58B | n/r | n/r | n/r | n/r | OCR-only; paper uses edit-distance/F1, not VQA metrics |
| **PaddleOCR-VL-0.9B** | ~0.9B | n/r | n/r | n/r | n/r | Parsing-only; OmniDocBench edit-distance (~0.115), not VQA |

`n/r` = not reported / not published in a comparable form.

**Caveats (must read before quoting):**

- **Split mismatch.** InternVL DocVQA/InfoVQA are **test**; SmolVLM DocVQA is **val**. Do not
  rank across them naively. Our pipeline uses **val** for both, so reproduced numbers will be
  internally consistent (and typically a little lower than test for InternVL).
- **OCRBench scale.** SmolVLM reports OCRBench on a **0–100** scale (52.6, 61.0 ≈ 526, 610 on
  the standard /1000). Do not place these in the same column as InternVL's /1000 (785, 790)
  without normalising.
- **OCR/parser specialists** (GOT, Florence-2, PaddleOCR-VL) deliberately do **not** report
  VQA-style scores — they are transcription/parsing models. Their low DocVQA/InfoVQA under our
  pipeline is *expected* and is the "recognition ≠ reasoning" finding, not a bug.
- **Self-reported.** All figures are author-reported (VLMEvalKit/OpenCompass). Treat as
  orientation; the reproduced section is the apples-to-apples comparison.

### Reading of the reference figures

- **Strongest small document model: InternVL2.5-1B** (best DocVQA/InfoVQA/ChartQA at the size;
  OCRBench within 5 pts of InternVL3-1B).
- **Consistent weak axis across the field: InfoVQA** — every model drops ~25–30 points vs
  DocVQA. This is the layout-and-numeric-reasoning gap, not a recognition gap, and is the
  target of the Part-2 improvement plan.
- **Edge frontier:** SmolVLM-256M→500M shows the size/quality slope (DocVQA 58→71); the 500M
  is a credible *on-device* document reader, the 256M a floor.

---

## 2. Reproduced results (filled by the pipeline)

> Run on a free Colab/Kaggle T4 (see `README.md` / `notebooks` section), then run
> `python scripts/make_comparison_table.py`. It overwrites the block below with measured
> ANLS/relaxed-acc/OCRBench **plus ECE and robustness retention**.

| Model | Params (M) | DocVQA | InfoVQA | ChartQA | OCRBench | Robustness | Mean ECE |
|-------|-----------:|:------:|:-------:|:-------:|:--------:|:----------:|:--------:|
| internvl2_5-1b | 938 | – | – | – | – | – | – |
| internvl3-1b | 938 | – | – | – | – | – | – |
| smolvlm-500m | 500 | – | – | – | – | – | – |
| smolvlm-256m | 256 | – | – | – | – | – | – |
| llava-ov-0.5b | 894 | – | – | – | – | – | – |
| got-ocr2 | 580 | – | – | – | – | – | – |
| florence2-large | 770 | – | – | – | – | – | – |
| paddleocr-vl | 900 | – | – | – | – | – | – |

### Robustness retention (perturbed ANLS / clean ANLS) — filled by the pipeline

| Model | downscale | jpeg | blur | rotate | noise | term_paraphrase | Worst |
|-------|:---------:|:----:|:----:|:------:|:-----:|:---------------:|:-----:|
| internvl2_5-1b | – | – | – | – | – | – | – |
