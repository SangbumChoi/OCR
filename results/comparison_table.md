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

| Model                        | Params        | DocVQA (ANLS)   | InfoVQA (ANLS)  | OCRBench         | ChartQA (relaxed) | Source / caveat                                                 |
| ---------------------------- | ------------: | :-------------: | :-------------: | :--------------: | :---------------: | --------------------------------------------------------------- |
| **InternVL2.5-1B**           | ~0.94B        | **84.8** (test) | **56.0** (test) | 785              | **75.9** (test)   | InternVL2.5 paper / card (arXiv:2412.05271); test splits        |
| **InternVL3-1B**             | ~0.94B        | 81.9 (test)     | 53.7 (test)     | **790**          | 75.3 (test)       | InternVL3 paper Table 3 (arXiv:2504.10479); test splits         |
| **InternVL2-1B**             | ~0.94B        | 81.7            | 50.9            | 754              | 72.9              | InternVL2 card (older 1B; OpenGVLab/InternVL2-1B)               |
| **Ovis2-1B**                 | ~1.0B         | n/r             | n/r             | ~890 (=89.0/100) | n/r               | AIDC-AI card; OCRBench on /100 scale; DocVQA n/r                |
| **LLaVA-OneVision-0.5B**     | ~0.9B         | 73.7            | 46.3            | n/r              | 61.4              | HF card (self-reported; split unlabeled)                        |
| **H2OVL-Mississippi-0.8B**   | 0.8B          | n/r             | n/r             | 751              | n/r               | h2oai card; OCR-focused; DocVQA/ChartQA n/r                     |
| **SmolVLM-500M**             | 0.5B          | 70.5 (val)      | n/r             | 61.0 (=~610)     | 62.8 (test)       | SmolVLM paper (arXiv:2504.05299); OCRBench on /100 scale        |
| **SmolVLM-256M**             | 0.26B         | 58.3 (val)      | n/r             | 52.6 (=~526)     | 55.6 (test)       | SmolVLM paper; OCRBench on /100 scale                           |
| **SmolDocling-256M**         | 0.26B         | n/r             | n/r             | n/r              | n/r               | ds4sd; doc-conversion (DocTags), not VQA-scored                 |
| **Florence-2-large / -base** | 0.77B / 0.23B | unconfirmed     | n/r             | n/r              | n/r               | CVPR'24 paper reports DocVQA(large); cards omit. OCR-specialist |
| **GOT-OCR2.0**               | ~0.58B        | n/r             | n/r             | n/r              | n/r               | OCR-only; paper uses edit-distance/F1, not VQA metrics          |
| **PaddleOCR-VL 0.9B / 1.5**  | ~0.9B         | n/r             | n/r             | n/r              | n/r               | Parsing-only → see OmniDocBench table below                     |

`n/r` = not reported / not published in a comparable form.

### Document-parsing benchmark (OmniDocBench) — for the parsing specialists

The VQA columns above don't fit transcription/parsing models, which report **OmniDocBench**
(lower edit-distance / higher TEDS-CDM is better). Included so PaddleOCR-VL **1.0 vs 1.5** are
comparable, with stronger-but-larger models for context:

| Model                          | Params | OmniDocBench overall                  | Text edit↓ | Table TEDS↑ | Formula CDM↑ | Source                                        |
| ------------------------------ | -----: | :-----------------------------------: | :--------: | :---------: | :----------: | --------------------------------------------- |
| **PaddleOCR-VL-1.5**           | ~0.9B  | **94.5** (v1.5)                       | 0.035      | 92.76       | 94.21        | PaddlePaddle/PaddleOCR-VL-1.5 card            |
| **PaddleOCR-VL (0.9B, "1.0")** | ~0.9B  | SOTA-class (v1.0/1.5; exact v1.0 n/e) | low        | high        | high         | PaddlePaddle/PaddleOCR-VL card                |
| GOT-OCR2.0                     | ~0.58B | text Edit 0.035 / F1 0.972 (own eval) | 0.035      | n/r         | n/r          | arXiv:2409.01704                              |
| *MonkeyOCR-pro-1.2B* (>1B)     | 1.2B   | edit 0.153(EN)/0.223(ZH)              | –          | 76.5/83.7   | –            | echo840/MonkeyOCR-pro-1.2B (out of <1B scope) |
| *dots.ocr* (>1B)               | ~1.7B  | Table-TEDS 88.6(EN)                   | 0.032(EN)  | 88.6        | –            | rednote-hilab/dots.ocr (out of scope)         |

> ⚠ The PaddleOCR-VL-1.5 figures and its arXiv id should be re-verified directly against the
> card/paper before publication (numbers self-reported; v1.0 per-metric values not enumerated
> on the card).

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

| Model            | Params (M) | DocVQA | InfoVQA | ChartQA | OCRBench | Robustness | Mean ECE |
| ---------------- | ---------: | :----: | :-----: | :-----: | :------: | :--------: | :------: |
| internvl2_5-1b   | 938        | –      | –       | –       | –        | –          | –        |
| internvl3-1b     | 938        | –      | –       | –       | –        | –          | –        |
| internvl2-1b     | 938        | –      | –       | –       | –        | –          | –        |
| ovis2-1b         | 1000       | –      | –       | –       | –        | –          | –        |
| h2ovl-0.8b       | 800        | –      | –       | –       | –        | –          | –        |
| smolvlm-500m     | 500        | –      | –       | –       | –        | –          | –        |
| smolvlm-256m     | 256        | –      | –       | –       | –        | –          | –        |
| smoldocling-256m | 256        | –      | –       | –       | –        | –          | –        |
| llava-ov-0.5b    | 894        | –      | –       | –       | –        | –          | –        |
| got-ocr2         | 580        | –      | –       | –       | –        | –          | –        |
| florence2-large  | 770        | –      | –       | –       | –        | –          | –        |
| florence2-base   | 230        | –      | –       | –       | –        | –          | –        |
| paddleocr-vl     | 900        | –      | –       | –       | –        | –          | –        |
| paddleocr-vl-1.5 | 900        | –      | –       | –       | –        | –          | –        |

### Robustness retention (perturbed ANLS / clean ANLS) — filled by the pipeline

| Model          | downscale | jpeg | blur | rotate | noise | term_paraphrase | Worst |
| -------------- | :-------: | :--: | :--: | :----: | :---: | :-------------: | :---: |
| internvl2_5-1b | –         | –    | –    | –      | –     | –               | –     |
