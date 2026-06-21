# Cross-model insights (auto-generated)

Synthesized by `scripts/build_insights.py` from `docs/results/`. The tables below are from the
**full GPU sweep** (19 models, T4; `notebooks/colab_full_comparison.ipynb`); re-run the script
after the next sweep to refresh.

## Top findings

- **Relational reasoning (H2)** is cleared by only **lfm2_5-vl-1.6b, minicpm-v-4_6, smolvlm-500m**; both reasoning axes (H1+H2) only by **lfm2_5-vl-1.6b and minicpm-v-4_6**. The best strictly-sub-1B model is **qwen3_5-0.8b**.
- **Grounding is the systemic gap:** capability-probe bbox ≈ 0 for all; on the custom-eval only **lfm2_5-vl-1.6b reaches spot-IoU 0.229** (others ≤ 0.04). **Box-tracking (L4) is unsolved by every model.**
- **Efficiency frontier:** **lfm2_5-vl-1.6b** is the only *capable* fast model (0.98 s/sample) — ~14× faster than qwen3_5-0.8b (13.9 s) — which is why it is the Part-2 fine-tuning base.

**Models with results (19):** dummy-echo, florence2-base, florence2-large, got-ocr2, h2ovl-0.8b, internvl2-1b, internvl2_5-1b, internvl3-1b, lfm2_5-vl-1.6b, lightonocr-1b, llava-ov-0.5b, minicpm-v-4_6, paddleocr-vl, paddleocr-vl-1.5, paddleocr-vl-1.6, qwen3_5-0.8b, smoldocling-256m, smolvlm-256m, smolvlm-500m

## 1. Capability leaders (capability probe)

| axis          | best model               | score |
| ------------- | ------------------------ | ----- |
| cap_text      | lfm2_5-vl-1.6b / qwen3_5-0.8b | 1.00 |
| cap_kie       | lfm2_5-vl-1.6b (+ many)  | 1.00  |
| cap_integ_sum | lfm2_5-vl-1.6b (+ many)  | 1.00  |
| cap_integ_rel | lfm2_5-vl-1.6b / minicpm-v-4_6 / smolvlm-500m | 1.00 |
| cap_chart     | lfm2_5-vl-1.6b (+ many)  | 1.00  |
| cap_ground    | smoldocling-256m         | 0.02  |

**Reasoning emergence (integrative axes, measured GPU):**
- smolvlm-256m (~256M): sum=0.0, rel=0.0
- smolvlm-500m (~500M): sum=0.0, rel=1.0
- qwen3_5-0.8b (~873M): sum=1.0, rel=0.0
- internvl2_5-1b (~938M): sum=1.0, rel=0.0
- internvl3-1b (~938M): sum=0.0, rel=0.0
- minicpm-v-4_6 (~1300M): sum=1.0, rel=1.0
- lfm2_5-vl-1.6b (~1597M): sum=1.0, rel=1.0

## 2. Grounding (spatial localisation)

Capability-probe bbox ≈ 0 for all. On the proposed custom-eval, **lfm2_5-vl-1.6b = 0.229 spot-IoU** is the only usable grounder (next best ≤ 0.042). **L4 box-tracking = 0 for every model.**

## 3. Efficiency vs quality (T4)

| model            | params(M) | avg lat(s) | peak GPU(MB) |
| ---------------- | --------- | ---------- | ------------ |
| lfm2_5-vl-1.6b   | 1597      | 0.98       | 3405         |
| h2ovl-0.8b       | 800       | 0.077      | 1663         |
| florence2-base   | 230       | 0.998      | 649          |
| minicpm-v-4_6    | 1300      | 2.267      | 2903         |
| got-ocr2         | 580       | 4.114      | 3223         |
| internvl2_5-1b   | 938       | 8.003      | 3655         |
| qwen3_5-0.8b     | 873       | 13.913     | 2266         |
| paddleocr-vl-1.6 | 900       | 114.682    | 2910         |

## 4. Custom-eval leaders (class / language)

**By content class:** text → lfm2_5-vl-1.6b (0.829); stamp → lfm2_5-vl-1.6b (0.551, unique); direction → lfm2_5-vl-1.6b / qwen3_5-0.8b / minicpm-v-4_6 (1.0).

**By language (text):** en → lfm2_5-vl-1.6b / qwen3_5-0.8b / smolvlm-500m (~0.77); ko → paddleocr-vl-1.5/1.6 (1.0), lfm2_5-vl-1.6b (0.875); zh → several (1.0).
