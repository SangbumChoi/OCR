# Cross-model insights (auto-generated)

Synthesized by `scripts/build_insights.py` from `docs/results/`. Run it after a full sweep (GPU) to populate every section; partial data degrades gracefully.

## Top findings

- **Relational reasoning** is cleared by: internvl3-1b, internvl2_5-1b (emerges around ~1B; smaller models fail).
- **No model grounds**: spatial localisation (bbox) is ~0 for the tested general VLMs — a systemic gap, motivating spotting-capable models.

**Models with results:** florence2-base, florence2-large, got-ocr2, h2ovl-0.8b, internvl2-1b, internvl2_5-1b, internvl3-1b, llava-ov-0.5b, smoldocling-256m, smolvlm-256m, smolvlm-500m

## 1. Capability leaders (capability probe)

| axis          | best model       | score |
| ------------- | ---------------- | ----- |
| cap_text      | smolvlm-256m     | 0.93  |
| cap_kie       | internvl3-1b     | 1.00  |
| cap_integ_sum | internvl3-1b     | 1.00  |
| cap_integ_rel | internvl3-1b     | 1.00  |
| cap_chart     | florence2-large  | 1.00  |
| cap_ground    | smoldocling-256m | 0.02  |

**Reasoning emergence vs size:** integrative axes by model (params from summaries):
- florence2-base (~230M): sum=0.0, rel=0.0
- florence2-large (~770M): sum=0.0, rel=0.0
- got-ocr2 (~580M): sum=0.0, rel=0.0
- h2ovl-0.8b (~800M): sum=0.0, rel=0.0
- internvl2-1b (~938M): sum=1.0, rel=0.0
- internvl2_5-1b (~938M): sum=1.0, rel=1.0
- internvl3-1b (~938M): sum=1.0, rel=1.0
- llava-ov-0.5b (~894M): sum=1.0, rel=0.0
- smoldocling-256m (~256M): sum=0.0, rel=0.0
- smolvlm-256m (~256M): sum=0.0, rel=0.0
- smolvlm-500m (~500M): sum=1.0, rel=0.0

## 2. Grounding (spatial localisation)

Best grounding score: **smoldocling-256m = 0.02**. No model produces usable boxes (general VLMs lack a spotting head).


## 3. Efficiency vs quality

| model            | params(M) | mean score | avg lat(s) | peak CPU(MB) | peak GPU(MB) |
| ---------------- | --------- | ---------- | ---------- | ------------ | ------------ |
| florence2-base   | 230.0     | 0.000      | 2.474      | –            | –            |
| florence2-large  | 770.0     | 0.167      | 7.374      | –            | –            |
| got-ocr2         | 580.0     | 0.167      | 12.473     | –            | –            |
| h2ovl-0.8b       | 800.0     | 0.000      | 0.037      | –            | –            |
| internvl2-1b     | 938.0     | 0.500      | 74.127     | –            | –            |
| internvl2_5-1b   | 938.0     | 0.667      | 77.384     | –            | –            |
| internvl3-1b     | 938.0     | 0.667      | 77.254     | –            | –            |
| llava-ov-0.5b    | 894.0     | 0.478      | –          | –            | –            |
| smoldocling-256m | 256.0     | 0.161      | 40.466     | –            | –            |
| smolvlm-256m     | 256.0     | 0.319      | 72.576     | 3091.0       | –            |
| smolvlm-500m     | 500.0     | 0.452      | –          | –            | –            |

## 4. Custom-eval leaders (class / language)


**By content class:**
- text: dummy-echo (0.012)

**By language:**
- en: dummy-echo (0.028)
