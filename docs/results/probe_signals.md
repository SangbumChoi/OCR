# Spatial/context shortcut-robust signals

PASS = clears the shortcut control (counterfactual / distractor / position-bias / absence), not just
raw accuracy. Source: full GPU sweep on a T4 (`scripts/run_full_comparison.sh`, captured in
`notebooks/colab_full_comparison.ipynb`).

| model            | L2   | L3   | L4   | H4   | H5   | H6   | H7   |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dummy-echo       | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL |
| florence2-base   | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| florence2-large  | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| got-ocr2         | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| h2ovl-0.8b       | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| internvl2-1b     | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| internvl2_5-1b   | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| internvl3-1b     | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | PASS |
| lfm2_5-vl-1.6b   | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS |
| lightonocr-1b    | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| llava-ov-0.5b    | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| minicpm-v-4_6    | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS |
| paddleocr-vl     | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| paddleocr-vl-1.5 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| paddleocr-vl-1.6 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| qwen3_5-0.8b     | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL |
| smoldocling-256m | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| smolvlm-256m     | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| smolvlm-500m     | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |

Reading: **L4 (box-tracking) is unsolved by every model.** The strongest spatial/context profiles are
LFM2.5-VL-1.6B and MiniCPM-V-4.6 (clear L2/L3/H5/H6/H7), then InternVL2.5-1B (L2/H4/H5/H6/H7). The
best sub-1B model is Qwen3.5-0.8B (L2/H5/H6).
