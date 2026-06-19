# Cross-benchmark result matrix (preview set)

Models run: 1/1 · benchmarks: 27

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model      | ce_text_read | ce_text_value | ce_spot_total | ce_table | ce_formula | ce_chart | ce_qr | ce_barcode | ce_stamp | ce_logo | ce_rot0_read | ce_rot0_angle | ce_rot15_read | ce_rot15_angle | ce_rot90_read | ce_rot90_angle | ce_rot180_read | ce_rot180_angle | ce_rot270_read | ce_rot270_angle | ce_lang_en | ce_lang_ko | ce_lang_ja | ce_lang_zh | ce_dir_ja | ce_dir_zh | ce_dir_en |
| ---------- | ------------ | ------------- | ------------- | -------- | ---------- | -------- | ----- | ---------- | -------- | ------- | ------------ | ------------- | ------------- | -------------- | ------------- | -------------- | -------------- | --------------- | -------------- | --------------- | ---------- | ---------- | ---------- | ---------- | --------- | --------- | --------- |
| dummy-echo | 0.08         | 0.00          | 0.00          | 0.00     | 0.00       | 0.00     | 0.00  | 0.00       | 0.00     | 0.00    | 0.00         | 0.00          | 0.00          | 0.00           | 0.00          | 0.00           | 0.00           | 0.00            | 0.00           | 0.00            | 0.06       | 0.00       | 0.00       | 0.00       | 0.00      | 0.00      | 0.00      |

## Efficiency (load / latency / memory)

| model      | device | params(M) | load(s) | avg lat(s) | p90(s) | peak CPU(MB) | peak GPU(MB) |
| ---------- | ------ | --------- | ------- | ---------- | ------ | ------------ | ------------ |
| dummy-echo | cpu    | 0         | 0.0     | 0.0        | 0.0    | 217.9        | None         |

## Run status

- **dummy-echo**: ok
