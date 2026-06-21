# Cross-benchmark result matrix (preview set)

Models run: 19/19 · benchmarks: 6

Per-cell = task score, each scored by its sample's own metric (ANLS / NED / relaxed-acc / exact / grounding-IoU). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy. Source: full GPU sweep on a T4 (`scripts/run_full_comparison.sh`, captured in `notebooks/colab_full_comparison.ipynb`).

| model            | cap_text | cap_kie | cap_integ_sum | cap_integ_rel | cap_chart | cap_ground |
| ---------------- | -------- | ------- | ------------- | ------------- | --------- | ---------- |
| dummy-echo       | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.00       |
| florence2-base   | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.00       |
| florence2-large  | 0.00     | 0.00    | 0.00          | 0.00          | 1.00      | 0.00       |
| got-ocr2         | 0.00     | 0.00    | 0.00          | 0.00          | 1.00      | 0.00       |
| h2ovl-0.8b       | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.00       |
| internvl2-1b     | 0.93     | 1.00    | 1.00          | 0.00          | 1.00      | 0.01       |
| internvl2_5-1b   | 0.69     | 1.00    | 1.00          | 0.00          | 1.00      | 0.00       |
| internvl3-1b     | 0.93     | 1.00    | 0.00          | 0.00          | 1.00      | 0.00       |
| lfm2_5-vl-1.6b   | 1.00     | 1.00    | 1.00          | 1.00          | 1.00      | 0.00       |
| lightonocr-1b    | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.01       |
| llava-ov-0.5b    | 0.00     | 0.00    | 0.00          | 0.00          | 1.00      | 0.00       |
| minicpm-v-4_6    | 0.93     | 1.00    | 1.00          | 1.00          | 1.00      | 0.00       |
| paddleocr-vl     | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.00       |
| paddleocr-vl-1.5 | 0.00     | 0.00    | 0.00          | 0.00          | 1.00      | 0.00       |
| paddleocr-vl-1.6 | 0.00     | 0.00    | 0.00          | 0.00          | 0.00      | 0.00       |
| qwen3_5-0.8b     | 1.00     | 1.00    | 1.00          | 0.00          | 1.00      | 0.00       |
| smoldocling-256m | 0.93     | 0.00    | 0.00          | 0.00          | 0.00      | 0.02       |
| smolvlm-256m     | 0.59     | 0.94    | 0.00          | 0.00          | 1.00      | 0.00       |
| smolvlm-500m     | 0.93     | 0.94    | 0.00          | 1.00          | 1.00      | 0.00       |

## Efficiency (load / latency / memory)

| model            | device | params(M) | load(s) | avg lat(s) | p90(s)  | peak CPU(MB) | peak GPU(MB) |
| ---------------- | ------ | --------- | ------- | ---------- | ------- | ------------ | ------------ |
| dummy-echo       | cpu    | 0         | 0.0     | 0.0        | 0.0     | 513.2        | None         |
| florence2-base   | cuda   | 230       | 21.81   | 0.998      | 1.172   | 2516.4       | 649.1        |
| florence2-large  | cuda   | 770       | 33.52   | 2.549      | 2.713   | 4004.2       | 1890.1       |
| got-ocr2         | cuda   | 580       | 24.18   | 4.114      | 4.545   | 2906.7       | 3223.0       |
| h2ovl-0.8b       | cuda   | 800       | 28.94   | 0.077      | 0.088   | 3180.5       | 1663.2       |
| internvl2-1b     | cuda   | 938       | 38.38   | 6.506      | 7.161   | 3413.9       | 3654.8       |
| internvl2_5-1b   | cuda   | 938       | 32.51   | 8.003      | 8.172   | 3409.4       | 3654.8       |
| internvl3-1b     | cuda   | 938       | 35.36   | 8.1        | 7.946   | 3407.8       | 3654.8       |
| lfm2_5-vl-1.6b   | cuda   | 1597      | 53.22   | 0.984      | 1.288   | 3570.6       | 3404.8       |
| lightonocr-1b    | cuda   | 1161      | 62.48   | 1.308      | 1.2     | 3974.8       | 2062.3       |
| llava-ov-0.5b    | cuda   | 894       | 52.09   | 6.408      | 6.94    | 4770.2       | 5179.0       |
| minicpm-v-4_6    | cuda   | 1300      | 87.23   | 2.267      | 3.266   | 3762.3       | 2903.3       |
| paddleocr-vl     | cuda   | 900       | 35.61   | 82.746     | 128.928 | 3527.7       | 2910.0       |
| paddleocr-vl-1.5 | cuda   | 900       | 37.18   | 110.891    | 129.16  | 3552.1       | 2910.1       |
| paddleocr-vl-1.6 | cuda   | 900       | 38.48   | 114.682    | 129.255 | 3538.6       | 2910.2       |
| qwen3_5-0.8b     | cuda   | 873       | 59.29   | 13.913     | 16.421  | 2815.3       | 2266.1       |
| smoldocling-256m | cuda   | 256       | 24.43   | 2.874      | 3.238   | 2370.8       | 2410.7       |
| smolvlm-256m     | cuda   | 256       | 21.98   | 2.628      | 2.893   | 2344.8       | 2410.7       |
| smolvlm-500m     | cuda   | 500       | 27.16   | 3.011      | 3.422   | 2605.4       | 2926.4       |

## Run status

All 19 models ran on a T4 GPU (cached predictions committed). dummy-echo is the wiring sentinel.
