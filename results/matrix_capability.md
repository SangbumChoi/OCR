# Cross-benchmark result matrix (preview set)

Models run: 11/11 · benchmarks: 6

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model | cap_text | cap_kie | cap_integ_sum | cap_integ_rel | cap_chart | cap_ground |
|---|---|---|---|---|---|---|
| dummy-echo | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| florence2-base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| florence2-large | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| got-ocr2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| h2ovl-0.8b | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| internvl2-1b | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| internvl2_5-1b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| internvl3-1b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| smoldocling-256m | 0.00 | 0.94 | 0.00 | 0.00 | 0.00 | 0.02 |
| smolvlm-256m | 0.93 | 0.94 | 0.00 | 0.00 | 1.00 | 0.00 |
| smolvlm-500m | 0.93 | 0.94 | 1.00 | 0.00 | 1.00 | 0.00 |

## Efficiency (load / latency / memory)

| model | device | params(M) | load(s) | avg lat(s) | p90(s) | peak CPU(MB) | peak GPU(MB) |
|---|---|---|---|---|---|---|---|
| dummy-echo | cpu | 0 | 0.0 | None | - | - | - |
| florence2-base | cpu | 230 | 4.91 | 2.474 | - | - | - |
| florence2-large | cpu | 770 | 5.6 | 7.374 | - | - | - |
| got-ocr2 | cpu | 580 | 2.9 | 12.473 | - | - | - |
| h2ovl-0.8b | cpu | 800 | 4.44 | 0.037 | - | - | - |
| internvl2-1b | cpu | 938 | 5.04 | 74.127 | - | - | - |
| internvl2_5-1b | cpu | 938 | 5.19 | 77.384 | - | - | - |
| internvl3-1b | cpu | 938 | 5.02 | 77.254 | - | - | - |
| smoldocling-256m | cpu | 256 | 14.44 | 40.466 | - | - | - |
| smolvlm-256m | cpu | 256 | 4.52 | 19.304 | 19.448 | 3093.7 | None |
| smolvlm-500m | cpu | 500 | 2.55 | None | - | - | - |

## Run status

- **dummy-echo**: ok (cached)
- **florence2-base**: ok (cached)
- **florence2-large**: ok (cached)
- **got-ocr2**: ok (cached)
- **h2ovl-0.8b**: ok (cached)
- **internvl2-1b**: ok (cached)
- **internvl2_5-1b**: ok (cached)
- **internvl3-1b**: ok (cached)
- **smoldocling-256m**: ok (cached)
- **smolvlm-256m**: ok
- **smolvlm-500m**: ok (cached)
