# Cross-benchmark result matrix (preview set)

Models run: 5/6 · benchmarks: 6

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model | cap_text | cap_kie | cap_integ_sum | cap_integ_rel | cap_chart | cap_ground |
|---|---|---|---|---|---|---|
| dummy-echo | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| got-ocr2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| internvl2-1b | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| smoldocling-256m | 0.00 | 0.94 | 0.00 | 0.00 | 0.00 | 0.02 |
| smolvlm-256m | 0.93 | 0.94 | 0.00 | 0.00 | 1.00 | 0.00 |
| smolvlm-500m | 0.93 | 0.94 | 1.00 | 0.00 | 1.00 | 0.00 |

## Run status

- **dummy-echo**: ok (cached)
- **got-ocr2**: ok (cached)
- **internvl2-1b**: FAIL: AttributeError: 'InternVLChatModel' object has no attribute 'all_tied_weights_keys'
- **smoldocling-256m**: ok (cached)
- **smolvlm-256m**: ok (cached)
- **smolvlm-500m**: ok (cached)
