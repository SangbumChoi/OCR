# Cross-benchmark result matrix (preview set)

Models run: 2/2 · benchmarks: 15

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model        | sp_quad_top-left | sp_quad_top-right | sp_quad_bottom-left | sp_quad_bottom-right | sp_relpos_normal | sp_relpos_counterfactual | sp_box_top | sp_box_mid | sp_box_bot | ctx_consistency_consistent | ctx_consistency_inconsistent | ctx_absence | ctx_distractor | ctx_xref_bob | ctx_xref_alice |
| ------------ | ---------------- | ----------------- | ------------------- | -------------------- | ---------------- | ------------------------ | ---------- | ---------- | ---------- | -------------------------- | ---------------------------- | ----------- | -------------- | ------------ | -------------- |
| smolvlm-256m | 1.00             | 0.00              | 0.00                | 0.00                 | 0.00             | 1.00                     | 0.00       | 0.00       | 0.00       | 1.00                       | 0.00                         | 0.00        | 1.00           | 1.00         | 0.00           |
| smolvlm-500m | 1.00             | 1.00              | 0.00                | 1.00                 | 0.00             | 1.00                     | 0.00       | 0.01       | 0.01       | 1.00                       | 0.00                         | 0.00        | 1.00           | 0.00         | 1.00           |

## Efficiency (load / latency / memory)

| model        | device | params(M) | load(s) | avg lat(s) | p90(s) | peak CPU(MB) | peak GPU(MB) |
| ------------ | ------ | --------- | ------- | ---------- | ------ | ------------ | ------------ |
| smolvlm-256m | cpu    | 256       | 7.46    | None       | -      | -            | -            |
| smolvlm-500m | cpu    | 500       | 2.64    | None       | -      | -            | -            |

## Run status

- **smolvlm-256m**: rescored (15/15 samples)
- **smolvlm-500m**: rescored (15/15 samples)
