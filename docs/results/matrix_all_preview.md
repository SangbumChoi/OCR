# Cross-benchmark result matrix (preview set)

Models run: 3/3 · benchmarks: 16

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model        | ai2d | chartqa | docvqa | iam  | im2latex | infovqa | latexocr | mathvista | ocrbench | ocrbench_v2 | ocrvqa | pope | recognition_fullpage | robustness | sroie | textvqa |
| ------------ | ---- | ------- | ------ | ---- | -------- | ------- | -------- | --------- | -------- | ----------- | ------ | ---- | -------------------- | ---------- | ----- | ------- |
| dummy-echo   | 0.00 | 0.00    | 0.00   | 0.00 | 0.00     | 0.00    | 0.00     | 0.00      | 0.00     | 0.00        | 0.00   | 0.00 | 0.00                 | 0.00       | 0.00  | 0.00    |
| smolvlm-256m | 0.00 | 0.00    | 0.75   | 0.56 | 0.00     | 0.00    | 0.00     | 0.00      | 0.00     | 0.00        | 0.00   | 0.00 | 0.00                 | 0.00       | 1.00  | 0.00    |
| smolvlm-500m | 0.00 | 0.00    | 0.60   | 0.80 | 0.00     | 0.90    | 0.00     | 0.00      | 0.00     | 0.00        | 0.00   | 0.00 | 0.00                 | 0.60       | 1.00  | 0.00    |

## Run status

- **dummy-echo**: ok (cached)
- **smolvlm-256m**: ok
- **smolvlm-500m**: ok
