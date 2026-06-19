# Cross-benchmark result matrix (preview set)

Models run: 1/1 · benchmarks: 5

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model      | ui_login_box | ui_cta_text | ui_nav_list | ui_checkout_box | ui_search_box |
| ---------- | ------------ | ----------- | ----------- | --------------- | ------------- |
| dummy-echo | 0.00         | 0.00        | 0.13        | 0.00            | 0.00          |

## Efficiency (load / latency / memory)

| model      | device | params(M) | load(s) | avg lat(s) | p90(s) | peak CPU(MB) | peak GPU(MB) |
| ---------- | ------ | --------- | ------- | ---------- | ------ | ------------ | ------------ |
| dummy-echo | cpu    | 0         | 0.0     | 0.0        | 0.0    | 218.1        | None         |

## Run status

- **dummy-echo**: ok
