# Cross-benchmark result matrix (preview set)

Models run: 1/1 · benchmarks: 4

Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.

| model      | oov_nolegend | oov_legend | oov_runic | oov_7seg |
| ---------- | ------------ | ---------- | --------- | -------- |
| dummy-echo | 0.33         | 0.14       | 0.20      | 0.00     |

## Efficiency (load / latency / memory)

| model      | device | params(M) | load(s) | avg lat(s) | p90(s) | peak CPU(MB) | peak GPU(MB) |
| ---------- | ------ | --------- | ------- | ---------- | ------ | ------------ | ------------ |
| dummy-echo | cpu    | 0         | 0.0     | 0.0        | 0.0    | 217.7        | None         |

## Run status

- **dummy-echo**: ok
