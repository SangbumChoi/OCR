# UDD mock multi-model evaluation (no GPU — deterministic mock models)

718 public-heldout samples; per-task metrics: classification=exact; kie=anls/ned; localization=grounding; reasoning=anls/exact/relaxed_acc; recognition=ned; table=anls; vqa=anls/exact

Each cell = mean score under the task's OWN metric (per-sample `score_sample` dispatch), with `semantic_match` in parentheses as the bank comparison.

| model | classification | kie | localization | reasoning | recognition | table | vqa |
|---|---|---|---|---|---|---|---|
| oracle | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| oracle-caseflip | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| oracle-wrapped | 0.00 (0.53) | 0.51 (0.69) | 1.00 (0.67) | 0.15 (0.51) | 0.89 (0.95) | 0.99 (1.00) | 0.66 (0.84) |
| oracle-truncate | 0.00 (0.09) | 0.46 (0.53) | 0.00 (0.00) | 0.24 (0.19) | 0.50 (0.63) | 0.11 (0.66) | 0.19 (0.55) |
| constant | 0.00 (0.00) | 0.04 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.03 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| echo-question | 0.00 (0.07) | 0.05 (0.00) | 0.00 (0.00) | 0.00 (0.01) | 0.15 (0.01) | 0.00 (0.02) | 0.10 (0.35) |

Sanity: `oracle` ≈ 1.0 and `constant` ≈ 0.0 in every column; the gap between `oracle-caseflip` / `oracle-wrapped` / `oracle-truncate` rows and 1.0 is each task metric's tolerance profile applied to real UDD golds (compare [`metric_tendency.md`](metric_tendency.md)).

![mock eval](../report/figures/udd_mock_eval.png)
