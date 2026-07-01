# Task-value ablation — is each task worth adding? **[DEMO — illustrative numbers; rerun `run_task_value.py` on a GPU to replace.]**

For each UDD task we LoRA-fine-tune the base **on that task alone, with an equal number of samples**, and score the fixed synthetic probe suite. `Δ` is versus the un-tuned baseline; a positive Δ means the task earns its slot at that data budget, ≈0/negative means it does not help (or transfers poorly to the validation distribution). `task:all` = the mixed mix.

## lfm2_5-vl-1.6b

Equal budget: **N=30 samples/task**, steps=300, LoRA placement=all.

| training set | capability | Δ | realistic | Δ | verdict |
|---|---|---|---|---|---|
| _baseline (no FT)_ | 41.2 | — | 38.5 | — | reference |
| recognition | 45.6 | +4.4 | 42.7 | +4.2 | worth adding |
| vqa | 44.8 | +3.6 | 41.0 | +2.5 | worth adding |
| kie | 43.1 | +1.9 | 39.9 | +1.4 | worth adding |
| reasoning | 42.0 | +0.8 | 39.1 | +0.6 | worth adding |
| table | 41.0 | -0.2 | 38.2 | -0.3 | marginal |
| **all (mixed)** | 46.9 | +5.7 | 43.5 | +5.0 | worth adding |

![task value](../report/figures/task_value.png)
