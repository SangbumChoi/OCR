# W&B metrics — what is logged during fine-tuning, and how to read it

When `--wandb-project` is set (or the notebooks' W&B cell runs), every fine-tuning run started by
`scripts/run_ablation.py` → `docvlm_eval.finetune.lora_vlm.train_lora_vlm` streams a small, fixed set
of metrics to Weights & Biases. This page is the decoder ring: **what each logged key means, on which
x-axis, and how to interpret the curves.** All scores are in **[0, 1]** (higher = better); loss is
unbounded (lower = better).

## 1. The logged keys

| Key                       | x-axis (`step_metric`) | Cadence                  | Meaning |
| ------------------------- | ---------------------- | ------------------------ | ------- |
| `train/loss`              | `train/global_step`    | every `log_every` micro-steps (default 10) | Cross-entropy on the answer span of the current micro-batch. Noisy by nature. |
| `epoch/loss`              | `epoch`                | once per epoch           | Mean `train/loss` over the epoch — the smoothed "is it still learning?" line. |
| `eval/<split>_score`      | `epoch`                | once per epoch, per eval split | **Overall** score on that split (mean over its samples), using each sample's own metric. |
| `eval/<split>_<axis>`     | `epoch`                | once per epoch, per (split × answer_type) | **Per-capability** score: the same split sliced by `answer_type` (T/L/H axes). This is where you see *which* ability moved. |

There is **one W&B run per training job**. In the A0 sweep, each data size is its own run
(`A0-<model>-n<size>`); each arm is its own run (`<arm>-<model>-<placement>`).

## 2. Which `<split>`s appear (depends on the run)

`<split>` is the name of an eval set scored *inside the training loop* after each epoch:

| Run type (how it was launched)                    | Splits logged              |
| ------------------------------------------------- | -------------------------- |
| **A0** (`--arm A0`)                               | `train`, `heldout`         |
| **Arm** (`--arm A1_…`, with `--heldout-seed`)     | `train` (+ `heldout`)      |
| **Public** (`--arm public --train-jsonl …`)       | `train` (+ `heldout`)      |

- **`train`** = score on the *training* data the model just fit → the **memorization** signal.
- **`heldout`** = score on a **different-seed** synthetic set never trained on → the
  **generalization / understanding** signal. (In the public-data notebook, `heldout` is synthetic
  while training is public, so it doubles as a *cross-distribution transfer* signal — see
  [`ablation_plan.md`](ablation_plan.md) §4.)

## 3. What the per-axis `<axis>` codes mean

`eval/<split>_<axis>` is sliced by the sample's `answer_type`. The codes come from the capability
taxonomy ([`capability_axes.md`](capability_axes.md)); the per-sample scorer is chosen per sample by
its `metric` field:

| `answer_type` you'll see        | Capability                          | Underlying metric | Reads as |
| ------------------------------- | ----------------------------------- | ----------------- | -------- |
| `T1` / `text` / `multilingual`  | text recognition / transcription    | `ned`             | 1 − normalized edit distance |
| `T2` / `kie`                    | localized key-value extraction      | `anls`            | edit-distance match, 0 below 0.5 |
| `H1-aggregate` / `H-count`      | numeric sum / counting              | `relaxed_acc` / `exact` | within 5% / exact string |
| `H-comprehension` / `H2`        | relational compare ("which larger") | `anls` / `exact`  | answer match |
| `H3` / `chart`                  | chart-value read-off                | `relaxed_acc`     | within 5% |
| `L1` / `grounding`              | spotting (bounding box)             | `grounding`       | IoU of predicted vs gold box |
| `table`                         | table structure                     | `teds` / `anls`   | tree-edit similarity |

Metric cheat-sheet (all 0–1): **anls** = edit-distance match, hard-zeroed below 0.5; **ned** =
soft 1 − edit distance (good for OCR/CER-style text); **exact** = 1 if normalized strings match;
**relaxed_acc** = numeric within 5% else exact (charts/sums); **grounding** = box IoU; **teds** =
table tree-edit similarity.

## 4. How to read the curves (recipes)

- **"Is it training at all?"** `train/loss` should trend down within the first epoch and
  `epoch/loss` should drop epoch-to-epoch. A flat/NaN loss = bad inputs (wrong chat template / image
  keys) or LR issues, not a slow model.
- **"Does it understand or just memorize?" (the A0 question).** Compare `eval/train_score` vs
  `eval/heldout_score` across sizes/epochs. **Understanding** → `heldout` rises and the
  `train − heldout` gap stays small. **Memorization** → `train` climbs toward ~1.0 while `heldout`
  plateaus (large, growing gap). Pick the data scale where `heldout` plateaus with a small gap.
- **"Is it overfitting within a run?"** Within one run, watch for `eval/train_*` still rising while
  `eval/heldout_*` flattens or dips — stop earlier / add data diversity.
- **"Which capability did this arm actually move?"** Don't read only `eval/heldout_score` (an
  average that can hide a swap). Read the **per-axis** `eval/heldout_<axis>` deltas: e.g. the A1
  spotting arm should lift `eval/heldout_grounding` / `_L1`; A2 reasoning should lift
  `_H-comprehension` / `_H1-aggregate`; A4 should lift the per-language `_multilingual` slices
  *without* tanking English. A gain on the average but a drop on a target axis is a red flag.
- **"Is one capability being traded for another?" (transfer/interference).** Because every arm logs
  *all* axes, a rising target axis with a falling unrelated axis (e.g. grounding up, `T1` text
  down) is the cross-capability cost the ablation is meant to expose.

## 5. What is NOT in W&B (and where it lives instead)

- The **final whole-suite evaluation** (capability + spatial + realistic probes) is run *after*
  training and written to `docs/results/ablation_results.json`, then visualized by the notebooks'
  before/after bars and the staircase — **not** streamed per-epoch to W&B. W&B carries the
  *in-training* `train`/`heldout` curves; the JSON carries the *final* cross-probe scores.
- **ECE (calibration)**, **answer-rate**, and **per-sample** detail are computed by
  `docvlm_eval.metrics.aggregate` and saved in each run's `summary.json` / `per_sample` — they are
  not currently pushed per-epoch to W&B (the W&B stream is intentionally minimal: loss + score +
  per-axis score).

## 6. Quick reference — namespaces

```
train/loss            ← micro-step training loss        (x = train/global_step)
epoch/loss            ← epoch-mean training loss         (x = epoch)
eval/train_score      ← overall score on the train set   (x = epoch)   [memorization]
eval/heldout_score    ← overall score on the held-out set(x = epoch)   [generalization]
eval/<split>_<axis>   ← that split, sliced by capability (x = epoch)   [which ability moved]
```
