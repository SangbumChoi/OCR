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
| `eval_by_axis/<axis>/<split>` | `epoch`            | once per epoch, per (axis × split)       | **The same numbers, regrouped by axis** (e.g. `eval_by_axis/grounding/train` + `…/heldout` on one chart). Use this to read train-vs-heldout *for a single capability* side by side; `eval_by_axis/score/<split>` is the overall. |

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
eval_by_axis/<axis>/<split>  ← same numbers regrouped by axis (x = epoch) [train-vs-heldout per capability on one chart]
```

> **Two groupings, same data.** `eval/<split>_<axis>` is keyed *split-first* (good for "how does the
> heldout set look across all axes?"); `eval_by_axis/<axis>/<split>` is keyed *axis-first* (good for
> "for grounding specifically, how do train and heldout compare?"). Pick whichever W&B panel grouping
> you want — the values are identical.

## 7. Glossary — every term you'll see

### Splits (the `<split>` in `eval/<split>_*`)

- **train** — the *exact* data the model was just fine-tuned on. A high `train` score only proves it
  *fit* the data, not that it learned the task. It's the **memorization** reference.
- **heldout** — a separate set the model **never trained on**, generated with a **different random
  seed** (same distribution, unseen content). It's the **generalization / understanding** signal —
  the number you actually trust. (In the public-data notebook the held-out set is *synthetic* while
  training is *public*, so it also measures cross-distribution **transfer**.)
- **train − heldout gap** — the headline of A0: small gap = understanding; large/growing gap
  (train→~1.0, heldout flat) = memorizing the finite synthetic templates.

### Capability-axis codes (canonical — [`capability_axes.md`](capability_axes.md))

Three families; **T** = read, **L** = where, **H** = reason/context.

| Code | Name | Plain meaning |
| ---- | ---- | ------------- |
| **T1** | text recognition | read an exact printed string off the page |
| **T2** | key-value extraction (KIE) | read **one named field's** value from its region |
| **L1** | grounding / spotting | return the **bounding box** of a named element (scored by IoU) |
| **L2** | absolute region | which **quadrant** an element sits in |
| **L3** | relative position | above/below/left-of relations (with a counterfactual control) |
| **L4** | spatial tracking | follow a box as the element moves top→bottom (hardest; unsolved by all) |
| **H1** | reasoning — sum | **arithmetic** (sum/total/mean) over read values |
| **H2** | reasoning — compare | **comparison** ("which is largest?") over read values |
| **H3** | chart-value | read a number **off a chart** (graphic perception) |
| **H4** | context — consistency | do the line items match the printed total? |
| **H5** | context — anti-hallucination | an **absent** field → answer "none"/abstain, don't invent |
| **H6** | context — disambiguation | pick *Total* vs a look-alike *Subtotal* (distractor) |
| **H7** | context — cross-reference | follow a link ("Bill-to" name → its amount) |

### Realistic-case tags (the `answer_type`s the synthetic doc cases emit)

These are the labels you'll see most in `eval/heldout_<axis>` for the realistic set. They map onto
the H/L families above but name the *task* concretely:

| Tag | What the question asks | Example |
| --- | ---------------------- | ------- |
| **kie** | one field's value (≈ T2) | "What is the invoice number?" |
| **ocr-full** | transcribe the whole page (≈ T1) | "Transcribe all text." |
| **handwriting** | read handwritten text | IAM-style line |
| **multilingual** | read/answer in a non-Latin script | CJK / Arabic transcription; measures language coverage |
| **direction** | reading-direction recognition | vertical / right-to-left manuscript |
| **special-glyph** | out-of-vocabulary / invented glyphs | the OOV fallback probe |
| **selection** | which option/checkbox is chosen | "Which plan is selected?" |
| **H-count** | **count** occurrences | "How many contact methods are checked?" |
| **H1-aggregate** | aggregate arithmetic over a column | sum / mean / max of a table column |
| **H-accounting** | multi-step accountant calc | "grand total after adding 10% tax" |
| **H-comprehension** | **understand a relation & pick** | "which row has the larger amount?", argmax-lookup |
| **H-action** | **next-action** / affordance reasoning | "what should the user do next?", primary CTA |
| **H-extract-strict** | exact substring, no paraphrase allowed | "the surname from the MRZ" (scored `exact`) |
| **consistency** | items-vs-total agreement (≈ H4) | "do the line items add up to the total?" |
| **reading-order** | correct sequence of regions | "list the sections top-to-bottom" |
| **ui** | web/app UI element understanding | locate the nav / button |
| **infographic** | dense infographic QA | layout + chart + numeric fusion |
| **form/total** | the total field on a form | "What is the total due?" |

So **"comprehension"** (`H-comprehension`) = *understand a cross-cell relation and choose the right
answer* (compare/lookup), and **"action"** (`H-action`) = *infer what to do next* (the agentic /
UX-affordance axis) — distinct from plain arithmetic (`H1-aggregate`) or counting (`H-count`).

#### The 14 tags, spelled out

- **kie** — *key information extraction*: read one named field's value. "What is the invoice number?"
  → `INV-2024-019`. (≈ T2; metric `anls`.)
- **ocr-full** — *full-page transcription*: return all the text on the page, in order (≈ T1; `ned`).
- **multilingual** — the answer is in a **non-Latin script** (Korean/Japanese/Chinese/Arabic); it
  measures language coverage, not a different task. "Transcribe the characters." (`ned`.)
- **direction** — *reading-direction recognition*: is the text left-to-right, right-to-left, or
  vertical? "Is this text right-to-left or left-to-right?" → `right-to-left`. (`exact`/`anls`.)
- **selection** — *which option/checkbox/state is chosen*. "Which shipping speed is selected?" →
  `Express`. (`anls`.)
- **H-count** — *count occurrences*. "How many contact methods are checked?" → `3`. (`exact`.)
- **H1-aggregate** — *aggregate arithmetic over a column*: sum / mean / min / max. "What is the total
  of the Amount column?" → `60.00`. (`relaxed_acc`.)
- **H-accounting** — *multi-step accountant calc* on top of the table. "What is the grand total after
  adding 10% sales tax to the total?" (`relaxed_acc`/`anls`.)
- **H-extract-strict** — *exact substring, no paraphrase tolerated*. "Extract only the surname from
  the MRZ (after the country code, before `<<`)." Scored `exact`, so near-misses get 0.
- **H-comprehension** — *understand a relation and pick*: compare two rows, argmax-lookup ("in the row
  with the highest amount, what is the item?"). (`anls`/`exact`.)
- **H-action** — *next-action / affordance reasoning*: "What is the primary action this page wants?" →
  the CTA; "what should the support agent do next?" (`anls`.)
- **consistency** — *do the parts agree?* line items vs the printed total (≈ H4). "Do the line items
  add up to the total?" → `yes`/`no`. (`anls`.)
- **reading-order** — *the correct sequence of regions*. "List the sections top-to-bottom." (`anls`.)
- **ui** — *web/app UI element understanding*: locate/name the nav, button, or field on a screen
  (the UX-as-document axis). (`anls`/`grounding`.)
- **infographic** — *dense infographic QA*: layout + embedded chart + text + numeric fusion (the
  hardest layout-reasoning slice). (`anls`.)
- **form/total** — the *total* field on a form/receipt specifically. "What is the total due?" (`anls`.)

### Control-probe tags (`probe:<kind>`) — the shortcut-robust checks

Some samples are **control probes**: questions deliberately written so a model can't pass by guessing
from a language prior — it must read the pixels (and sometimes *refuse*). They carry the answer_type
`probe:<kind>` (e.g. **`probe:abstain`**, **`probe:direction`**) and are all scored with `anls`
against an accept-set. The `probe:` prefix just marks "this is the falsifiable control variant".

| Tag | What it tests | Correct answer is… |
| --- | ------------- | ------------------ |
| **probe:abstain** | **anti-hallucination**: the field is *absent / not legible*, so inventing a value is the failure. "What is the shipping tracking number?" on a doc with none. | an abstention — any of `not present`, `none`, `n/a`, `redacted`, `unknown`, … (`ABSTAIN_OK`) |
| **probe:direction** | **reading direction** read off the pixels, not assumed. "Is this text right-to-left or left-to-right?" | `right-to-left` / `left-to-right` / `vertical` |
| **probe:consistency** | **cross-check**: do two stated quantities agree? "Do the line items match the total?" | `yes`/`agree` or `no`/`disagree` |
| **probe:order** | **reading-order** of regions as a control | the described correct sequence |

> Why they exist: a model that always echoes a plausible number scores fine on ordinary `kie` but
> **fails `probe:abstain`** (it hallucinates a tracking number that isn't there). Watching
> `eval/heldout_probe:abstain` separately tells you whether fine-tuning improved *honesty*, not just
> recall. (`spotting`/`grounding` samples likewise get the `grounding` answer_type, scored by box IoU.)

### Metrics (how each axis's score is computed — all 0–1)

- **ned** — *normalized edit distance similarity* = 1 − (edits / length). Forgiving, character-level;
  used for OCR/transcription (`T1`, handwriting, multilingual). 1.0 = perfect read.
- **anls** — *average normalized Levenshtein similarity*, but **hard-zeroed below 0.5**. The official
  DocVQA metric; used for KIE/short answers. Tolerates `$1,200` vs `1200` but not a wrong value.
- **exact** — 1 only if the normalized strings match exactly (yes/no, counts, strict extraction).
- **relaxed_acc** — numeric answer within **5%** relative error, else exact. Official ChartQA metric
  (`H1`, `H3`).
- **grounding** — **IoU** between the predicted and gold bounding box (`L1`). 0 = no overlap.
- **teds** — tree-edit-distance similarity for **table** structure+content.
- **ocrbench** — gold string contained in the prediction (OCRBench convention).

> The signal axes (`L2–L4`, `H4–H7`) on the spatial/context probe are **PASS/FAIL**, not a 0–1 score:
> a model must clear a *shortcut control* (counterfactual / distractor / position-bias) to pass, so a
> PASS means it read the pixels rather than guessing from a language prior.
