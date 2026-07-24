# Ablation plan (PRD) — improving a small VLM via fine-tuning

## 1. Objective

Once evaluation localises the **best small model's weaknesses** (for InternVL2.5-1B: no spotting/
grounding, weaker InfoVQA layout-reasoning, CJK < EN, rotation/robustness drop), we improve it by
fine-tuning on a synthetic document-generation dataset. Rather than changing many things at once, we
run **isolated ablations** — each answers exactly one research question against a held-out control —
then **stack the winners** into a cumulative **staircase** of gain. The per-capability
*which-module-to-adapt* hypotheses these arms test are laid out in
[`research_novelty.md`](research_novelty.md).

## 2. Target model & base data

Default target model: **LFM2.5-VL-1.6B**. It is the practical Part-2 default because its hybrid-conv
backbone fine-tunes much faster on a T4 than Qwen3.5-VL's full-attention visual prefill (Qwen runs
~0.05 it/s; the per-stage/per-layer reason is dissected in
[`../../notebooks/latency_profile.ipynb`](../../notebooks/latency_profile.ipynb)). Qwen3.5-0.8B
remains selectable with `run_ablation.py --models qwen3_5-0.8b`, but the notebook and ablation
registry default to LFM for feasible iteration. Base data: synthetic document pairs (image →
structured text) from the generator, augmented per-ablation.

## 3. Experimental controls

**Control factor (held fixed across every arm so a delta is attributable to the factor alone):**
the **number of training images / iterations** (`--count` = #images, `--steps`) and the optimiser/seed
and eval suite. Most arms keep the *image set fixed* and change only **what GT is read per image**
(reasoning = capturing the per-image information diversity instead of missing it); **A4** instead
fixes the **total sample count** and varies the language mix (equal totals → synergy/interference is
comparable). Every arm is scored on the **whole suite** (capability, spatial, realistic) to expose
**cross-capability transfer** — e.g. does spotting also help CER/KIE/reasoning? does a 2nd language
hurt EN? (`scripts/run_ablation.py`).

## 4. A0 — memorization vs understanding (PREREQUISITE, run first)

### 4.1 Why A0 comes first: the synthetic space is finite

Synthetic generation *looks* infinite — we can sample endlessly — but the reachable space is in fact
**finite, and far smaller than real-world data**. Every document is a combination of a bounded set of
**contexts** (templates, field vocabularies, value ranges) and **layouts** (a fixed catalogue of
document types / structural arrangements). The cartesian product is large but **closed**. An LLM in
the loop could widen this space dramatically (paraphrased contexts, novel layouts), but that requires
an LLM gateway which is **out of scope here on cost grounds** (see
[`synth_generation_survey.md`](synth_generation_survey.md)). So we deliberately live with a finite,
simulation-only space — and must guard against the model simply **memorising** it.

### 4.2 What A0 measures

Because the space is finite, a model with enough capacity can **fit the templates** instead of
learning the task. A0 detects this directly: train at increasing data **scale** (`count` = #images,
seed 7) for a fixed #epochs and, at every scale, score **two splits**:

- **train split** — the exact images/QA the model just fit (the *memorization* signal). If this
  climbs toward **~100%** it only proves the model *can* fit the finite data, **not** that it
  understands the task.
- **held-out validation / test split** — generated with a **different seed** (unseen content, same
  distribution; the *identical* set is reused for every scale so the curve is comparable). This is the
  *understanding / generalization* signal.

**Read-out:**

- *Understanding* → the held-out curve keeps rising with scale and the **train − heldout gap stays
  small**.
- *Memorization* → **train → ~1.0 while held-out plateaus** (a large, growing gap).

The recommended synthetic data scale is the point where the **held-out curve plateaus while the gap is
still small**. Past that point, pouring in more of the same finite templates buys memorization, not
generalization — so the lever to push instead is **diversity** (more doc types / harder GT), not raw
count (see [`synthetic_data_dto.md`](synthetic_data_dto.md)).

Run the size sweep with `scripts/run_ablation.py --arm A0 --a0-sizes 50 100 200 400 800` (full curve
adds `3200`); the prerequisite section of `notebooks/finetune_ablation.ipynb` plots the **train vs
held-out** learning curves + the gap and reads off the size. **A0's result fixes the data scale used
by A1–A7.** (Per-epoch `train`/`heldout` curves stream to W&B — see
[`wandb_metrics.md`](wandb_metrics.md) for what each logged key means and how to read it.)

### 4.3 Two synthetic-quality axes A0 trades off

**Two synthetic-quality axes we scale** (see [`synthetic_data_dto.md`](synthetic_data_dto.md)):
*visual diversity* (14 doc types × acquisition/lighting/colour — `D_visual_diverse`) and *annotation
difficulty* (accountant-style multi-step calc, MRZ field-parse, table-extremes diff, next-action —
the understanding layer). They are the levers A0 trades off (more diversity vs more count): when A0
shows the held-out curve plateauing, the productive move is to widen these axes rather than add more
samples of the existing finite templates.

## 5. Baseline gaps the ablations must close

The arms exist to close specific **measured** gaps, mapped to a module + ablation in
[`research_novelty.md`](research_novelty.md) and shown in
[`../../notebooks/finetune_ablation.ipynb`](../../notebooks/finetune_ablation.ipynb).

> ⚠️ **The gap list below was measured on Qwen3.5-0.8B** (the original evaluation base): L1
> grounding/spotting ≈ 0, L4 box-tracking = 0, H2 relational-compare = 0, 180° rotation collapse, slow
> latency. **Since the Part-2 default base moved to LFM2.5-VL-1.6B, those numbers are now a legacy
> reference and may be deprecated** — LFM has a different vision stack and tokenizer, so its gap
> profile must be **re-measured before the arms are interpreted**. *Action:* run the baseline arm on
> the LFM default to refresh `gaps.lfm2_5-vl-1.6b` in `configs/ablations.yaml`:
> ```bash
> python scripts/run_ablation.py --arm baseline          # LFM is the default base
> ```
> Until that run lands, treat the current LFM gap set `[L1_grounding, spot_iou]` as **provisional**
> (carried over from the Qwen profile, not yet re-measured on LFM).

The **latency** gap is dissected per-stage/per-layer (why the *smaller* Qwen3.5-0.8B is slower than
LFM2.5-VL-1.6B — vision-token count vs hybrid-conv per-layer cost) in
[`../../notebooks/latency_profile.ipynb`](../../notebooks/latency_profile.ipynb).

> **A1 grounding curriculum (LFM default).** The first LFM A1 W&B run showed that connector-only
> training with sparse pixel-coordinate rows barely moved train grounding. The current training path
> therefore repeats grounding rows during A1/A3 spotting arms and trains boxes as normalized 0-1
> coordinates; the existing grounding metric accepts normalized predictions and rescales them to the
> original-pixel gold frame. Run A1 with held-out logging before interpreting train-only gains.
>
> **Spotting coordinate caveat (Qwen3.5-VL optional).** Qwen smart-resizes the input and may emit
> boxes in the resized frame's absolute pixels, so a predicted box can be offset/scaled vs our
> original-pixel GT. Check the processor frame before reading Qwen A1 grounding numbers.

## 6. Method: one factor at a time → integrate → staircase

For each ablation A_i we train two (or more) variants differing **only** in A_i, evaluate on the
suite, and record Δ vs the control. Winners are composed in dependency order; the cumulative run
should step the headline metric up at each addition (`scripts/plot_ablation.py` draws it).

| ID                                                | Research question                                                                                                                 | Factor varied (data/training)                                             | Control (what stays fixed)  | Primary metric(s)                          | Hypothesis                                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **A1 Spotting supervision**                       | Does adding *where* (bbox) targets to training improve extraction & spatial understanding, vs. answer-only?                       | grounding rows repeated; target = normalized `[x1,y1,x2,y2]` vs answer-only | same images/QA, same steps  | grounding IoU; KIE F1; InfoVQA ANLS        | spotting injects spatial grounding → IoU↑ and KIE↑ (localised attention)                   |
| **A2 Reasoning supervision**                      | Does CoT/rationale in the target help integrative reasoning, or is answer-only as good?                                           | target = `rationale → answer` vs `answer`                                 | same data                   | content-reasoning (sum/cmp); InfoVQA               | rationale teaches multi-region procedure → reasoning↑                                      |
| **A3 Spotting+Reasoning vs combined-direct**      | Is it the *act of adding* these signals that helps, or does the plain task combination already capture it?                        | {answer} vs {+spot} vs {+reason} vs {+spot+reason}                        | same data/steps             | composite + per-axis                       | the structured signals beat plain combination; spot & reason are complementary             |
| **A4 Multilingual mixing & language correlation** | Does training several languages in one mix beat single-language? Which language *pairs* transfer?                                 | train sets: {en}, {en+es}, {en+ja}, {ko+en}, {en+zh}, {all}               | equal total samples         | per-language NED (custom_eval)             | related scripts transfer (en↔es, ko↔en); distant scripts (en↔ja) help less or interfere    |
| **A5 LoRA placement**                             | Which modules to adapt to inject which capability — **vision encoder** vs **connector/projector** vs their **union** vs **LLM-attn** vs **LLM-MLP**? | LoRA target_modules set                                                   | rank/alpha/lr fixed for discovery; trainable parameters matched for the confirmatory union test | per-axis (spatial→vision?, reasoning→LLM?) | spatial/recognition gains come from vision+connector; reasoning/language from LLM attn+mlp |
| **A6 Hyperparameter optimization**                | Best rank `r`, `alpha`, lr, epochs for the chosen placement?                                                                      | r∈{8,16,32,64}, α, lr, epochs                                             | placement fixed (A5 winner) | composite + overfit gap                    | moderate r (16–32) best; too-high r overfits the small data                                |
| **A7 Preprocessing / resize logic**               | How does the image-resizing/tiling logic affect small-text & layout?                                                              | dynamic tiling (n_max) vs fixed resize; max-resolution; aspect-ratio keep | model/data fixed            | InfoVQA, OCRBench, small-text NED          | higher-res dynamic tiling → InfoVQA/OCRBench↑ (small text legible)                         |

## 7. A4 detail — language-correlation matrix

We don't just ask "is multilingual better" — we measure **pairwise transfer**:
`transfer(L1←L2) = score_{L1}(train L1+L2) − score_{L1}(train L1)`. A heatmap over
{en, es, ko, ja, zh} reveals complementary pairs (expected positive: en↔es shared latin; ko↔en,
zh↔en via shared doc structure) vs. interference (possible negative: en↔ja if capacity is split).
This guides the final training mix (include synergistic languages, drop interfering ones).

## 8. A5 detail — where to put LoRA (resolved by introspection, not hardcoded)

Because the two bases have different module names (Qwen3.5-VL vs LFM2-VL), the placement groups are
resolved at **runtime** from the loaded model
(`docvlm_eval.finetune.lora_vlm.resolve_lora_targets(model, group)`): every `nn.Linear` is bucketed
by its path into one of the groups below, so the same A5 ablation runs unchanged on either model.

> **Preliminary (LFM, brief run).** A first **vision vs connector** comparison on the spotting arm
> *appears* to favour the **vision encoder** for grounding (held-out spot-IoU ≈ 0.14 vs 0.08;
> `L1-locate` train 0.163 vs ~0), with `kie` held (0.986) under both — consistent with
> "spatial/grounding ← vision". Small-scale, direction-only; see
> [`technical_report.md`](technical_report.md) §Part 2.1c.

| Target group   | Bucketed by path                                  | Capability it should move                  |
| -------------- | ------------------------------------------------- | ------------------------------------------ |
| Vision encoder | `visual* / vision_tower / siglip / navit / aimv2` | recognition, small-text, layout perception |
| Connector      | `merger / projector / multi_modal / resampler`    | visual→text alignment, **grounding**       |
| LLM attention  | LLM side, leaf `{q,k,v,o}_proj`                    | **reasoning**, cross-region attention      |
| LLM MLP        | LLM side, leaf `{gate,up,down}_proj`              | knowledge, **language/CJK**                |

Each group is a separate LoRA run; we attribute per-axis gains to placement (e.g., if grounding
only improves when the connector+vision are adapted, that confirms the spatial pathway).

## 9. Integration → the staircase

> ⚠️ **Deprecated as the report headline.** The technical report is now refocused on a single
> selected objective — **spotting / grounding (L1) for human-in-the-loop verification**
> ([`technical_report.md`](technical_report.md) §Part 2.1b/1c). The cumulative staircase below is
> **retained for optional later use** (the `plot_ablation.py` tooling still works) but is **no longer
> the deliverable**; the headline is the before/after on held-out spot-IoU.

Compose winners in dependency order and re-evaluate after each addition:

```
baseline → +A7 preprocessing → +A1 spotting → +A2 reasoning → +A4 multilingual(synergistic)
         → +A5 best placement → +A6 HPO
```

Expected: a **monotone staircase** on the headline metric (e.g., InfoVQA / composite), with each
step's height = that component's marginal contribution. Plotted by `scripts/plot_ablation.py`
(reads `results/ablation_results.json`). A flat or negative step is itself a finding (component
doesn't help / interferes → drop it).

![Ablation staircase](figures/ablation_staircase.png)
![Per-ablation marginal gain](figures/ablation_deltas.png)

*(Figures above use illustrative DEMO numbers until the real runs land; regenerate with
`python scripts/plot_ablation.py` once `results/ablation_results.json` holds measured scores.)*

The A4 language-transfer and A5 placement experiments produce:

![Language transfer heatmap](figures/ablation_lang_transfer.png)
![LoRA placement gains](figures/ablation_lora_placement.png)

## 10. Dependency / relationship diagram

Ablations aren't independent: preprocessing gates recognition (A7→A1), spotting and reasoning are
parallel supervision signals that both feed the composite, placement (A5) and HPO (A6) are
*how* we train any of them, and multilingual (A4) is orthogonal data mixing.

![Ablation relationship](figures/ablation_relationship.png)

## 11. Data generation for the ablations

Each ablation arm is a **synthetic dataset variant** whose ground truth carries the factor being
varied. The generator stores every factor as typed GT (`DocSample` DTO) and exposes one knob per
factor in a single config, so an arm differs from its control in exactly one factor family:

- **Read:** [`synthetic_data_dto.md`](synthetic_data_dto.md) — the GT DTO, the realism/distribution
  matching (digital-native PDF → exact boxes; degradation = acquisition modality; language mix), and
  the **ablation-factor → DTO/config mapping** (A1→`bbox`, A2→`rationale`, A4→`language`, A7→render).
- **Generate:** `python scripts/make_realistic_cases.py --config configs/synth_data.yaml
  --ablation <id>` — `configs/synth_data.yaml` holds `base:` + `ablation_overrides:` (A1/A2/A3/A4/A7).

## 11b. Public-data realization of the arms (UDD)

Every data-side arm now also runs on **public data** via UDD
([`unified_loader.md`](unified_loader.md)) — same one-factor-at-equal-N control, real documents
instead of synthetic templates. `scripts/run_udd_ablation.py` composes each arm's training mix from
the equal-N per-task/per-language sets and drives `run_ablation.py --arm public`; every run scores
BOTH the synthetic probe suite and the **UDD public heldout fold** (the `fold` column: deterministic
~90/10 split keyed by *image identity*, so no held-out image leaks into training via a sibling QA).

| Arm | UDD realization (equal totals) | Status / gaps |
| --- | --- | --- |
| A0 scale | `--arm A0 --train-jsonl data/udd_tasks/all.jsonl` (existing public A0 path) | ✅ |
| A1 spotting | `A1_spotting_on` = vqa+kie+grounding thirds vs `A1_spotting_off` = vqa+kie halves; grounding rows from DocLayNet/PubLayNet/OmniDocBench via `to_grounding_samples()` | ✅ |
| A2 reasoning | `A2_reason_chain` vs `A2_reason_answer`: the SAME geometry-derived records with rationale-target vs position-only target (`derive_spatial_reasoning(style=…)`) | ✅ (derived rationales — spatial subset of A2) |
| A3 combination | `A3_base` / `A3_spot` / `A3_reason` / `A3_spot_reason`, equal totals | ✅ |
| A4 language mix | `A4_en` vs `A4_en_{ko,ar,zh,id}` from `lang_<code>.jsonl` | ⚠️ partial: no es/ja rows yet (MTVQA streams language-ordered → ar-heavy; ja needs a deeper scan or a stride sampler); zh/id pools are small — the runner warns when an arm falls short of the equal-N target |
| A5 placement | `A5_{vision,connector,llm_attn,llm_mlp}`: the FIXED composite mix (vqa+kie+grounding+rationale quarters), placement varied | ✅ (4 composed arms) |
| A6 HPO | `A6_r{8,16,32,64}` (α=2r) on the same fixed composite mix; `--lora-r/--lora-alpha/--lr` now thread through `run_ablation` | ✅ (4 composed arms) |
| A7 preprocessing | `--max-image-long-side` knob on any arm; `image_width/height` columns support small-text slicing | ✅ |

```bash
# 1) equal-N sets + heldout fold + derived A2 pair (offline)
python scripts/build_task_trainsets.py --per-task 300 --merge-qa --derive-spatial-reasoning
python scripts/build_task_trainsets.py --out data/udd_langs --group-by language --per-task 300
# 2) compose + inspect the 13 arm mixes without a GPU
python scripts/run_udd_ablation.py --arm A1 A2 A3 A4 --count 300 --dry-run
# 3) train + eval (GPU); results -> docs/results/udd_ablation_results.json under U-<arm>
python scripts/run_udd_ablation.py --arm A1 A2 --count 300 --steps 300
```

The synthetic arms (§11) remain the controlled-GT reference; the UDD arms answer the external-
validity question — *do the same factors help on real documents?*

Beyond these, the corpus columns support **seven more candidate arms** (resolution strata,
QA-packing, output style, numeric reasoning, grounding-box difficulty, form factor, license
filter) — bucket support and recipes are measured in
[`../results/udd_ablation_features.md`](../results/udd_ablation_features.md); wiring verification
(composition, leakage, A2 control, mock eval) lives in
[`../../notebooks/udd_ablation.ipynb`](../../notebooks/udd_ablation.ipynb).

## 12. Run one arm end-to-end (GPU)

```bash
# baseline (eval only), then the A1 spotting curriculum on the default LFM base:
python scripts/run_ablation.py --arm baseline
python scripts/run_ablation.py --arm A1_spotting_on --placement connector --heldout-seed 999
python scripts/run_lora_placement_sweep.py --dry-run
```
`run_ablation.py` generates the arm's data → LoRA-fine-tunes (`docvlm_eval.finetune.lora_vlm`,
placement resolved by introspection) → evaluates on the probe suite → appends to
`docs/results/ablation_results.json`, which
[`notebooks/finetune_ablation.ipynb`](../../notebooks/finetune_ablation.ipynb) reads for the
**side-by-side before/after** of both models.

## 13. Outputs & reproducibility
- `configs/synth_data.yaml` — data-generation config: `base` knobs + per-ablation `ablation_overrides`
  (binds to `docvlm_eval.synth.dto.GenConfig`); one file = one dataset variant.
- `configs/ablations.yaml` — declarative ablation registry (two bases, factor, control, metric, gap).
- `scripts/run_ablation.py` — gen → LoRA train → eval → record (per model, per arm).
- `scripts/run_lora_placement_sweep.py` — six-job paired vision versus vision+connector
  confirmatory sweep with a fail-closed adapter-parameter budget.
- `scripts/analyze_lora_placement_sweep.py` — paired bootstrap aggregation and promotion gates for
  the completed confirmatory sweep.
- `docvlm_eval.finetune.lora_vlm` — model-agnostic LoRA (chat-template) + the A5 placement resolver.
- `scripts/plot_ablation.py` — staircase + Δ bars + transfer heatmap from the results JSON.
- Every variant is scored by the **same** eval pipeline, so the staircase is apples-to-apples.
