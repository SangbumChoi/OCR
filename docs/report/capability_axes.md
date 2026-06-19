# Capability axes for document VLMs — the full axis catalogue, probes, and prompt plan

Benchmark *names* hide what a model can actually do. To select and compare sub-1B document VLMs
meaningfully we score them along **orthogonal capability axes**, isolate each axis with a small
controlled probe, and compare models as a **capability vector** (T1, T2, L1, S1 … C4). This file is
the single source for all axes. Probes: [`data/probes/capability_probe`](../../data/probes/capability_probe)
and [`data/probes/spatial_context_probe`](../../data/probes/spatial_context_probe); prompts:
[`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).

## Capability axis catalogue (three families)

Axes are grouped into **three families**. **T** and **L** are clean single-mechanism families; **H**
is a deliberate **composite** (content reasoning + chart-value + context) — see the note below, it is
*not* one mechanism. Codes are stable per axis, but **the code letter is not always the family
letter**: `S1–S3` (spatial) and `C1–C4` (context) keep their historical codes, so the family is the
*grouping*, not the code prefix. A model's result is the per-axis vector across all codes.

| Family | Code | Axis | What it tests | Probe item | Metric |
| ------ | ---- | ---- | ------------- | ---------- | ------ |
| **T · Text** *(read the value — pure text)* | **T1** | text recognition | read an exact printed string | `cap_text` | NED / ANLS |
| | **T2** | KIE-localized | one field's value from a single region | `cap_kie` | ANLS |
| **L · Location & spatial** *(where — position)* | **L1** | grounding / spotting | bounding box of a named element | `cap_ground` | IoU |
| | **S1** | absolute region | which quadrant (vs position bias) | spatial `S1` | signal |
| | **S2** | relative position | above/below + **counterfactual** | spatial `S2` | signal |
| | **S3** | spatial tracking | box center tracks the element | spatial `S3` | signal |
| **H · Reasoning & context** *(composite — see note)* | **H1** | content reasoning — sum | arithmetic over read values | `cap_integ_sum` | relaxed_acc |
| | **H2** | content reasoning — compare | comparison over read values | `cap_integ_rel` | exact |
| | **H3** | chart-value *(graphic perception)* | read a value off a chart | `cap_chart` | relaxed_acc |
| | **C1** | context — consistency | items vs printed total (vs rubber-stamp) | context `C1` | signal |
| | **C2** | context — anti-hallucination | absent field → "none" | context `C2` | signal |
| | **C3** | context — disambiguation | Total vs look-alike Subtotal | context `C3` | signal |
| | **C4** | context — cross-reference | "Bill to" name → amount (counterfactual) | context `C4` | signal |

**Why three families (and not the old "two abilities" + "answer natures" split):** the previous
framing listed *Text/Location* abilities and *KIE/integrative/chart* "natures" separately, which
**double-listed** the same axes. The grouping is by *what the answer depends on*:

- **T — text only:** the answer is a string you read off one place. Single mechanism.
- **L — location/space only:** the answer is *where* (a box) or a *positional* relation — no value
  computation. (L1 grounding + S1–S3 spatial belong together: both are "where", not "what".)
- **H — reasoning & context (composite):** a **deliberate bucket**, not one mechanism — it holds
  everything that needs *more than reading one field*: arithmetic/comparison over values (H1/H2),
  **graphic value perception** (H3, which is really perception, parked here for convenience), and
  cross-region **context** reasoning (C1–C4). If H ever needs to split, C1–C4 (context) is the
  natural sub-family to peel off.

> **Boundaries (where they matter).** *Within the H bucket* the distinct operations are kept
> separate, never scored as one: "**add** the prices" = H1 (compute); "**do** they **add up to** the
> total?" = C1 (context verify); and "**where** is the total?" = L1 (locate, family L). Same invoice,
> three codes. And **H3 chart-value ⟂ pure figure understanding**: H3 reads a *value* off a chart;
> general figure/diagram comprehension (scientific-figure reasoning, arbitrary diagrams) is **not in
> scope** here (left to CharXiv/MathVista if ever needed).

## The controlled probes

Two probes cover the catalogue, each single-purpose so a failure is unambiguous:

- **[`capability_probe`](../../data/probes/capability_probe)** (generator
  [`make_capability_probe.py`](../../scripts/make_capability_probe.py)) — the **measured-score** axes
  **T1, T2, H1, H2, H3, L1**. Rendered with exact GT incl. pixel boxes, so the value/box are exact.
- **[`spatial_context_probe`](../../data/probes/spatial_context_probe)** (generator
  [`make_spatial_context_probe.py`](../../scripts/make_spatial_context_probe.py)) — the **signal**
  axes **S1–S3, C1–C4**, each paired with a shortcut **control** (below).

**Why custom + synthetic?** Public sets entangle axes (DocVQA mixes T2 + H1); the probes isolate
*one* axis per item, we control the layout so sums/boxes are exact, and they run through the *same*
pipeline + metrics as the other benchmarks (catalogued in family F1 under
[`data/probes/`](../../data/probes/README.md)) — directly comparable.

## Prompt plan — what goes in the prompt

Every axis uses a prompt that *elicits* the capability and *constrains* the output so it is scorable;
templates in [`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml). Two rules
from real runs:

- **Constrain verbosity.** Small VLMs answer in sentences ("Pinterest has a heavy female audience.")
  which sink ANLS/exact even when right → append *"Answer concisely with only the value."*
- **Ask for a machine-parseable box** (L1): *"Return the bounding box … as [x1,y1,x2,y2] in pixel
  coordinates. The image is W×H."*

## L1 — location/spotting: the fair-comparison problem

The crux of comparing OCR engines against general VLMs: **models don't share an interface for
"where is X".**

| Model family                                     | Native spotting?                                                                 | How we get a box                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **PaddleOCR-VL**                                 | ✅ built-in **spotting/layout task** (prompt already returns elements + polygons) | take the matching element's polygon → axis-aligned bbox |
| **Florence-2**                                   | ✅ **task tokens** `<OD>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>`   | take the box of the matching label (already pixels)     |
| **GOT-OCR2.0**                                   | ✅ **region/fine-grained OCR** (read-with-coordinates)                            | convert its coords → [x1,y1,x2,y2]                      |
| **SmolVLM / InternVL / LLaVA-OV / Ovis / H2OVL** | ❌ general chat VLM, no spotting head                                             | **text-prompted box** via the normalised instruction    |

**The fairness fix (normalisation layer):** (1) give **every** model the *same* normalised
instruction ("return [x1,y1,x2,y2] in pixels"); (2) for spotting-capable models also run their
native task and **map the output back to the same [x1,y1,x2,y2] pixels** (polygon→bbox,
task-token→box, loc-token→box); (3) score both with the **same IoU** metric
([`metrics/grounding.py`](../../src/docvlm_eval/metrics/grounding.py); permissive parser absorbs
0–1 / 0–1000 / pixel / `loc_<n>` formats). This makes "can it localise?" apples-to-apples — and the
expected finding is real: general small VLMs produce weak/no boxes, spotting-trained models dominate.

## S1–S3 / C1–C4 — spatial & context: signal, not score

These axes are easy to fake (language priors, position bias, binary guessing, rubber-stamping,
hallucination), so each is paired with a **control** a shortcut would *fail*, and we report a
**signal** = (passes the test) **AND** (passes the control). The null hypothesis "uses a shortcut"
is thus falsifiable. Computed by [`analyze_probe_signals.py`](../../scripts/analyze_probe_signals.py).

| Code | Hypothesis | Test | Shortcut it must beat | Signal / criterion |
| ---- | ---------- | ---- | --------------------- | ------------------ |
| **S1** | absolute region | "Which quadrant has ZEBRA?" ×4 | position bias (always one quadrant) | acc **> 25%** & not all-same; ≥ 3/4 |
| **S2** | relative position from *perception* | "TOTAL above/below items?" on **normal** + **counterfactual** (total moved up) | prior "total = bottom" | correct on **both**; prior-reliance gap = acc(normal) − acc(cf) ≈ 0 |
| **S3** | box tracks element | bbox of ANCHOR at top/mid/bottom | constant-box prior | center-y correlation **r > 0.8** & mean IoU > 0.3 |
| **C1** | cross-region consistency | "Do items add up to TOTAL?" **consistent** + **inconsistent** | rubber-stamp "yes" | catches the inconsistent case (control=no) |
| **C2** | absence / anti-hallucination | "What is the discount?" with **no discount field** | invent a number | answers **"none"**; any number = fail |
| **C3** | disambiguation by context | "TOTAL (not subtotal)?" with look-alike Subtotal | grab nearest number | returns Total, not Subtotal |
| **C4** | cross-reference + counterfactual | "How much does 'Bill to' owe?" header **Bob** vs **Alice** | answer invariant to header | answer **flips** Bob→45, Alice→80 (both correct) |

S2's counterfactual is the key idea — answering "below" on a *top*-total invoice is pattern-matching,
not seeing. S3 makes grounding *relational* (a box that moves with the element shows tracking even at
low IoU). C1 defeats always-"consistent"; C2 measures honesty under absence; C4 measures whether the
answer tracks the only relevant token.

## Results — the capability vector (SmolVLM, CPU)

One row per model across **all** axes. Measured-score axes (T/L1/H1–H3) show the metric value;
signal axes (S/C) show ✅/❌ = clears the shortcut control. From
`scripts/run_matrix.py` + `scripts/analyze_probe_signals.py`.

> **Read as illustrative, not conclusive.** This is **2 models (SmolVLM 256M/500M), ~1 controlled
> item per axis, on clean *synthetic* renders** — enough to demonstrate the framework and expose
> shortcut failures, **not** a powered benchmark. The 1B / spotting models are **not run here** (need
> a GPU — `scripts/run_all.sh`); the T scores are a *best case* (crisp synthetic text, no
> degradation); and the row mixes **units** (numeric scores for T/L1/H vs pass/fail signals for S/C),
> so compare *within* an axis, not across. Treat the cells as direction, not measurement.

| model        | T1 text | T2 kie | L1 ground | S1 | S2 | S3 | H1 sum | H2 cmp | H3 chart | C1 | C2 | C3 | C4 |
| ------------ | :-----: | :----: | :-------: | :-: | :-: | :-: | :----: | :----: | :------: | :-: | :-: | :-: | :-: |
| SmolVLM-256M | 0.93    | 0.94   | **0.00**  | ❌  | ❌  | ❌  | **0.00** | **0.00** | 1.00   | ❌  | ❌  | ✅  | ❌  |
| SmolVLM-500M | 0.93    | 0.94   | **0.00**  | ✅  | ❌  | ❌  | **1.00** | **0.00** | 1.00   | ❌  | ❌  | ✅  | ❌  |

**Reading the vector:**
- **T (text) is solved** even at 256M (T1/T2 ≈ .93–.94): reading a field is the easy family.
- **H content reasoning splits by size.** H1 (sum) emerges 256M→500M (256M miscalculates
  `$45+$80+$20.50=$125`, 500M gets 145.50); **H2 (compare) fails for both** (both say "Widget A",
  truth "Gadget B"). H3 (chart-value) works on a clean chart (both 1.00).
- **L is missing.** L1 grounding fails for both (256M echoes the image size, 500M emits a nonsense
  box); S1 absolute spatial **emerges at 500M** (acc .75) but S2/S3 fail. The whole **L family is the
  gap** for general small VLMs → prefer a spotting-trained model (PaddleOCR-VL / Florence-2 / GOT).
- **C context mostly fails.** Only C3 (disambiguation) passes; C1 (rubber-stamps "consistent" even
  when the total is wrong), C2 (invents a discount), C4 (ignores the "Bill to" name) all fail — and
  these are invisible to single-example accuracy, visible only under the paired controls. (S2's
  control even exposes a `prior_reliance_gap = −1.0`: both answer "above" for *every* layout.)

**Bottom line:** at ≤0.5B you get a **T-strong reader** with partial **H** (sum, chart) but **no L
(location/space)** and **weak C (context)**.

## How this informs model selection

Pick by the **family the use case needs**:

- **Read a known field** → **T** dominates; a 256–500M VLM is viable.
- **Totals / cross-field math** → needs **H1/H2**; 1B models (InternVL) separate from 256M here.
- **Charts** → needs **H3** (ChartQA-strong models).
- **Localisation / structured parsing** → needs the **L family** → prefer a spotting-capable model
  (PaddleOCR-VL / Florence-2 / GOT); general VLMs lack it.
- **Auditable / safety-critical extraction** → needs **C** (consistency, abstain) + L1 — see below.

Selection is therefore a **capability vector per model**, not a single score.

## Output-side requirements per document type (L1 spotting & abstain)

The [document-type taxonomy](document_type_taxonomy.md) lists only *input* stressors; the
**output-side** requirements that gate deployment are summarised here. These are **deployment-lens
*views* of axes already in the catalogue, not new axes** — so they intentionally re-use **L1** and
**C2** rather than double-count:

- **Spotting / grounding** — this *is* **L1** (the grounding axis), viewed as "can we audit *where*
  each value came from?"; scored by IoU (the L1 normalisation above makes it fair across OCR engines
  and chat VLMs).
- **Hallucination control (correct-or-abstain)** — this *is* **C2** (anti-hallucination), viewed as a
  per-type deployment gate. When a field is **absent / redacted / illegible**, the correct answer is
  "not present" / "[redacted]"; inventing a value is the worst failure. Scored by **abstain accuracy**
  (the `abstain` probes in [`data/probes/realistic_cases`](../../data/probes/realistic_cases/README.md))
  + **ECE calibration** ([`metrics/calibration.py`](../../src/docvlm_eval/metrics/calibration.py)).

These are *why* a type's **anchor metric** in the taxonomy reads "… + spotting IoU" or "… + abstain".
Types where each is a **primary** requirement (✓✓ = critical):

| Document type               | Spotting (L1 IoU) | Abstain (correct-or-abstain) |
| --------------------------- | :---------------: | :--------------------------: |
| Invoice / receipt           | ✓                 | ✓                            |
| Bank stmt / payslip         | ✓                 | ✓                            |
| Contract / NDA              | ◐                 | ✓                            |
| **ID / passport / license** | ✓                 | ✓✓                           |
| Certificate / diploma       | ◐                 | ✓                            |
| **Cheque**                  | ✓                 | ✓                            |
| **Prescription**            | ✓                 | ✓ (illegible → abstain)      |
| Ticket / boarding pass      | ✓                 | —                            |
| Map / floor plan            | ✓                 | —                            |
| Ledger / census             | ✓                 | —                            |
| LCD / meter / 7-seg         | ✓                 | —                            |
| UI / chat / code screenshot | ✓                 | —                            |
| **Form w/ checkboxes**      | ✓                 | ✓                            |
| **Redacted document**       | ✓                 | ✓✓                           |
| Chemical / circuit diagram  | ✓                 | —                            |

The **identity/administration** family (ID, cheque, prescription, redaction) is where abstain matters
most — a confidently-wrong passport number is worse than "unreadable". Secondary (◐) needs are
captured in each type's anchor-metric column in the taxonomy.
