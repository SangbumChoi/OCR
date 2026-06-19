# Capability axes for document VLMs — the axis catalogue, probes, and prompt plan

Benchmark *names* hide what a model can actually do. To compare sub-1B document VLMs meaningfully we
score them along a fixed set of **capability axes**, isolate each with a small controlled probe, and
report a per-model **capability vector** (`T1 … H7`). This file is the single source for the axes,
the probes, and the prompts. Probes live under
[`data/probes/`](../../data/probes/README.md); prompts in
[`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).

## Axis catalogue (three families, sequential codes)

Axes are grouped into three **families**; within a family the codes run **sequentially** (`L1…L4`,
`H1…H7`). **T** and **L** are clean single-mechanism families; **H** is a deliberate **composite**
(it holds everything that needs *more than reading one field* — content reasoning, chart-value, and
context). The family letter *is* the code letter, so a model's result reads as one ordered vector.

| Family | Code | Axis | What it tests | Probe item | Metric |
| ------ | ---- | ---- | ------------- | ---------- | ------ |
| **T · Text** *(read the value)* | **T1** | text recognition | read an exact printed string | `cap_text` | NED / ANLS |
| | **T2** | key-value extraction | one field's value from a single region | `cap_kie` | ANLS |
| **L · Location & space** *(where)* | **L1** | grounding / spotting | bounding box of a named element | `cap_ground` | IoU |
| | **L2** | absolute region | which quadrant an element is in | `sp_quad_*` | signal |
| | **L3** | relative position | above/below, with a counterfactual | `sp_relpos_*` | signal |
| | **L4** | spatial tracking | box follows the element top→bottom | `sp_box_*` | signal |
| **H · Reasoning & context** *(composite)* | **H1** | content reasoning — sum | arithmetic over read values | `cap_integ_sum` | relaxed_acc |
| | **H2** | content reasoning — compare | comparison over read values | `cap_integ_rel` | exact |
| | **H3** | chart-value *(graphic perception)* | read a value off a chart | `cap_chart` | relaxed_acc |
| | **H4** | context — consistency | items vs printed total | `ctx_consistency_*` | signal |
| | **H5** | context — anti-hallucination | absent field → "none" | `ctx_absence` | signal |
| | **H6** | context — disambiguation | Total vs look-alike Subtotal | `ctx_distractor` | signal |
| | **H7** | context — cross-reference | "Bill to" name → amount (counterfactual) | `ctx_xref_*` | signal |

This replaces an earlier framing that listed *Text/Location abilities* and *KIE/integrative/chart
"natures"* in separate tables, which **double-listed** the same axes. The grouping is by *what the
answer depends on*:

- **T — text only:** the answer is a string read off one place. Single mechanism.
- **L — location/space only:** the answer is *where* (a box, `L1`) or a *positional* relation
  (`L2–L4`) — no value computation.
- **H — reasoning & context (composite):** a deliberate bucket, **not one mechanism**. It holds
  arithmetic/comparison over values (`H1/H2`), **chart-value graphic perception** (`H3`, really
  perception, parked here for convenience), and cross-region **context** (`H4–H7`). If H is ever
  split, the context block `H4–H7` is the natural sub-family to peel off.

> **Boundaries (where they matter).** The distinct operations on the *same* invoice stay separate,
> never scored as one: "**add** the prices" = `H1` (compute); "**do** they **add up to** the total?"
> = `H4` (context verify); "**where** is the total?" = `L1` (locate). And **`H3` chart-value ⟂ pure
> figure understanding** — `H3` reads a *value*; general figure/diagram comprehension is **out of
> scope** here (left to CharXiv/MathVista if ever needed).

## Probes & prompts

Two controlled probes cover the catalogue, each single-purpose so a failure is unambiguous:

- **[`capability_probe`](../../data/probes/capability_probe)**
  ([generator](../../scripts/make_capability_probe.py)) — the **measured-score** axes
  **`T1 T2 H1 H2 H3 L1`**, rendered with exact GT incl. pixel boxes.
- **[`spatial_context_probe`](../../data/probes/spatial_context_probe)**
  ([generator](../../scripts/make_spatial_context_probe.py)) — the **signal** axes **`L2–L4`**
  (spatial) and **`H4–H7`** (context), each paired with a shortcut **control** (next section).

**Prompt rules** (templates in [`capability_prompts.yaml`](../../configs/capability_prompts.yaml)):
**constrain verbosity** — append *"Answer concisely with only the value."* (small VLMs answer in
sentences, sinking ANLS/exact even when right); and **ask for a machine-parseable box** for `L1`
(*"… as [x1,y1,x2,y2] in pixel coordinates. The image is W×H."*).

## L1 — grounding/spotting: the fair-comparison problem

Models don't share an interface for "where is X", so a naive prompt would score chat VLMs at 0 by
*interface mismatch*, not inability:

| Model family | Native spotting? | How we get a box |
| ------------ | ---------------- | ---------------- |
| **PaddleOCR-VL** | ✅ built-in spotting/layout task (elements + polygons) | matching element's polygon → axis-aligned bbox |
| **Florence-2** | ✅ task tokens `<OD>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>` | box of the matching label (already pixels) |
| **GOT-OCR2.0** | ✅ region / fine-grained OCR (read-with-coordinates) | convert its coords → [x1,y1,x2,y2] |
| **SmolVLM / InternVL / LLaVA-OV / Ovis / H2OVL** | ❌ general chat VLM, no spotting head | text-prompted box via the normalised instruction |

**Fairness fix:** give every model the *same* normalised instruction; additionally run the
spotting-capable models' native task and **map every output back to the same [x1,y1,x2,y2] pixels**
(polygon→bbox, task-token→box, loc-token→box); score all with the **same IoU**
([`metrics/grounding.py`](../../src/docvlm_eval/metrics/grounding.py); the parser absorbs
0–1 / 0–1000 / pixel / `loc_<n>` formats). The expected finding is real: general small VLMs produce
weak/no boxes; spotting-trained models dominate.

## L2–L4 / H4–H7 — control-paired signals (not raw score)

These axes are easy to fake (language priors, position bias, binary guessing, rubber-stamping,
hallucination), so each is paired with a **control** a shortcut would *fail*; the reported **signal**
= (passes the test) **AND** (passes the control), making the null "uses a shortcut" falsifiable.
Computed by [`analyze_probe_signals.py`](../../scripts/analyze_probe_signals.py).

| Code | Test | Shortcut it must beat | PASS criterion |
| ---- | ---- | --------------------- | -------------- |
| **L2** | "Which quadrant has ZEBRA?" ×4 | position bias (always one quadrant) | acc **> 25%** & not all-same; ≥ 3/4 |
| **L3** | "TOTAL above/below items?" on **normal** + **counterfactual** (total moved up) | prior "total = bottom" | correct on **both**; prior-reliance gap = acc(normal) − acc(cf) ≈ 0 |
| **L4** | bbox of ANCHOR at top/mid/bottom | constant-box prior | center-y correlation **r > 0.8** & mean IoU > 0.3 |
| **H4** | "Do items add up to TOTAL?" **consistent** + **inconsistent** | rubber-stamp "yes" | catches the inconsistent case (control = no) |
| **H5** | "What is the discount?" with **no discount field** | invent a number | answers **"none"**; any number = fail |
| **H6** | "TOTAL (not subtotal)?" with a look-alike Subtotal | grab nearest number | returns Total, not Subtotal |
| **H7** | "How much does 'Bill to' owe?" header **Bob** vs **Alice** | answer invariant to header | answer **flips** Bob→45, Alice→80 (both correct) |

`L3`'s counterfactual is the key idea — answering "below" on a *top*-total invoice is pattern-matching,
not seeing. `L4` makes grounding *relational* (a box that moves with the element shows tracking even
at low IoU). `H4` defeats always-"consistent"; `H5` measures honesty under absence; `H7` measures
whether the answer tracks the only relevant token.

## Results — the capability vector (SmolVLM, CPU)

One row per model across every axis (measured-score axes show the metric; signal axes show ✅/❌ =
clears the control). From `run_matrix.py` + `analyze_probe_signals.py`.

> **Illustrative, not conclusive.** This is **2 models, ~1 controlled item per axis, on clean
> synthetic renders** — enough to demonstrate the framework and expose shortcut failures, not a
> powered benchmark. 1B / spotting models are **not run here** (need a GPU); T scores are a *best
> case* (crisp synthetic text); and the row mixes **units** (numeric vs pass/fail), so compare
> *within* an axis, not across. Treat cells as direction, not measurement.

| model        | T1 | T2 | L1 | L2 | L3 | L4 | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
| ------------ | :-: | :-: | :--: | :-: | :-: | :-: | :--: | :--: | :-: | :-: | :-: | :-: | :-: |
| SmolVLM-256M | 0.93 | 0.94 | **0.00** | ❌ | ❌ | ❌ | **0.00** | **0.00** | 1.00 | ❌ | ❌ | ✅ | ❌ |
| SmolVLM-500M | 0.93 | 0.94 | **0.00** | ✅ | ❌ | ❌ | **1.00** | **0.00** | 1.00 | ❌ | ❌ | ✅ | ❌ |

**Reading the vector:**
- **T (text) is solved** even at 256M (≈ .93–.94) — reading a field is the easy family.
- **H content reasoning splits by size:** `H1` (sum) emerges 256M→500M; `H2` (compare) fails for
  both; `H3` (chart-value) works on a clean chart.
- **L is the gap:** `L1` grounding fails for both; only `L2` (absolute region) emerges at 500M;
  `L3`/`L4` fail (`L3`'s control even shows a `prior_reliance_gap = −1.0` — both answer "above" for
  *every* layout). General small VLMs lack a location head → prefer a spotting-trained model.
- **H context mostly fails:** only `H6` (disambiguation) passes; `H4` rubber-stamps "consistent",
  `H5` invents a discount, `H7` ignores the "Bill to" name — gaps invisible to plain accuracy,
  visible only under the paired controls.

**Bottom line:** at ≤0.5B you get a **T-strong reader** with partial **H** (sum, chart) but **no L
(location/space)** and **weak context**.

## Model selection

Pick by the **family the use case needs**:

- **Read a known field** → **T**; a 256–500M VLM is viable.
- **Totals / cross-field math** → **H1/H2**; 1B models (InternVL) separate from 256M here.
- **Charts** → **H3** (ChartQA-strong models).
- **Localisation / structured parsing** → the **L family** → prefer a spotting-capable model
  (PaddleOCR-VL / Florence-2 / GOT); general VLMs lack it.
- **Auditable / safety-critical extraction** → **L1** + context (esp. `H5` abstain) — see below.

Selection is therefore a **capability vector per model**, not a single score.

## Output-side requirements per document type (deployment views of L1 & H5)

The [document-type taxonomy](document_type_taxonomy.md) lists only *input* stressors. The two
requirements that gate deployment are **deployment-lens *views* of axes already in the catalogue,
not new axes** — they re-use **L1** and **H5** rather than double-count:

- **Spotting / grounding** = **L1**, viewed as "can we audit *where* each value came from?" (IoU).
- **Correct-or-abstain** = **H5** (anti-hallucination), viewed as a per-type deployment gate: when a
  field is **absent / redacted / illegible**, the right answer is "not present" / "[redacted]";
  inventing a value is the worst failure. Scored by **abstain accuracy** (the `abstain` probes in
  [`data/probes/realistic_cases`](../../data/probes/realistic_cases/README.md)) + **ECE calibration**
  ([`metrics/calibration.py`](../../src/docvlm_eval/metrics/calibration.py)).

These are *why* a type's **anchor metric** in the taxonomy reads "… + spotting IoU" or "… + abstain".
Types where each is a **primary** requirement (✓✓ = critical):

| Document type               | Spotting (L1) | Abstain (H5) |
| --------------------------- | :-----------: | :----------: |
| Invoice / receipt           | ✓             | ✓            |
| Bank stmt / payslip         | ✓             | ✓            |
| Contract / NDA              | ◐             | ✓            |
| **ID / passport / license** | ✓             | ✓✓           |
| Certificate / diploma       | ◐             | ✓            |
| **Cheque**                  | ✓             | ✓            |
| **Prescription**            | ✓             | ✓ (illegible → abstain) |
| Ticket / boarding pass      | ✓             | —            |
| Map / floor plan            | ✓             | —            |
| Ledger / census             | ✓             | —            |
| LCD / meter / 7-seg         | ✓             | —            |
| UI / chat / code screenshot | ✓             | —            |
| **Form w/ checkboxes**      | ✓             | ✓            |
| **Redacted document**       | ✓             | ✓✓           |
| Chemical / circuit diagram  | ✓             | —            |

The **identity/administration** family (ID, cheque, prescription, redaction) is where abstain
matters most — a confidently-wrong passport number is worse than "unreadable". Secondary (◐) needs
are captured in each type's anchor-metric column in the taxonomy.
