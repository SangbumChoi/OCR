# Capability axes for document VLMs — the full axis catalogue, probes, and prompt plan

Benchmark *names* hide what a model can actually do. To select and compare sub-1B document
VLMs meaningfully we evaluate along **capability axes**, build small **controlled probes** that
isolate each axis, and define **what to put in the prompt** to elicit each one. This single file
is the rationale for **all** the axes:

- **perception + content reasoning** (read / compute / compare / chart-value / locate) →
  [`data/probes/capability_probe`](../../data/probes/capability_probe);
- **spatial & context understanding** (position, consistency, absence/honesty, disambiguation,
  cross-reference) → [`data/probes/spatial_context_probe`](../../data/probes/spatial_context_probe);
- **output-side requirements** (spotting + correct-or-abstain).

Prompts: [`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).

## Two top-level abilities

A document model is really two abilities stacked:

| Ability                            | Question it answers   | How we elicit it               | How we score it        |
| ---------------------------------- | --------------------- | ------------------------------ | ---------------------- |
| **Text understanding**     | "What does it *say*?" | read / answer prompts          | ANLS / exact / relaxed |
| **Location understanding** | "*Where* is it?"      | "return the bounding box of X" | **IoU** (grounding)    |

These are genuinely different heads: a model can read a field perfectly yet be unable to point
at it (and vice-versa for a detection-first model). Most leaderboards score only the first;
production document parsing (reading order, field localisation, table cells, redaction) needs
the second. So we score **both**, with a dedicated grounding metric
([`metrics/grounding.py`](../../src/docvlm_eval/metrics/grounding.py), IoU with a permissive box
parser).

## When the output is text, *what kind* of text?

Not all text answers are equal. Three natures, in increasing difficulty:

| Nature                    | What it demands                                                        | Example (probe sample)                                                                  | Metric              |
| ------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------- |
| **KIE-localized**         | read **one region**, give a clear field value                          | "What is the vendor name?" → `Acme Corporation`                                         | anls                |
| **Content reasoning** (integrative) | **combine several read *values*** by *arithmetic / comparison* — layout-agnostic | "Add up the line-item prices." → `145.50`; "Which item is most expensive?" → `Gadget B` | relaxed_acc / exact |
| **Chart-value reading**   | read a *value* off a chart/plot — graphic value perception            | "What is the value of bar B?" → `70`                                                    | relaxed_acc         |

The jump from *localized* to *content reasoning* is where small LMs fail: it needs multi-region
attention + arithmetic/comparison, not just OCR. The *chart-value* nature additionally needs
graphic perception, not text at all.

> **Two boundaries kept strict.**
> 1. **Content reasoning ⟂ spatial/context.** The axes above are **perception + content reasoning**
>    (read values, then compute/compare them) — they are *layout-agnostic*. **Spatial & context
>    understanding** — *where* regions are and how they relate (relative position, cross-region
>    *consistency/verification*, absence/honesty, disambiguation, cross-reference) — is a **separate,
>    independent axis** (see [Spatial & context understanding](#spatial--context-understanding-independent-axis)
>    below; probe [`data/probes/spatial_context_probe`](../../data/probes/spatial_context_probe)). The
>    two probes share **no items**: capability_probe *computes/compares/reads values*; the
>    spatial/context probe tests *position + relation/context*. (e.g. "**add** the prices" is content
>    reasoning here; "**do** the items **add up to** the total?" is a context *verification* test there
>    — compute vs verify.)
> 2. **Chart-value reading ⟂ pure figure understanding.** This axis is reading a *value* off a chart.
>    **General figure/diagram comprehension** (scientific-figure reasoning, arbitrary diagrams) is
>    **not mandatory here** — it is out of scope for the capability probe (left to dedicated
>    figure benchmarks such as CharXiv/MathVista if needed).

## The custom capability probe (rationale)

[`data/probes/capability_probe`](../../data/probes/capability_probe) is rendered by
[`scripts/make_capability_probe.py`](../../scripts/make_capability_probe.py) so the ground truth
— **including exact pixel boxes** — is known. Six samples, one per axis:

| Sample          | Axis                       | Prompt intent              | Gold                   |
| --------------- | -------------------------- | -------------------------- | ---------------------- |
| `cap_text`      | text-recognition           | direct read                | `INV-2025-0042`        |
| `cap_kie`       | kie-localized              | single-field KIE           | `Acme Corporation`     |
| `cap_integ_sum` | content-reasoning (sum)    | value arithmetic           | `145.50`               |
| `cap_integ_rel` | content-reasoning (compare)| value comparison           | `Gadget B`             |
| `cap_chart`     | chart-value                | chart value read           | `70`                   |
| `cap_ground`    | location-grounding         | text-prompted bounding box | box `(40,335,270,359)` |

All six axes are **perception + content reasoning** (read / compute / compare / locate). None tests
*relative position* or *cross-region context* — those are deliberately left to the independent
spatial/context probe (see the boundary note above), so each probe stays single-purpose.

**Why custom + synthetic?** (a) Public sets entangle axes (DocVQA mixes localized + content
reasoning); the probe isolates *one* axis per item so a failure is unambiguous. (b) We control the
layout, so the **sum, the comparison, and the box are exact** — no annotation noise. (c) It sits **at
the same level as the other benchmarks** (catalogued in the custom family F1, under
[`data/probes/`](../../data/probes/README.md)) and runs through the *same* pipeline and metrics, so
it is directly comparable.

## Prompt plan — what goes in the prompt

Every axis uses a prompt that both *elicits* the capability and *constrains* the output so it
is scorable; full templates in [`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).
Two design rules learned from real runs:

- **Constrain verbosity.** Small VLMs answer in sentences ("Pinterest has a heavy female
  audience.") which sink ANLS/exact even when right. Short-answer prompts append *"Answer
  concisely with only the value."* — matching how VLMEvalKit runs these tasks.
- **Ask for a machine-parseable box** for grounding: *"Return the bounding box … as
  [x1,y1,x2,y2] in pixel coordinates. The image is W×H."*

## Location/spotting — the fair-comparison problem

This is the crux of comparing OCR engines against general VLMs. **Models do not have the same
interface for "where is X":**

| Model family                                     | Native spotting?                                                                 | How we get a box                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **PaddleOCR-VL**                                 | ✅ built-in **spotting/layout task** (prompt already returns elements + polygons) | take the matching element's polygon → axis-aligned bbox |
| **Florence-2**                                   | ✅ **task tokens** `<OD>`, `<OCR_WITH_REGION>`, `<CAPTION_TO_PHRASE_GROUNDING>`   | take the box of the matching label (already pixels)     |
| **GOT-OCR2.0**                                   | ✅ **region/fine-grained OCR** (read-with-coordinates)                            | convert its coords → [x1,y1,x2,y2]                      |
| **SmolVLM / InternVL / LLaVA-OV / Ovis / H2OVL** | ❌ general chat VLM, no spotting head                                             | **text-prompted box** via the normalised instruction    |

**The fairness fix (normalisation layer).** If we only used PaddleOCR's spotting prompt, the
chat VLMs would score 0 by *interface mismatch*, not by inability — and if we only asked for a
text box, we'd ignore the spotting models' real strength. So:

1. Give **every** model the *same* normalised instruction ("return [x1,y1,x2,y2] in pixels").
2. For **spotting-capable** models, additionally run their native task and **map the output
   back into the same [x1,y1,x2,y2] pixel format** (polygon→bbox, task-token→box, loc-token→box).
3. Score both with the **same IoU metric**. The permissive parser in `metrics.grounding`
   already absorbs format differences (0–1, 0–1000, or pixel coords; `loc_<n>` tokens).

This makes "can it localise?" an apples-to-apples number. The expected result — a real
*capability* finding, not a harness artefact — is that **general small VLMs produce weak or no
boxes** while **spotting-trained models (PaddleOCR-VL, Florence-2, GOT) localise well**, which
is precisely a reason to prefer a spotting-capable model when field localisation matters.

## Results on the capability probe (small models run here)

Run on CPU via `python scripts/run_matrix.py --models … --benchmark
data/probes/capability_probe/capability.jsonl` (the two smallest models actually ran here;
the 1B+ models need a GPU — see `scripts/run_all.sh`). Scores are the per-axis task metric:

| model        | text-recog | kie-localized | integrative-sum | integrative-rel | chart | grounding |
| ------------ | :--------: | :-----------: | :-------------: | :-------------: | :---: | :-------: |
| SmolVLM-256M | 0.93       | 0.94          | **0.00**        | **0.00**        | 1.00  | **0.00**  |
| SmolVLM-500M | 0.93       | 0.94          | **1.00**        | **0.00**        | 1.00  | **0.00**  |

**Interpretation (a real capability vector, not a single score):**

- **Text recognition + localized KIE are solved** even at 256M (0.93–0.94): reading a field
  is the easy axis.
- **Integrative reasoning splits by size.** 256M *attempts* the sum but miscalculates
  (`$45+$80+$20.50 = $125.00`, truth 145.50 → 0.00); **500M gets it right (145.50 → 1.00)**.
  So simple aggregation emerges between 256M→500M. **Relational** reasoning ("which item is
  most expensive?") *fails for both* (both answer "Widget A"; truth "Gadget B") — cross-region
  comparison is beyond this size.
- **Chart reading works** (bar B = 70, both 1.00) on a clean synthetic chart.
- **Grounding fails for both** (256M echoes the image size; 500M emits a nonsense box
  `[145.50,820,600,820]`). General small VLMs have **no usable location head** — confirming the
  *Location/spotting* section: localisation needs a spotting-capable model, and is the axis where PaddleOCR-VL /
  Florence-2 / GOT are expected to dominate.

Net: at ≤0.5B you get a *reader* (text + chart), not a *reasoner* or a *localizer*. That
directly shapes selection (see *How this informs model selection*).

## Spatial & context understanding (independent axis)

The capability probe above tests *reading and computing values*. This axis tests the orthogonal
question — **does the model understand *space* and *context*?** — and it is easy to fake: a model
can score well by exploiting **shortcuts** (language priors like "totals are at the bottom",
position bias, binary guessing, rubber-stamping "yes it's consistent", or hallucinating a plausible
value). So every hypothesis is paired with a **control** that a shortcut would *fail*, and a
**signal criterion** only true understanding satisfies. Probe:
[`data/probes/spatial_context_probe`](../../data/probes/spatial_context_probe) (rendered, exact GT);
generator `scripts/make_spatial_context_probe.py`; signals computed by
`scripts/analyze_probe_signals.py`.

> **Scope boundary (kept strict).** This axis is *only* position + context — where regions are, how
> they relate, consistency/verification, absence/honesty, disambiguation, cross-reference. *Reading
> or computing a value* (recognition, KIE, arithmetic sum, value comparison, chart-value) is the
> capability probe above. The two probes share **no items**: *computing* the total is content
> reasoning; *verifying* the items match the printed total (C1) is a context test here.

### Methodology: signal, not score
For each capability we report a **signal** = (passes the test) **AND** (passes the control). Plain
accuracy is reported too, but the *criterion for "understands"* is the shortcut-robust signal —
designed so the **null hypothesis = "uses a shortcut"** is falsifiable.

### Spatial understanding

| #      | Hypothesis                                | Test                                                                                                 | Shortcut it must beat                      | Signal / criterion                                                                                                                    |
| ------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **S1** | Knows absolute region                     | "Which quadrant has ZEBRA?" ×4 (one per quadrant)                                                    | position bias (always answer one quadrant) | accuracy **> 25%** chance **and** not all-same-answer; criterion ≥ 3/4                                                                |
| **S2** | Knows relative position from *perception* | "Is TOTAL above or below the items?" on **normal** (total bottom) **and counterfactual** (total top) | language prior "total = bottom"            | **must be correct on BOTH**. Define **prior-reliance gap = acc(normal) − acc(counterfactual)**; understanding ⇒ gap ≈ 0 with both = 1 |
| **S3** | Box tracks the element                    | return bbox of ANCHOR at top / mid / bottom                                                          | constant-box prior (emit same box always)  | **center-tracking**: predicted box center-y must increase with true position (correlation **r > 0.8**) **and** mean IoU > 0.3         |

S2's counterfactual is the key idea — a model that answers "below" on the *top* invoice is
pattern-matching, not seeing. S3 turns grounding into a *relational* test: even if absolute IoU is
low, a box that *moves with* the element shows real spatial tracking; a constant prior box scores r≈0.

### Context understanding

| #      | Hypothesis                                   | Test                                                                                                  | Shortcut it must beat                       | Signal / criterion                                                                                     |
| ------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **C1** | Cross-region numeric consistency             | "Do the items add up to the TOTAL?" on **consistent** (yes) **and inconsistent** (no)                 | rubber-stamp "yes"                          | **must catch the inconsistent case** (n→no); criterion: both correct, esp. control=no                  |
| **C2** | Absence / anti-hallucination                 | "What is the discount?" when **no discount field exists**                                             | invent a number                             | answers **"none"**; any numeric answer = hallucination (fail)                                          |
| **C3** | Field disambiguation by context              | "What is the TOTAL (not subtotal)?" with a look-alike Subtotal present                                | grab the nearest/first number               | returns the **Total**, not the Subtotal                                                                |
| **C4** | Cross-reference + counterfactual sensitivity | "How much does the 'Bill to' person owe?" with header **Bob** vs **Alice** (table maps names→amounts) | answer invariant to header (ignore context) | **answer must flip** Bob→45, Alice→80; criterion: both correct (= sensitivity to the relevant context) |

C1's inconsistent variant defeats a model that always says "consistent"; C2 measures *honesty under
absence*; C4's two-variant design measures **counterfactual sensitivity** — if changing the only
relevant token (the name) doesn't change the answer, the model isn't using context, it's guessing.

### Signal-criteria summary (what counts as "has the capability")

| Capability               | PASS criterion (shortcut-robust)                                     |
| ------------------------ | -------------------------------------------------------------------- |
| Absolute spatial (S1)    | ≥ 3/4 quadrants correct, answers not constant                        |
| Relative spatial (S2)    | correct on **both** normal & counterfactual (prior-reliance gap ≈ 0) |
| Spatial tracking (S3)    | center-y correlation r > 0.8 **and** mean IoU > 0.3                  |
| Consistency (C1)         | catches the inconsistent case (control=no correct)                   |
| Anti-hallucination (C2)  | "none" on the absent field                                           |
| Disambiguation (C3)      | returns Total, not Subtotal                                          |
| Context sensitivity (C4) | answer flips correctly with the header (both variants)               |

A model "understands space/context" only if it clears the **control**, not just the easy case.

### Results (SmolVLM, CPU) — interpretation
Signal table (✅ = clears the shortcut control), from `scripts/analyze_probe_signals.py`:

| model        | S1 absolute       | S2 relative | S3 tracking | C1 consistency | C2 anti-halluc | C3 disambig | C4 context-sens |
| ------------ | :---------------: | :---------: | :---------: | :------------: | :------------: | :---------: | :-------------: |
| SmolVLM-256M | ❌ (acc .25, bias) | ❌           | ❌           | ❌              | ❌              | ✅           | ❌               |
| SmolVLM-500M | ✅ (acc .75)       | ❌           | ❌           | ❌              | ❌              | ✅           | ❌               |

What the *controls* exposed (the whole point):
- **S2 — constant-answer trap.** Both answer *"above"* for **every** layout → correct on the
  counterfactual, wrong on the normal one (`prior_reliance_gap = −1.0`). A counterfactual-only test
  would have *falsely* certified relative-position understanding. **This justifies the whole
  control-paired methodology.**
- **S1 — absolute spatial emerges with size.** 256M answers *"top-left"* for all quadrants (bias,
  acc .25); **500M reaches acc .75 → PASS.**
- **C1 — rubber-stamping caught.** Both say *"Yes, consistent"* even when the total is wrong → the
  inconsistent control flips them to FAIL. No real cross-region arithmetic verification.
- **C2 — hallucination caught.** Asked for a non-existent discount, 256M invents *"$20.50"*, 500M
  *"0"* — neither abstains → both FAIL.
- **C4 — no context sensitivity.** Each anchors to one table row regardless of the *"Bill to"* name.
- **C3 — the one real pass.** Both return the **Total** (not the look-alike Subtotal).

**Bottom line:** at ≤0.5B, document VLMs are **literal readers with local disambiguation** but lack
**relative-position, grounding, cross-region verification, honesty under absence, and context
sensitivity** — gaps that are invisible to single-example accuracy and surface only under the paired
controls. Run any model with `python scripts/run_matrix.py --models <m> --benchmark
data/probes/spatial_context_probe/probe.jsonl`; `scripts/analyze_probe_signals.py` then prints the
PASS/FAIL per criterion + the prior-reliance gap / tracking correlation.

## How this informs model selection

- If the use case is **read a known field** (KIE-localized) → text-understanding axis dominates;
  even a 256–500M VLM can be viable.
- If it needs **totals / cross-field reasoning** (integrative) → require a model that holds up
  on the integrative axis; this is where 1B models (InternVL) separate from 256M.
- If it needs **charts** → require the chart-dependent axis (ChartQA-strong models).
- If it needs **localisation / structured parsing** → require the *location* axis → prefer a
  **spotting-capable** model (PaddleOCR-VL / Florence-2 / GOT), since general VLMs lack it.

The selection is therefore a **capability vector per model**, not a single score — which is
what the probe + the cross-benchmark matrix produce.

## Output-side requirements per document type (spotting & abstain)

The [document-type taxonomy](document_type_taxonomy.md) deliberately lists only *input* stressors;
the two **output-side** requirements live here, with the evaluation axes, because they are
properties of the model's *answer*, not of the document:

- **Spotting / grounding** (see *Two top-level abilities* and *Location/spotting*) — *where* each value came from. Scored by IoU
  ([`metrics/grounding.py`](../../src/docvlm_eval/metrics/grounding.py)); the fairness/normalisation
  layer in *Location/spotting* makes it apples-to-apples across OCR engines and chat VLMs.
- **Hallucination control (correct-or-abstain)** — the reliability counterpart to spotting. When a
  field is **absent / redacted / illegible**, the correct answer is "not present" / "[redacted]";
  confidently inventing a value is the worst failure. Scored by **abstain accuracy** on planted
  absent/redacted fields (the `abstain` probes in
  [`data/probes/realistic_cases`](../../data/probes/realistic_cases/README.md)) plus
  **ECE calibration** ([`metrics/calibration.py`](../../src/docvlm_eval/metrics/calibration.py)) —
  does the model's confidence track whether it is actually right?

These two are *why* a type's **anchor metric** in the taxonomy reads "… + spotting IoU" or
"… + abstain". The types where each is a **primary** requirement (✓✓ = critical):

| Document type               | Spotting (IoU) | Abstain (correct-or-abstain) |
| --------------------------- | :------------: | :--------------------------: |
| Invoice / receipt           | ✓              | ✓                            |
| Bank stmt / payslip         | ✓              | ✓                            |
| Contract / NDA              | ◐              | ✓                            |
| **ID / passport / license** | ✓              | ✓✓                           |
| Certificate / diploma       | ◐              | ✓                            |
| **Cheque**                  | ✓              | ✓                            |
| **Prescription**            | ✓              | ✓ (illegible → abstain)      |
| Ticket / boarding pass      | ✓              | —                            |
| Map / floor plan            | ✓              | —                            |
| Ledger / census             | ✓              | —                            |
| LCD / meter / 7-seg         | ✓              | —                            |
| UI / chat / code screenshot | ✓              | —                            |
| **Form w/ checkboxes**      | ✓              | ✓                            |
| **Redacted document**       | ✓              | ✓✓                           |
| Chemical / circuit diagram  | ✓              | —                            |

The **identity/administration** family (ID, cheque, prescription, redaction) is where abstain
matters most — a confidently-wrong passport number or cheque amount is worse than "unreadable" —
so its anchor metrics pair spotting IoU with abstain. Secondary (◐) needs for other types are
captured directly in their anchor-metric column in the taxonomy.
