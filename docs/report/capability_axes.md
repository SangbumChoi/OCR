# Capability axes for document VLMs — selection, custom probe, and prompt plan

Benchmark *names* hide what a model can actually do. To select and compare sub-1B document
VLMs meaningfully we evaluate along **capability axes**, build a small **custom probe** that
isolates each axis on controlled images, and define **what to put in the prompt** to elicit
each axis from every model — including the tricky **location/spotting** axis where models
differ a lot. This document is the rationale; the probe is
[`data/benchmarks/capability_probe`](../../data/benchmarks/capability_probe), the prompts are
[`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).

## 1. Two top-level abilities

A document model is really two abilities stacked:

| Ability                            | Question it answers   | How we elicit it               | How we score it        |
| ---------------------------------- | --------------------- | ------------------------------ | ---------------------- |
| **Text understanding** (글자 이해)     | "What does it *say*?" | read / answer prompts          | ANLS / exact / relaxed |
| **Location understanding** (위치 이해) | "*Where* is it?"      | "return the bounding box of X" | **IoU** (grounding)    |

These are genuinely different heads: a model can read a field perfectly yet be unable to point
at it (and vice-versa for a detection-first model). Most leaderboards score only the first;
production document parsing (reading order, field localisation, table cells, redaction) needs
the second. So we score **both**, with a dedicated grounding metric
([`metrics/grounding.py`](../../src/docvlm_eval/metrics/grounding.py), IoU with a permissive box
parser).

## 2. When the output is text, *what kind* of text?

Not all text answers are equal. Three natures, in increasing difficulty:

| Nature                    | What it demands                                                        | Example (probe sample)                                                                  | Metric              |
| ------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------- |
| **KIE-localized**         | read **one region**, give a clear field value                          | "What is the vendor name?" → `Acme Corporation`                                         | anls                |
| **Integrative reasoning** | **combine several regions** into a value — a *sum* or a *relationship* | "Add up the line-item prices." → `145.50`; "Which item is most expensive?" → `Gadget B` | relaxed_acc / exact |
| **Chart-dependent**       | the answer is **not in the text** — must read a figure/chart           | "What is the value of bar B?" → `70`                                                    | relaxed_acc         |

The jump from *localized* to *integrative* is where small LMs fail: it needs multi-region
attention + arithmetic/comparison, not just OCR. The *chart-dependent* nature additionally
needs graphic perception, not text at all. Separating these tells us *why* a model is weak,
not just *that* it is.

## 3. The custom capability probe (rationale)

[`data/benchmarks/capability_probe`](../../data/benchmarks/capability_probe) is rendered by
[`scripts/make_capability_probe.py`](../../scripts/make_capability_probe.py) so the ground truth
— **including exact pixel boxes** — is known. Six samples, one per axis:

| Sample          | Axis               | Prompt intent              | Gold                   |
| --------------- | ------------------ | -------------------------- | ---------------------- |
| `cap_text`      | text-recognition   | direct read                | `INV-2025-0042`        |
| `cap_kie`       | kie-localized      | single-field KIE           | `Acme Corporation`     |
| `cap_integ_sum` | integrative-sum    | multi-region aggregation   | `145.50`               |
| `cap_integ_rel` | integrative-rel    | cross-region comparison    | `Gadget B`             |
| `cap_chart`     | chart-dependent    | chart value read           | `70`                   |
| `cap_ground`    | location-grounding | text-prompted bounding box | box `(40,335,270,359)` |

**Why custom + synthetic?** (a) Public sets entangle axes (DocVQA mixes localized + integrative);
the probe isolates *one* axis per item so a failure is unambiguous. (b) We control the layout,
so the **sum, the relationship, and the box are exact** — no annotation noise. (c) It sits **at
the same level as the other benchmarks** (`data/benchmarks/…`, catalogued as category 11) and
runs through the *same* pipeline and metrics, so it is directly comparable.

## 4. Prompt plan — what goes in the prompt

Every axis uses a prompt that both *elicits* the capability and *constrains* the output so it
is scorable; full templates in [`configs/capability_prompts.yaml`](../../configs/capability_prompts.yaml).
Two design rules learned from real runs:

- **Constrain verbosity.** Small VLMs answer in sentences ("Pinterest has a heavy female
  audience.") which sink ANLS/exact even when right. Short-answer prompts append *"Answer
  concisely with only the value."* — matching how VLMEvalKit runs these tasks.
- **Ask for a machine-parseable box** for grounding: *"Return the bounding box … as
  [x1,y1,x2,y2] in pixel coordinates. The image is W×H."*

## 5. Location/spotting — the fair-comparison problem

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

## 6. Results on the probe (small models run here)

Run on CPU via `python scripts/run_matrix.py --models … --benchmark
data/benchmarks/capability_probe/capability.jsonl` (the two smallest models actually ran here;
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
  `[145.50,820,600,820]`). General small VLMs have **no usable location head** — confirming
  §5: localisation needs a spotting-capable model, and is the axis where PaddleOCR-VL /
  Florence-2 / GOT are expected to dominate.

Net: at ≤0.5B you get a *reader* (text + chart), not a *reasoner* or a *localizer*. That
directly shapes selection (§7).

## 7. How this informs model selection

- If the use case is **read a known field** (KIE-localized) → text-understanding axis dominates;
  even a 256–500M VLM can be viable.
- If it needs **totals / cross-field reasoning** (integrative) → require a model that holds up
  on the integrative axis; this is where 1B models (InternVL) separate from 256M.
- If it needs **charts** → require the chart-dependent axis (ChartQA-strong models).
- If it needs **localisation / structured parsing** → require the *location* axis → prefer a
  **spotting-capable** model (PaddleOCR-VL / Florence-2 / GOT), since general VLMs lack it.

The selection is therefore a **capability vector per model**, not a single score — which is
what the probe + the cross-benchmark matrix produce.
