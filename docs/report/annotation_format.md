# Layered document annotation (DAR) — one record, every layer, and a test of which layer matters

The public corpus (UDD, [`unified_loader.md`](unified_loader.md)) and the synthetic generator
([`synthetic_data_dto.md`](synthetic_data_dto.md)) give us *breadth*. This document defines the
third data track: a **small, hand-annotated set of hard real documents** where every image carries
**every annotation layer** — orientation, layout, OCR, tables, KIE, and an *understanding* layer
(caption, reading guide, reasoned QA) — in one universal record, the **Document Annotation Record
(DAR)**. It also defines the experiment that decides **which of those layers actually helps a model
extract the right value**.

Code: `src/docvlm_eval/annotation/` · CLI: `scripts/annotate_dar.py` · worked example:
[`data/annotations/example/`](../../data/annotations/example/README.md) · tests:
`tests/test_annotation.py`.

## 1. Why one record with every layer

**The end task is the value, not the transcript.** Document understanding ends in *extracting the
right value* — the invoice total, the due date, "is the subtotal consistent?". OCR, layout and
orientation are only *means* to that end. Each of them is a **hypothesis** about what a model needs
in order to get the value right:

| Layer | Hypothesis it encodes |
| --- | --- |
| orientation | a model that does not know the page is rotated reads garbage |
| layout | knowing *where the blocks are* (and their order) narrows where a value can be |
| OCR (character level) | values fail because characters are mis-read (CER on the evidence span) |
| line grounding | binding text ↔ position is what lets the model look in the right place |
| table structure | row/column membership is what defines a cell's meaning |
| KIE spotting | answering *and* pointing forces the value to be grounded, not guessed |
| reasoning chains | multi-step values (dates + terms, sums, comparisons) need the steps |
| reading knowledge | some documents can only be read with conventions that no single string shows |

**Why not separate datasets per task?** An individual annotator produces hundreds of images, not
hundreds of thousands. With separate per-task sets, every layer would sit on *different* images, so
the layer and the image distribution would be confounded and no comparison would be fair. With one
record per image:

1. **Maximum supervision per image.** The expensive part, choosing and understanding a hard
   document, is paid once and reused by every layer.
2. **A clean ablation.** Every arm trains on the *same* images. Only the set of layers emitted as
   targets changes, so a held-out Δ is attributable to the layer (§6).
3. **Mechanical consistency.** Higher layers *cite the ids* of lower layers instead of copying text
   or boxes, so a KIE value that is not on the page, or a QA citing a deleted line, is caught by the
   validator (§4.3).
4. **Error attribution.** A wrong value can be traced to the lowest layer that failed (§7).

## 2. The layer stack

```
page          orientation · skew · doc_type · acquisition · languages · quality   "can it be read as-is?"
  layout      regions r*: class · box · reading order · parent                    "where are the parts?"
    ocr       lines t*: verbatim text · box · owning region                       "what does it say?"
      table   tb*: grid of cells (row, col, span) → line ids                      "how is it structured?"
      kie     f*: key → value (as printed) + normalized + evidence line ids       "what is the value?"
        understanding
              caption               "what is this document, what is it for?"
              guide  g*: notes scoped to ids  (structure / convention / disambiguation / domain / pitfall)
              qa     q*: question → answers + steps, each step citing evidence ids  "what does it mean?"
provenance    per layer: status (absent / pseudo / human / verified) · annotator · seconds spent
```

Ids are unique across the record. Table cells are addressed as `<table_id>:<row>,<col>` (e.g.
`tb1:2,3`). An evidence list may mix lines, cells, fields and regions.

## 3. The record

Abbreviated from the worked example
([`invoice_0001.json`](../../data/annotations/example/invoice_0001.json)):

```json
{
 "schema_version": "dar-1.0", "doc_id": "example/invoice_0001", "image": "invoice_0001.png",
 "split": "train",
 "page": {"width": 800, "height": 1000, "orientation": 0, "skew_deg": 0.0, "doc_type": "invoice",
          "domain": "finance", "acquisition": "digital", "languages": ["en"], "quality": []},
 "layout": [{"id": "r3", "cls": "kv_block", "bbox": [465,115,750,205], "order": 2, "label": "invoice metadata"}, "…"],
 "ocr":    [{"id": "t5", "text": "Date: 03/04/2024", "bbox": [470,154,629,169], "region": "r3",
             "handwritten": false, "legibility": "clear"}, "…"],
 "tables": [{"id": "tb1", "n_rows": 4, "n_cols": 4, "region": "r4",
             "cells": [{"row": 2, "col": 3, "text": "480.00", "lines": ["t21"]}, "…"]}],
 "kie": [
   {"id": "f2", "key": "invoice.date", "value": "03/04/2024", "normalized": "2024-04-03",
    "value_type": "date", "value_lines": ["t5"], "key_lines": ["t5"], "present": true},
   {"id": "f9", "key": "invoice.po_number", "present": false, "notes": "no PO number on this invoice"}],
 "understanding": {
   "caption": "A one-page supplier invoice … stamped PAID.",
   "guide": [{"id": "g2", "kind": "convention", "scope": ["t5"],
              "text": "The date is written DD/MM/YYYY (Korean vendor), so 03/04/2024 is 3 April 2024, not 4 March."}],
   "qa": [{"id": "q2", "question": "On what date is payment due? Answer as YYYY-MM-DD.",
           "answers": ["2024-05-03"], "qa_type": "temporal", "metric": "exact", "difficulty": 3,
           "evidence": ["t5", "t6"],
           "steps": [{"op": "read",    "text": "The invoice date is printed as 03/04/2024.", "evidence": ["t5"], "result": "03/04/2024"},
                     {"op": "parse",   "text": "Dates on this invoice are DD/MM/YYYY, so it is 3 April 2024.", "result": "2024-04-03"},
                     {"op": "read",    "text": "The terms are 'Net 30'.", "evidence": ["t6"], "result": "30 days"},
                     {"op": "compute", "text": "3 April 2024 + 30 days = 3 May 2024.", "result": "2024-05-03"}]}]},
 "provenance": {"layers": {"ocr": {"status": "human", "annotator": "me", "seconds": 240}, "…"}}
}
```

### 3.1 Conventions that make the record unambiguous

- **Boxes** are `[x1, y1, x2, y2]` in **pixels of the stored image**. This is the frame the
  grounding metric scores in (`metrics/grounding.py`). Never draw boxes on a rotated or deskewed
  view of the image.
- **`page.orientation`** is how many degrees *clockwise* the upright page was rotated to produce
  the stored image (0/90/180/270). Small tilts go in `skew_deg`. `geometry.to_upright()` and
  `rotate_annotation()` move the image and every box between frames exactly.
- **`ocr[].text` is verbatim.** It keeps the original case, punctuation, line breaks between lines
  and typos. Write unreadable characters as `#` and set `legibility: partial|illegible`.
- **`kie[].value` is as printed; `normalized` is what the end task is scored on.** Amounts become
  plain decimals (`1298.00`), dates become ISO (`2024-04-03`), ids keep their hyphens. The gap
  between the two is itself knowledge: `03/04/2024 → 2024-04-03` is only correct under the
  DD/MM convention.
- **`present: false`** records a schema field that the document does **not** contain. These
  explicit abstain golds are how a model learns not to hallucinate a PO number. The KIE metric
  penalises a predicted value for an absent field (`kie_f1`).
- **`answers`** are gold *variants of one answer* (`["59.00", "59"]`), never answers to different
  questions. This is the same rule as UDD.
- **The split is per image** (`train` / `heldout`). It is never per QA, because a sibling QA would
  leak the held-out image.

## 4. Annotation guide (how to fill each layer)

The guiding rule: **annotate what a careful new employee would need to be told**. Characters and
boxes are cheap and partly automatable. The *interpretation* is what only you can provide.

### 4.1 Per layer

| Layer | Do | Don't | Typical cost / page |
| --- | --- | --- | --- |
| page | record the *true* capture orientation, the doc type (use the [`document_type_taxonomy.md`](document_type_taxonomy.md) names), acquisition and quality flags | label orientation by eye on a pre-rotated preview | ~15 s |
| layout | one region per semantic block from the closed class list (`schema.REGION_CLASSES`); give `order` in human reading order; nest with `parent` | split a block into lines (that is the OCR layer) or invent classes | 1–2 min |
| ocr | one entry per visual line, with a tight box and the owning `region`; mark `handwritten` | normalise text, merge two columns into one line | 3–6 min (pre-label first, §5) |
| table | a cell for every grid position, with spans for merged cells; `header: true` for header cells; link the cell's `lines` | encode the table as free text | 1–3 min |
| kie | one field per schema key: printed `value`, `normalized`, `value_type`, `value_lines` (+ `key_lines` for the printed label); add `present: false` for expected-but-missing keys | put a derived value in KIE (e.g. the due date): that is a QA | 2–3 min |
| caption | 1–3 sentences on what the document is and what it is for | describe the layout box by box | 30 s |
| guide | a note for each convention, trap or structural rule needed to read *this* document, `scope`d to the ids it concerns | restate what a string literally says | 2–4 min |
| qa | 3–8 questions per hard document, mixing types; every non-trivial one gets `steps`, each step citing `evidence` and producing a `result` | ask questions whose answer is a single verbatim string you already have in KIE (redundant) | 4–8 min |

### 4.2 The understanding layer: what makes a good note or chain

**Guide notes** fall into five kinds. Each one has to be *falsifiable from the page*:

| kind | example | what it teaches |
| --- | --- | --- |
| `structure` | "Amount = Qty × Unit Price; the Subtotal is the sum of the Amount column." | relations between regions/cells |
| `convention` | "Dates are DD/MM/YYYY, so 03/04/2024 is 3 April." | locale/format rules that change the value |
| `disambiguation` | "TOTAL DUE already includes VAT; do not add tax again." | which of several candidate values is meant |
| `domain` | "'Net 30' means payment is due 30 days after the invoice date." | outside knowledge the document assumes |
| `pitfall` | "The PAID stamp is a status mark, not part of any field." | things that look relevant but are not |

**Reasoning chains** follow fixed rules so they stay trainable and checkable:

- Each step has one `op` from `locate | read | parse | compare | compute | aggregate | lookup |
  infer | conclude`.
- `read`/`locate` steps **must cite evidence**. At export time, their boxes are written into the
  chain inline (`[470,154,629,169]`), so the chain is *grounded*, not free-form prose.
- `result` holds the intermediate value. The last step's result must be one of the `answers`
  (the validator warns otherwise).
- Rate `difficulty` 1 (single lookup) / 2 (one operation: compare, sum, parse) / 3 (multi-hop and
  arithmetic or convention).
- Include at least one **abstain** question per few documents (`qa_type: abstain`, answer
  `"not stated"`).

### 4.3 What the validator checks

`python scripts/annotate_dar.py validate <dir>` (`annotation/validate.py`):

- **errors** (export refuses): duplicate or unknown ids; a box outside the image or degenerate
  (the usual cause is a box drawn on a rotated view); reading-order indices that are not unique;
  overlapping table cells, or cells outside the grid; `present: false` with a value; a QA with no
  answer; a layer with content but provenance `absent`.
- **warnings** (look at them): a KIE value not found in its evidence lines (a typo, or a
  normalisation that belongs in `normalized`); a line centre outside its region; cell text
  different from its lines; a QA without evidence; a last step result that is not an answer.

## 5. The annotation pipeline

```
images ──► init ──► pre-label (pseudo) ──► human correct ──► validate ──► stats ──► export per arm
           skeleton   OCR / layout model     status: human     errors = 0    hours,      train.jsonl
           per image  status: pseudo         + seconds spent                 coverage    heldout.jsonl
```

```bash
python scripts/annotate_dar.py init data/annotations/mine/*.png --prefix mine/ --doc-type invoice --acquisition photo
# pre-label the ocr/layout layers with any model (status "pseudo", tool = model name), then correct by hand
python scripts/annotate_dar.py validate data/annotations/mine
python scripts/annotate_dar.py stats    data/annotations/mine      # layer coverage + annotation hours
python scripts/annotate_dar.py arms                                  # list the ablation arms
python scripts/annotate_dar.py export   data/annotations/mine --arm L1_+ocr --out data/dar_arms/L1_ocr
```

- **Pre-label the cheap layers, hand-write the expensive ones.** OCR lines and layout boxes are
  where models are already good. Run one (the repo wraps `got-ocr2` / `paddleocr-vl`, see
  `unified/pseudo_label.py`), store the output with `status: pseudo`, and correct it. KIE
  normalisation, guide notes and reasoning chains are where your time buys the most.
- **Log the seconds per layer** in provenance. That is the denominator of "Δ per annotation-hour"
  (§6.3), the number that decides what to annotate next.
- **`--human-only`** drops pseudo layers at export, so "does reviewing the pseudo-labels matter?"
  is a free extra ablation.
- **Bridge to UDD:** `annotation.to_unified(ann)` yields a `UnifiedSample` (fields with evidence
  boxes, regions, full text, table HTML, grouped QAs). Hand-annotated documents can therefore join
  the public corpus. Reasoning steps and guide notes stay in the DAR file, because UDD has no
  column for them.

### Exported tasks (one per export layer)

| export layer | sample | metric |
| --- | --- | --- |
| `orientation` | "how many degrees clockwise is this rotated?" (+ exact 90/180/270 copies, gold for free) | `exact` |
| `doctype` | "what type of document is this?" | `anls` |
| `layout` | "where is the `<class>`?" (all boxes of that class are golds) + reading-order list | `grounding`, `ned` |
| `ocr` | full transcript in reading order | `ned` |
| `line_grounding` | "where is the text "…"?" and "read the text inside box …" (unique, clear lines) | `grounding`, `ned` |
| `table` | table → HTML | `teds` |
| `kie` | all fields as JSON (absent → `null`) + one question per field | `kie_f1`, `anls` |
| `kie_spotting` | "value and where?" → `value \| x1,y1,x2,y2` | `anls` |
| `qa` | `answer`: bare answer · `chain`: grounded steps + `Answer:` · `guided_chain`: the same, opened by the in-scope guide notes | QA's own metric / `final_answer` |
| `caption`, `guide` | "describe this document" / "explain how to read it" | `ned` |

`kie_f1` (field-level F1 that penalises hallucinated absent fields) and `final_answer` (scores the
text after the last `Answer:`, with numeric tolerance only for purely numeric golds) live in
`metrics/structured.py` and are registered in the standard scorer.

## 6. Does the knowledge help? The layer-value ablation

### 6.1 Design

Every arm trains on **the same annotated training images** and always includes the **end task**
(KIE + QA). Arms differ only in which other layers are also emitted as targets
(`annotation/ablation.py`, `ARMS`):

| family | export layers it adds | QA style |
| --- | --- | --- |
| `orientation` | orientation (+ rotated copies), doctype | — |
| `layout` | layout | — |
| `ocr` | full transcript | — |
| `line_grounding` | read-box / where-is-text | — |
| `table` | table HTML | — |
| `spotting` | KIE value + box | — |
| `reasoning` | — | `chain` |
| `knowledge` | caption, guide | `guided_chain` *if* `reasoning` is also on |

- **Additive arms** `L1_+<family>`: target + one family. Does this layer help on its own?
- **Interaction arm** `L1_+reasoning+knowledge`, read against `L1_+reasoning`: the same chains,
  with and without the reading guide. This is the direct test of whether *written interpretation
  knowledge* helps beyond reasoning steps alone.
- **Leave-one-out arms** `L2_all-<family>`: everything except one family. Is the layer still
  needed once the rest is present?

**Controls.** Same images, same held-out set (`split: heldout`, end task only, prompted in the
arm's QA style), same seed, and **fixed training steps** (`run_ablation.py --steps`). Extra layers
mean extra samples per image, so a fixed step budget makes auxiliary targets *compete* with the end
task for updates instead of getting free compute. The held-out set is scored by `kie_f1`, per-field
`anls`, and the QA metric / `final_answer`, which read the final value whether or not the model
wrote a chain. All arms are therefore compared on the same gold values.

```bash
for arm in L0_target L1_+ocr L1_+spotting L1_+reasoning L1_+reasoning+knowledge L2_all; do
  python scripts/annotate_dar.py export data/annotations/mine --arm $arm --out data/dar_arms/$arm
done
# then train + eval each arm with the public-data path (same --steps for every arm), e.g.
python scripts/run_ablation.py --arm public --steps 300 \
  --train-jsonl   data/dar_arms/L1_+ocr/train.jsonl \
  --heldout-jsonl data/dar_arms/L1_+ocr/heldout.jsonl --record-key dar:L1_+ocr
```

### 6.2 Hypotheses and how to read them

| id | hypothesis | supported if | falsified if |
| --- | --- | --- | --- |
| **H1** | orientation is cheap and load-bearing for photos | `L1_+orientation` > `L0` on rotated/photo held-out docs; `L2_all-orientation` drops on them | no Δ even on the rotated slice |
| **H2** | character-level OCR supervision transfers to value extraction | `L1_+ocr` > `L0`; attribution shows fewer `ocr` failures | KIE unchanged while transcript CER improves (reading ≠ extracting) |
| **H3** | grounding beats transcription per hour | `L1_+line_grounding` or `L1_+spotting` ≥ `L1_+ocr` at lower cost/hour | transcription wins and grounding adds nothing in LOO |
| **H4** | table structure matters for cell-relational QA | `L1_+table` > `L0` on `comparison`/`aggregation` QAs only | no Δ on table QAs |
| **H5** | reasoning chains help multi-step values | `L1_+reasoning` > `L0` on difficulty ≥ 2; no loss on difficulty 1 | chains hurt lookups or don't move hard QAs |
| **H6** | **interpretation knowledge helps beyond reasoning** | `L1_+reasoning+knowledge` > `L1_+reasoning`, concentrated on QAs whose evidence is in a guide note's scope | no Δ, or Δ only on train (memorised notes) |

Slice every arm's held-out score by `qa_type`, `difficulty`, `doc_type`, `orientation` and
`acquisition` (all in `Sample.meta`). Most layers are expected to help *one slice* rather than the
average.

### 6.3 Value per annotation-hour

`ablation.value_per_hour(scores, anns)` divides each additive arm's Δ over `L0_target` by the
human hours logged for that layer. This is the decision number for an individual annotator. A layer
with a small Δ that takes 10 s per page can beat a layer with a big Δ that takes 8 minutes. Re-run
it as the corpus grows, because the cheapest-useful layer can change with scale.

### 6.4 Statistical honesty at small N

With a few hundred images the held-out set is small, so **report bootstrap CIs over held-out
images** (resample images, not QAs) and do not claim a Δ inside the interval. Run the cheap,
high-prior arms first (`L0`, `+ocr`, `+spotting`, `+reasoning`, `+reasoning+knowledge`, `L2_all`).
Run leave-one-out only for families that won additively. If an arm's *train* score climbs while
held-out stays flat, it is memorising the notes. This is the same A0 read-out as
[`ablation_plan.md`](ablation_plan.md) §4.

## 7. Error attribution: which layer broke?

Every KIE field and QA cites its evidence lines, so a wrong value can be traced down the stack
(`annotation/diagnose.py`):

```
orientation probe wrong         → orientation
evidence lines mis-read (CER)   → ocr
evidence not localised (IoU)    → localization
all probes pass                 → reasoning
```

`diagnostic_samples(ann)` emits the end-task questions plus the probes (orientation, read-this-box
and where-is-this-text for every evidence line). Run the model on them, then pass
`{sample_id: prediction}` to `attribute()` and `failure_profile()`. The result is the **failure
profile** of a model: the share of value errors per broken layer. It complements the ablation. The
ablation measures what *training* on a layer buys. The failure profile shows where the *current*
model actually fails, which is where to annotate next. A value counts as correct only when fully
correct (`correct_min=1.0`): `1289.00` for `1298.00` is a wrong total, even at ANLS 0.71.

## 8. How this relates to the other tracks

| | synthetic (`DocSample`) | public (UDD) | **hand-annotated (DAR)** |
| --- | --- | --- | --- |
| scale | unbounded | ~28k images | hundreds |
| GT exactness | exact by construction | as published | human, validated |
| layers per image | fields, QA, rationale, boxes | whatever the source had (sparse) | **all, linked by id** |
| reasoning | templated rationales | derived spatial rationales | **hand-written grounded chains + reading guide** |
| role | controlled factor sweeps (A1–A7) | external validity | **hard real cases + which-layer-matters** |

The DAR families map onto the existing ablations: `spotting` ≈ A1, `reasoning` ≈ A2,
`spotting + reasoning` ≈ A3, `orientation` covers the orientation TODO in [`../plan.md`](../plan.md)
§5. What is new is the `knowledge` family and the per-hour value read-out.
