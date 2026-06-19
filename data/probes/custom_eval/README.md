# Proposed custom evaluation set — design & rationale

A document-AI eval that mirrors **what production actually needs**. Unlike a single VQA accuracy
number, every sample here is tagged so results can be sliced along the axes that decide whether a
VLM is deployable, and **each content class is scored with the metric that fits it**. Built by
`scripts/make_custom_eval.py`; sliced by `scripts/analyze_custom_eval.py`.

## The format (why a richer schema)

Each line in `custom_eval.jsonl` is a normal `Sample` plus `meta`:

```json
{"sample_id":"ce_lang_ja","image_path":"images/lang_ja.png","question":"...","answers":["請求金額合計"],
 "answer_type":"text","metric":"ned",
 "meta":{"content_class":"text","language":"ja","rotation_deg":0,"reading_direction":"ltr",
         "spotting":null,"needs_reasoning":false}}
```

A single accuracy hides *where* a model fails. With these tags the same run yields breakdowns by
**content class / language / rotation / reading-direction / spotting**, which is what a buyer of an
OCR system actually asks about.

## 1. Per-content-class metrics (and why)

| Class               | Metric we use                                                          | Why this and not plain accuracy / CER                                                                                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **text**            | **NED (1−norm. edit dist)** + report **CER/WER**; **field‑F1** for KIE | CER alone over-penalises one-char slips on long strings and gives no credit structure; NED is bounded [0,1] and partial-credit-fair, and **field-level F1** (precision/recall) is what tells you *missed vs hallucinated* fields — the real KIE question. |
| **table**           | **TEDS / TEDS‑Struct** (+ cell‑content)                                | A table's value is its **structure + cell relationships**; a flat-text metric can't tell if rows/cols/spans are right. TEDS compares the HTML trees (structure *and* content); TEDS‑Struct isolates topology.                                             |
| **figure / layout** | **detection mAP / presence‑IoU**                                       | For "is there a figure and where", the question is *localisation*, so an IoU/mAP over predicted vs gold regions is the right signal — not a text score.                                                                                                   |
| **formula**         | **NED + exact** (token / LaTeX)                                        | Different LaTeX can render identically, so exact-match is too harsh; NED on the token string (and exact as a ceiling) balances it. (CDM is the gold-standard if a renderer is available.)                                                                 |
| **chart**           | **relaxed accuracy (±5%)**                                             | Reading a plotted value needs numeric tolerance, exactly ChartQA's metric.                                                                                                                                                                                |
| **QR / barcode**    | **exact decode match**                                                 | The payload is a discrete string; either it's decoded correctly or not — exact match on the decoded text.                                                                                                                                                 |
| **stamp / logo**    | **presence + IoU** (grounding)                                         | These are *objects to locate* (compliance: is the seal present and where) → detection-style IoU, plus a yes/no presence check.                                                                                                                            |

## 2. Rotation robustness (why + metric)

Real scans are often rotated — usually a few degrees (harmless), but sometimes **90°, 180°, 270°**
or arbitrary angles (phone photos, fed-upside-down). A deployable reader must cope. We render the
same page at **0/15/90/180/270°** and report:
* **retention** = read‑score(angle) / read‑score(0°) per angle (1.0 = rotation-invariant), and
* **orientation classification** ("by how many degrees is this rotated?") — because a good pipeline
  can *detect* and auto-correct skew.

A model that reads fine at 0° but collapses at 90/180° is a production risk this axis exposes.

## 3. Per-language performance (why + metric)

The same field in en/ko/ja/zh stresses the tokenizer and the visual encoder differently. We render
one text line per language and report **NED per language**, so a model that is strong in English but
weak in CJK is visible (a common failure for latin-centric small VLMs).

## 4. Reading direction (why + metric)

Most documents are horizontal left‑to‑right, but **Japanese and Chinese are often written vertically
(top‑to‑bottom)**, and some scripts are RTL. A reader that assumes LTR mis-orders vertical text. We
render vertical CJK + a horizontal control and score **direction classification accuracy**
(horizontal vs vertical) — a prerequisite for correct reading order.

## 5. Spotting + reasoning (why this matters most for trust)

Extraction results drive downstream actions, so even at 99% accuracy the remaining **1% must be
verifiable** — and a human verifier needs the **basis** of each extraction:
* **Spotting** — *where* on the page each value came from (bounding box). We ask for `[x1,y1,x2,y2]`
  and score **IoU**. Models without a native spotting task (general VLMs) are normalised to the same
  box format for fair comparison (see `docs/report/capability_axes.md` §5).
* **Reasoning** — *why* the model produced a value. Items with `needs_reasoning=true` should expose a
  short justification alongside the answer; when the reasoning itself contains additional fields,
  those are credited too. (Reasoning quality is partly qualitative; we score the answer with ANLS and
  treat the rationale as audit evidence.)

Together, spotting + reasoning turn a black-box extraction into an **auditable** one — the difference
between a demo and a system you can put in front of a compliance team.

## Slicing the results

After running any model on `custom_eval.jsonl`:
```bash
python scripts/run_matrix.py --models <m> --benchmark data/probes/custom_eval/custom_eval.jsonl --device cuda
python scripts/analyze_custom_eval.py     # -> docs/results/custom_eval_breakdown.md
```
yields per-class, per-language, rotation-retention, reading-direction and spotting tables — the
proposed evaluation matrix.

> Rendering note: CJK/QR/barcode need fonts + libs (`fonts-noto-cjk`, `qrcode`, `python-barcode`);
> the generator skips a class gracefully and flags it if a dependency is missing. Ground-truth boxes
> are computed from the actual render, so spotting GT stays correct under any fallback font.
