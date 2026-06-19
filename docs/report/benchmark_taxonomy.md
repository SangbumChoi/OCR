# OCR & Document-Understanding Benchmarks — A Taxonomy of Tasks and Metrics

> **Why this document.** "OCR" is not one task. For document understanding we care about (a)
> reading the characters, (b) **extracting the specific information we want** (key-value /
> entity extraction), (c) recovering **table structure and cell relationships**, (d) whether
> particular content is present, and (e) **whether the output is faithful — no typos /
> hallucinations**. Each capability needs a *different* benchmark and a *different* metric.
> Collapsing them into a single "accuracy" hides exactly the failures that matter in
> production. This file maps the landscape so model and metric choices are justified, not
> assumed. **Selection rule for this PoC: open-weight models only** (reproducibility +
> fine-tunability + sub-1B + free-GPU local execution).

The capability axes, and which metric answers which question:

| If the question is…                                            | Use this metric                           | On benchmarks like         |
| -------------------------------------------------------------- | ----------------------------------------- | -------------------------- |
| "Did it read the text correctly, character by character?"      | **CER / WER / NED**                       | full-page OCR, CROHME      |
| "Did it answer the document question?"                         | **ANLS**                                  | DocVQA, InfoVQA            |
| "Did it extract the right *value* for this field?"             | **entity-level F1** (+ field exact-match) | FUNSD, CORD, SROIE, DocILE |
| "Did it get the table *structure / cell relationships*?"       | **TEDS / TEDS-Struct, GriTS**             | PubTabNet, PubTables-1M    |
| "Did it read the *chart value* right?"                         | **relaxed accuracy** (±5%)                | ChartQA, PlotQA            |
| "How good is its OCR *overall* across many sub-skills?"        | **OCRBench** (/1000)                      | OCRBench (v1/v2)           |
| "Can it parse a *whole page* end to end?"                      | **normalized edit distance** per element  | OmniDocBench               |
| "Does it know when it's wrong / does it hold up on bad scans?" | **ECE calibration, robustness retention** | (our custom probe)         |

---

## How the ten categories relate (families)

The ten below are **not ten disjoint tasks** — they cluster into a few families, and two pairs
overlap on purpose. Read the numbered sections through this lens:

- **A · Recognition / transcription — "read it all."** §1 full-page recognition, §2 scene-text,
  §9 end-to-end parsing. Same core skill (convert the *whole* input to text/structure), scored by
  the **edit-distance family** (CER/WER/NED, plus TEDS for the tables inside §9); they differ only
  in domain — clean page (§1) vs text in the wild (§2) vs complex multi-element page (§9).
- **B · Question answering & extraction — "get the specific information."** §3 Document VQA,
  §4 KIE, §6 chart QA. **§3 and §4 overlap on purpose** — see the boundary note in §3.
- **C · Structure recovery.** §5 tables, §7 formulas — the output is a *structure* (HTML tree /
  LaTeX), not free text.
- **D · Umbrella suite (not a peer task).** §8 OCRBench is an *aggregate* that re-bundles families
  A–C into one score; treat it as a breadth smoke-test, not a distinct capability (see §8).
- **E · Cross-cutting reliability.** §10 (calibration / robustness / hallucination) applies *on top
  of* any of the above, not beside them.

---

## 1. Full-page / printed-text recognition (transcription)

**Task.** Transcribe an entire page/line to text (optionally formatted markdown/LaTeX/HTML).

**Metrics.**
- **CER (Character Error Rate)** = (S+D+I)/N_chars — substitutions + deletions + insertions
  over reference characters (Levenshtein at char level). Lower is better.
- **WER (Word Error Rate)** — same at word level.
- **NED (Normalized Edit Distance)** = edit_distance / max(len_pred, len_gold); often reported
  as **1−NED** (higher better). The robust default when outputs are long.
- **BLEU / METEOR / F1** — used by GOT-OCR2.0 for formatted/long transcription where exact
  CER is too brittle (e.g. markdown/LaTeX). (GOT-OCR2.0, arXiv:2409.01704)

**Note.** CER/WER are the right tools for *recognition fidelity* and **typo detection** — a
model that hallucinates or drops characters is penalised directly, which generic VQA accuracy
would miss.

## 2. Scene-text detection & recognition

**Task.** Localize + read text "in the wild" (signs, products). Adjacent to documents but tests
irregular/curved/artistic text.

**Benchmarks.** ICDAR-2013/2015, Total-Text, CTW1500 (curved), COCO-Text, SVT, IIIT5K.

**Metrics.**
- **Detection H-mean (F-measure)** = 2·P·R/(P+R) over box matches at an IoU threshold.
- **End-to-end / recognition: word accuracy** (case-insensitive exact) and **1−NED**.

## 3. Document Visual Question Answering (VQA)

**Task.** Answer a natural-language question about a document image. The core "understanding"
benchmark family and the anchor of this PoC.

**Benchmarks.** **DocVQA** (scanned business docs), **InfographicVQA** (dense infographics,
layout+numeric reasoning), **ST-VQA** (scene text), **TextVQA**, **VisualMRC** (long-form).

**Metric — ANLS (Average Normalized Levenshtein Similarity).** For a prediction p and gold set
{g_i}: NLS(p,g)=1−edit(p,g)/max(|p|,|g|); take the best g_i; **if best < 0.5 → score 0**,
else score = best. Averaged over questions. Designed to tolerate minor OCR/format differences
(`$1,200` vs `1200`) without rewarding near-misses. (DocVQA, WACV'21; InfoVQA, WACV'22)
VisualMRC (free-form) additionally uses **BLEU / METEOR / CIDEr / ROUGE-L**.

**The §3 ↔ §4 boundary — a spectrum, not a wall.** Document-VQA questions span two natures, and
the lower end *is* KIE:

- **3a — extractive ("KIE in question form").** "What is the total?" / "Who is the vendor?" — the
  answer is a *single field value* lifted from one region. This is **the same skill as §4 KIE**; the
  only difference is the **interface + metric**: a free-form NL question scored by **ANLS** here, vs
  a fixed field schema scored by **entity-F1** in §4. So §3 *contains* §4-style questions.
- **3b — integrative.** "How much higher is Q3 than Q1?" / "Which region has the most branches?" —
  the answer needs **related information combined across regions** (arithmetic, comparison,
  multi-hop), not just one given field. This is where the real "understanding" (and the failure of
  small models) lives.

Practical rule: a **fixed set of fields** with per-field precision/recall → use the **§4 KIE**
framing (entity-F1); **open-ended NL questions** (esp. 3b) → use **§3 ANLS**. Same underlying
reading; different evaluation contract.

## 4. Key Information Extraction (KIE) — "extract the value I want"

**Task.** Pull *specific* fields/entities from a document (total, date, vendor, line items) and
their **key↔value relations** — not transcribe everything. This is the **structured-schema
counterpart of the extractive §3a questions**: the same extraction skill, but driven by a fixed
field schema and scored per-field with entity-F1 instead of a free-form question scored by ANLS.

**Benchmarks.**
- **SROIE** — receipts; 4 fields (company, date, address, total). Field-level F1 / exact match.
- **CORD** — receipts; ~30 fine-grained fields + hierarchy. Entity F1.
- **FUNSD** — noisy scanned forms; entities (question/answer/header) **and linking** (key→value
  pairs). Entity F1 + relation/linking F1.
- **DocILE** — large-scale business docs; KILE (key information localization+extraction) and
  LIR (line-item recognition). F1 / AP.
- **XFUND** (multilingual FUNSD), **EPHOIE**, **Kleister** (long docs).

**Metric — entity-level F1** = 2·P·R/(P+R), where a predicted entity counts as correct only if
**both its type and its (normalised) value match** gold; line-item / relation tasks add
**relation F1** for key↔value links. This is the right metric for "did it extract the right
value for this field" — ANLS or VQA accuracy do **not** capture per-field precision/recall or
relational structure.

## 5. Table recognition & extraction — "table structure / cell relationships"

**Task.** Recover a table's **structure** (rows/cols/spans) and cell contents — i.e. the
*relationships* between cells, exactly as you described.

**Benchmarks.** **PubTabNet** (~568k), **FinTabNet**, **PubTables-1M** (~948k, also detection),
**SciTSR**, **WTW** (wild), **TableBank**.

**Metrics.**
- **TEDS (Tree-Edit-Distance-based Similarity)** = 1 − TreeEditDist(pred_tree, gold_tree)/
  max(nodes). Represents the table as an HTML tree (structure **+** cell text); captures
  merged cells and relationships. **TEDS-Struct** ignores cell text → pure structure. (PubTabNet,
  ECCV'20)
- **GriTS (Grid Table Similarity)** — matches predicted vs gold cell grids and reports
  precision/recall/F over content, location, and topology; arguably more faithful than TEDS for
  spanning cells. (PubTables-1M / Microsoft Table Transformer)
- **Cell-adjacency F1** (ICDAR-2013 style) — correctness of neighbour relationships between
  cells. Directly measures "which cells relate to which".

**Note.** For "correlation across a table" → **TEDS / GriTS** are the answer; a flat text metric cannot
score whether two cells are correctly in the same row/column or span.

## 6. Chart understanding

**Task.** Read and reason over plotted data (bar/line/pie).

**Benchmarks.** **ChartQA** (human + machine-augmented Qs), **PlotQA**, **Chart-to-Text**
(summarization), **ChartX / ChartBench**.

**Metric — relaxed accuracy.** A numeric answer is correct if within **5% relative error** of
gold; non-numeric uses exact match. (ChartQA, ACL Findings'22) Chart-to-Text uses BLEU/CIDEr.

## 7. Formula / mathematical-expression recognition

**Task.** Image → LaTeX (printed or handwritten math).

**Benchmarks.** **im2latex-100k** (printed), **CROHME** (handwritten).

**Metrics.** **Exact match**, **token accuracy**, **BLEU**, and **normalized edit distance** on
the LaTeX token sequence; CROHME also uses expression-level recognition rate.

## 8. Comprehensive OCR-capability suites for LMMs (umbrella — not a peer task)

**This is an aggregate, not a distinct capability.** Unlike §§1–7, category 8 does not test one new
skill — it **re-bundles the others** into a single breadth score, so think of it as a smoke-test
("does this LMM do OCR at all?") rather than a sibling task in the same list.

**OCRBench** — 1,000 curated items across **5 capability groups that map onto the categories
above**: (1) text recognition → §1, (2) scene-text-centric VQA → §2/§3, (3) document-oriented VQA
→ §3, (4) **key information extraction** → §4, (5) **handwritten mathematical expression**
recognition → §7. Scored **out of 1000** (1 point per correct item; a gold answer counted correct
if contained in the prediction). It is the single most useful "does this LMM do OCR broadly" probe.
(Liu et al., 2023) **OCRBench v2** extends to ~10k items and 31 scenarios with finer
text-spotting/structure tasks.

Related: **SEED-Bench-2-Plus** (text-rich images), **CharXiv** (scientific-figure reasoning).

## 9. End-to-end document PARSING

**Task.** Convert a full page to structured output (text + tables + formulas + reading order) —
the closest to "real document digitization".

**Benchmark — OmniDocBench.** Diverse real PDFs (papers, books, slides, financial, exams).
**Metric:** per-element **Normalized Edit Distance** for text/formula, **TEDS** for tables,
and a **reading-order** edit distance, aggregated into an overall edit-distance score (lower
better). This is what PaddleOCR-VL and dots.ocr / MinerU-style systems report. (**Fox** is a
related focused/fine-grained page benchmark.)

## 10. Reliability: calibration, robustness, hallucination

Public leaderboards almost never report these, yet they decide deployability — and they speak
directly to your "are there typos / hallucinations?" concern:

- **Calibration — ECE (Expected Calibration Error)** = Σ_bin (|B|/N)·|acc(B)−conf(B)|. Does the
  model's confidence track its correctness? A well-calibrated reader lets you **route
  low-confidence fields to human review**. (Guo et al., ICML'17) — *implemented in this repo.*
- **Robustness retention** = score(perturbed)/score(clean) under realistic degradations
  (downscale, JPEG, blur, skew, noise) + **terminology paraphrase**. Measures stability on
  phone-photo/fax/jargon inputs. — *implemented in this repo (custom probe).*
- **Hallucination / faithfulness** — for transcription, **CER/NED** already penalise invented
  characters; dedicated VLM-hallucination suites (POPE-style) target object hallucination and
  are less central to OCR but relevant for free-form document answers.

---

## How this PoC instantiates the taxonomy

| Capability (above)     | In this PoC                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| 3 Document VQA         | **DocVQA, InfoVQA** — ANLS (implemented)                                      |
| 6 Chart                | **ChartQA** — relaxed accuracy (implemented)                                  |
| 8 LMM OCR suite        | **OCRBench** — /1000 scoring (implemented)                                    |
| 1 Recognition fidelity | **CER/WER/NED** — defined here; metric hooks in `metrics/text.py`             |
| 4 KIE                  | **entity F1** — defined here; add CORD/SROIE loader to extend (see README §3) |
| 5 Tables               | **TEDS/GriTS** — defined here; add PubTabNet loader to extend                 |
| 10 Reliability         | **ECE + robustness retention** — implemented (the "beyond accuracy" core)     |

The evaluation harness already covers axes 3/6/8/10 end-to-end; axes 1/4/5 are documented with
exact metric definitions and are a drop-in extension (new loader + the metric, registered the
same way as the others), so the comparison can grow from "document QA" to full "document
understanding" without changing the pipeline.

### References (canonical)
DocVQA (WACV'21) · InfographicVQA (WACV'22) · ST-VQA (ICCV'19) · TextVQA (CVPR'19) · VisualMRC
(AAAI'21) · SROIE (ICDAR'19) · CORD (NeurIPS-W'19) · FUNSD (ICDAR-W'19) · DocILE (2023) ·
XFUND (ACL'22) · PubTabNet/EDD (ECCV'20) · PubTables-1M/GriTS (CVPR'22 / 2022) · ChartQA (ACL-F'22) ·
PlotQA (WACV'20) · im2latex (2017) · CROHME · OCRBench (2023) / OCRBench v2 (2024) ·
OmniDocBench (CVPR'25) · Fox (2024) · Guo et al. Calibration (ICML'17).
