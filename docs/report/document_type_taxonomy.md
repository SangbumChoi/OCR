# Document-type taxonomy — a type × stressor lens for "document understanding"

"Documents with text" is far wider than invoices and papers. To design evaluation that
generalises, we organise document *types* against the **stressors** they impose on a VLM, then
map each stressor to the metric/axis already in our proposed set
([`data/benchmarks/custom_eval`](../../data/benchmarks/custom_eval/README.md)). A type is only as
"hard" as the stressors it combines — so this lens turns an open-ended list into concrete
evaluation requirements.

## Stressor dimensions

| Stressor                   | What it tests                                            | Our axis / metric                               |
| -------------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| **Layout**                 | multi-column, dense, mixed regions, nested               | reading-order; per-class scoring                |
| **Read order / direction** | panel order, vertical (CJK), RTL (Arabic/Hebrew), reflow | `reading_direction`; order edit-distance        |
| **Language / script**      | non-latin, code-switching, ruby/furigana, diacritics     | `language` (per-lang NED)                       |
| **Degradation**            | blur, low-DPI, glare, fold, fade, noise, rotation        | rotation retention; robustness probe            |
| **Non-text class**         | table, chart, formula, barcode/QR, stamp, logo, figure   | `content_class` (TEDS / relaxed / decode / IoU) |
| **Handwriting**            | cursive, margins, mixed print+hand, signatures           | NED / CER on handwriting                        |
| **Hallucination risk**     | absent fields, redaction, illegible regions              | anti-hallucination (correct-or-abstain)         |
| **Spotting need**          | auditable extraction (where each value came from)        | grounding IoU + reasoning                       |

## The matrix (type × dominant stressors)

✓ = primary stressor, ◐ = secondary. Last column = the metric that should anchor that type.

| Document type                   | Layout   | Order/Dir          | Lang          | Degrade         | Non-text           | Handwrite       | Halluc. | Spot | Anchor metric                |
| ------------------------------- | :------: | :----------------: | :-----------: | :-------------: | :----------------: | :-------------: | :-----: | :--: | ---------------------------- |
| Invoice / receipt               | ◐        |                    |               | ◐               | ◐(table)           |                 | ✓       | ✓    | KIE F1 + spotting IoU        |
| Bank stmt / payslip             | ✓        |                    |               |                 | ✓(table)           |                 | ✓       | ✓    | TEDS + F1                    |
| Contract / NDA                  | ✓        |                    |               |                 | ◐(stamp)           | ◐(sign)         | ✓       | ◐    | F1 + clause NED              |
| Research paper                  | ✓        | ✓                  | ◐             |                 | ✓(formula/fig)     |                 |         | ◐    | read-order + NED + TEDS      |
| Textbook / exam                 | ✓        | ◐                  | ◐             |                 | ✓(figure)          | ✓(answers)      |         |      | NED + CER(hand)              |
| Slides / poster                 | ✓        | ◐                  |               |                 | ✓(chart/logo)      |                 |         |      | per-class                    |
| Newspaper / magazine            | ✓✓       | ✓                  | ◐             | ◐               | ✓(figure)          |                 |         |      | read-order edit-dist         |
| **Webtoon / manga**             | ✓        | ✓✓(panel/RTL/vert) | ✓             |                 | ✓(SFX art-text)    | ◐               |         | ◐    | read-order + direction + NED |
| Advertisement / flyer           | ✓        |                    | ◐             |                 | ✓(logo/figure)     |                 | ◐       |      | per-class + presence         |
| Menu / packaging label          | ◐        |                    | ✓             | ◐(curved/glare) | ◐(barcode)         |                 |         | ◐    | NED + decode                 |
| Road sign / signage             |          | ◐                  | ✓             | ✓(angle/persp)  | ◐(symbol)          |                 |         | ◐    | NED + orientation            |
| Map / floor plan                | ✓        | ✓(angled labels)   | ◐             |                 | ✓(figure)          |                 |         | ✓    | spotting + read-order        |
| **ID / passport / license**     | ◐        | ◐(MRZ)             | ✓             | ◐               | ✓(photo/stamp)     | ◐(sign)         | ✓✓      | ✓    | KIE F1 + abstain + IoU       |
| Certificate / diploma           | ◐        |                    | ✓             | ◐               | ✓(seal)            | ◐(sign)         | ✓       | ◐    | F1 + stamp IoU               |
| **Cheque / 수표**                 | ◐        |                    |               | ◐               | ◐(MICR)            | ✓(amount/sign)  | ✓       | ✓    | dual-amount F1 + sign detect |
| **Prescription / 처방전**          | ◐        |                    | ◐             | ✓               |                    | ✓✓(doctor hand) | ✓       | ✓    | CER(hand) + abstain          |
| Ticket / boarding pass          | ✓        |                    | ◐             |                 | ✓(barcode/QR)      |                 |         | ✓    | decode + KIE F1              |
| **Ancient manuscript / 고문서**    | ◐        | ✓(vertical/RTL)    | ✓✓(classical) | ✓✓(fade/stain)  |                    | ✓(calligraphy)  | ◐       |      | NED + robustness             |
| Ledger / census (historical)    | ✓(table) |                    | ◐             | ✓               | ✓(table)           | ✓               |         | ✓    | TEDS + CER(hand)             |
| Microfilm / carbon copy         | ◐        |                    |               | ✓✓              |                    |                 | ◐       |      | robustness retention         |
| **LCD / meter / 7-seg**         |          |                    |               | ◐(glare)        | ✓(non-font digits) |                 |         | ✓    | exact + IoU                  |
| **UI / chat / code screenshot** | ✓        | ✓(reflow)          | ◐             |                 | ◐(icons)           |                 |         | ✓    | NED + spotting               |
| **Form w/ checkboxes**          | ✓        |                    |               | ◐               | ✓(selection marks) | ✓(fill-in)      | ✓       | ✓    | selection-mark acc + F1      |
| **Redacted document**           | ◐        |                    |               |                 | ◐(black bars)      |                 | ✓✓      | ✓    | abstain (no-hallucination)   |
| Meme / image+overlay text       |          |                    | ◐             | ◐               | ✓(figure)          |                 | ◐       | ◐    | NED + presence               |
| Music score / sheet             | ✓        | ✓                  |               |                 | ✓(notation)        | ◐               |         |      | symbol NED                   |
| Chemical / circuit diagram      | ✓        |                    |               |                 | ✓(structure)       |                 |         | ✓    | structure + spotting         |
| RTL doc (Arabic/Hebrew)         | ◐        | ✓✓(RTL)            | ✓✓            |                 |                    | ◐               |         |      | direction + per-lang NED     |

## Deep dives (the four prioritized groups)

### A. Entertainment / media — webtoon · manga · newspaper
The hardest **reading-order + direction** family. Manga reads **right→left**, classical CJK runs
**top→bottom**, webtoons are an **infinite vertical scroll** of stacked panels, newspapers are
multi-column with jumps ("continued on p.4"). Text is fused with art (speech bubbles, **SFX
onomatopoeia** stylised into the drawing). Failure modes: wrong panel/bubble order, treating art
text as body text, latin-only reading of vertical CJK. → exercises `reading_direction`, an
**order edit-distance**, and `language`; add a *panel/bubble order* item.

### B. Identity / administration — ID · cheque · prescription
**Highest hallucination cost.** A confidently-wrong passport number or cheque amount is worse than
"unreadable". Cheques have **dual amounts** (courtesy numeric vs legal words) that must agree;
prescriptions are **doctor handwriting** (notoriously hard) where abstaining beats guessing; IDs
need **MRZ parsing + photo/stamp localisation**. → exercises **KIE F1 + correct-or-abstain +
spotting IoU + handwriting CER**; the *basis-of-extraction* (where + why) matters most here.

### C. History / degradation — ancient manuscripts · ledgers · microfilm
Pushes **robustness + non-standard glyphs + direction** to the limit: fading, stains, bleed-through,
calligraphic/archaic scripts, vertical/RTL layouts, classical-language vocabulary. Ledgers add
**handwritten tables**. → exercises rotation/degradation **retention**, `language`, handwriting CER,
and TEDS on noisy tables.

### D. Digital / special markers — LCD · UI/chat · checkbox forms · redaction
Non-document-looking but text-bearing: **7-segment/LCD digits** (no real font), **UI screenshots**
(reflowing layout, icons, code), **selection marks** (checkbox/radio — a detection task, not OCR),
and **redacted** docs (must report "[redacted]"/absent, not invent). → exercises a **selection-mark
accuracy**, special-glyph NED, anti-hallucination, and spotting.

## Mapping to the current `custom_eval` (covered vs gaps)

| Stressor         | Covered now                                    | Gap to add                                              |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------- |
| content classes  | text/table/formula/chart/qr/barcode/stamp/logo | selection-marks, 7-seg digits, MRZ, music/chem notation |
| language         | en/ko/ja/zh                                    | **RTL (ar/he)**, classical/vertical Chinese             |
| read direction   | vertical CJK vs horizontal                     | **panel/bubble order**, RTL flow, multi-column reflow   |
| rotation/degrade | 0/15/90/180/270                                | fade/stain/glare/bleed-through (historical)             |
| hallucination    | (absence in spatial/context probe)             | **redaction**, illegible-region abstain                 |
| handwriting      | —                                              | **doctor-hand, cursive, ledger** (CER)                  |
| spotting         | boxes on stamp/logo/total                      | MRZ zones, cheque amount zones, checkbox cells          |

## Why this matters for the proposal
Each row is a **capability vector**, not a genre. A model can be production-ready for *invoices*
yet fail *prescriptions* purely on the handwriting+abstain stressors — and our eval makes that
visible by scoring along the stressors, not the document label. The taxonomy therefore doubles as
a **coverage checklist**: pick target document types → read off their stressors → ensure the
benchmark + metrics exercise each. The gaps table above is the concrete backlog for extending
`custom_eval`.
