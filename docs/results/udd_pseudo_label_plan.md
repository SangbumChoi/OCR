# UDD pseudo-labeling plan (no inference run — future GPU work)

Corpus: 9089 image-rows. For each fillable column: how many rows a SOTA open-source OCR model could label, and where they come from. Provenance design: filled values land with a `pseudo_json` marker (`{column: labeler}`), gold is never overwritten.

| filler | column | rows needing fill | share | suggested models | top sources |
|---|---|---|---|---|---|
| full_text | `full_text` | 8140 | 90% | got-ocr2, paddleocr-vl | ai2d (300), charxiv (300), doclaynet (300), docvqa (300) |
| region_text | `elements_json` | 994 | 11% | got-ocr2, paddleocr-vl | doclaynet (300), publaynet (300), omnidocbench (274), ocrvqa (120) |
| table_html | `table_html` | 0 | 0% | got-ocr2 |  |

Run the fill (GPU): `docvlm_eval.unified.pseudo_label.apply(ds, '<filler>', labeler=..., name='got-ocr2')` — the repo's `docvlm_eval.models` already wraps got-ocr2 and paddleocr-vl for generation.
