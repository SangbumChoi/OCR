# DAR worked example

`invoice_0001.png` + `invoice_0001.json`: one record of the layered document annotation format
(DAR, [`docs/report/annotation_format.md`](../../../docs/report/annotation_format.md)) with every
layer filled in:

- **page**: upright, digital invoice, `en`
- **layout**: 7 regions in reading order (title, vendor, metadata, table, totals, stamp, footer)
- **ocr**: 30 lines; the boxes are exact, read from the renderer
- **table**: a 4×4 grid with a header row, each cell linked to its OCR line
- **kie**: 8 present fields with normalized values, plus `invoice.po_number` marked `present: false`
- **understanding**: a caption, 5 reading-guide notes (DD/MM date convention, the total already
  includes VAT, the PAID stamp is not a field, ...) and 6 QAs. Five of them have grounded
  reasoning chains, and one is an abstain question.

It is the fixture for `tests/test_annotation.py`. Try:

```bash
python scripts/annotate_dar.py validate data/annotations/example -v
python scripts/annotate_dar.py stats data/annotations/example
python scripts/annotate_dar.py export data/annotations/example --arm L2_all --out /tmp/dar_l2
```
