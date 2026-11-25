# UDD anomaly audit

Counts over 39,837 UDD rows, produced by `scripts/audit_udd_anomalies.py`. Definitions and handling: [`../report/udd_anomaly_catalogue.md`](../report/udd_anomaly_catalogue.md).

| id | family | anomaly | hits | rows | main sources | status |
| --- | --- | --- | ---: | ---: | --- | --- |
| G1 | gold | Derivation embedded in the gold answer | 3,057 | 1,453 | tatqa 3,051, docmatix 6 | fixed (derived data) |
| G2 | gold | Stray trailing period on a short gold | 23,076 | 6,719 | plotqa 7,500, dvqa 7,497, tatqa 4,813 | fixed (derived data) |
| G3 | gold | Markdown heading marks in title text | 351 | 160 | omnidocbench 351 | fixed (derived data) |
| G4 | gold | Abstain token as the gold | 86 | 86 | screenqa 60, ocrbench_v2 26 | gated |
| B1 | geometry | Pixel box outside the stored image (pre-resize frame) | 2,078 | 894 | screenqa 1,163, cord 915 | fixed (loader, needs rebuild) |
| B2 | geometry | Answer word recurs far from the title (tagline, spine) | 151 | 120 | ocrvqa 151 | fixed (derived data) |
| B3 | geometry | Title/author not found in the cover OCR | 1,229 | 941 | ocrvqa 1,229 | gated |
| C1 | convention | PubLayNet 'Title' also labels figure tags | 43 | 31 | publaynet 43 | skipped by design |
| C2 | convention | Table with merged cells (row/col spans) | 575 | 575 | pubtabnet 575 | skipped by design |
| C3 | convention | Receipt amounts that do not add up | 4 | 4 | cord 4 | skipped by design |
| R1 | rationale | Source rationale only restates the question | 125 | 125 | hallusionbench 125 | gated |
| R2 | rationale | Rationale opens with the PAIRED question's verdict | 3 | 3 | hallusionbench 3 | fixed (derived data) |
| R3 | rationale | Derivation text without values (a plan, not evidence) | 15 | 11 | tatqa 15 | gated |
| R4 | rationale | Count derivation listed with '##' separators | 109 | 107 | tatqa 109 | fixed (derived data) |
| I1 | instruction | Brevity suffix that conflicts with a new format clause | 17,003 | 4,500 | plotqa 7,500, dvqa 7,497, stvqa 1,950 | fixed (derived data) |
| I2 | instruction | Non-English question | 3,424 | 1,469 | mtvqa 2,959, ocrbench_v2 399, mathvista 64 | gated |
| I3 | instruction | Unnumbered inline option list inside a sentence | 201 | 201 | ocrbench_v2 200, tatqa 1 | fixed (derived data) |
| N1 | number | Dot as thousands separator ('60.000') | 171 | 33 | cord 171 | fixed (derived data) |
| N2 | number | Accounting negative in a derivation ('(6,919)') | 5 | 4 | tatqa 5 | fixed (derived data) |
| N3 | number | Percent used as a fraction ('65% * 3,433') | 10 | 9 | tatqa 10 | fixed (derived data) |
| N4 | number | Decimal with a leading zero and 3 decimals ('0.039') | 10 | 10 | charxiv 8, docvqa 2 | fixed (derived data) |
