#!/usr/bin/env python
"""Count every catalogued UDD anomaly (src/docvlm_eval/unified/anomalies.py) over the corpus.

    python scripts/audit_udd_anomalies.py --from-hub          # image column is never downloaded
    python scripts/audit_udd_anomalies.py --rows rows.json

Writes docs/results/udd_anomalies.json (counts, per-source split, examples) and
docs/results/udd_anomalies.md (the table). Narrative: docs/report/udd_anomaly_catalogue.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from derive_udd_reasoning import load_rows

from docvlm_eval.unified.anomalies import FAMILIES, REGISTRY, audit_row

STATUS_LABEL = {"fixed_in_derived": "fixed (derived data)", "fixed_in_loader": "fixed (loader, needs rebuild)",
                "gated": "gated", "skipped_by_design": "skipped by design", "open": "open"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-hub", action="store_true")
    src.add_argument("--rows")
    ap.add_argument("--json", default=str(ROOT / "docs" / "results" / "udd_anomalies.json"))
    ap.add_argument("--md", default=str(ROOT / "docs" / "results" / "udd_anomalies.md"))
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)

    rows = load_rows(args)
    hits, rows_hit = Counter(), Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    for r in rows:
        for aid, h in audit_row(r).items():
            hits[aid] += len(h)
            rows_hit[aid] += 1
            by_source[aid][r["source"]] += len(h)
            if len(examples[aid]) < args.examples:
                examples[aid].append({"sample_id": r["sample_id"], "detail": h[0]})
    out = {"n_rows": len(rows), "anomalies": []}
    for a in REGISTRY:
        out["anomalies"].append({
            "id": a.id, "family": a.family, "title": a.title, "impact": a.impact,
            "handling": a.handling, "status": a.status, "hits": hits[a.id], "rows": rows_hit[a.id],
            "by_source": dict(by_source[a.id].most_common()), "examples": examples[a.id]})
    Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# UDD anomaly audit", "",
             (f"Counts over {len(rows):,} UDD rows, produced by `scripts/audit_udd_anomalies.py`. "
              "Definitions and handling: [`../report/udd_anomaly_catalogue.md`]"
              "(../report/udd_anomaly_catalogue.md)."), "",
             "| id | family | anomaly | hits | rows | main sources | status |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for fam in FAMILIES:
        for a in out["anomalies"]:
            if a["family"] != fam:
                continue
            srcs = ", ".join(f"{s} {n:,}" for s, n in list(a["by_source"].items())[:3]) or "—"
            lines.append(f"| {a['id']} | {fam} | {a['title']} | {a['hits']:,} | {a['rows']:,} | {srcs} | "
                         f"{STATUS_LABEL[a['status']]} |")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for a in out["anomalies"]:
        print(f"{a['id']:3s} {a['family']:11s} hits={a['hits']:6d} rows={a['rows']:6d}  {a['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
