#!/usr/bin/env python3
"""Pseudo-label plan for UDD: what could a SOTA open-source OCR model fill in?

Default (and currently only wired) mode is **--plan**: report, per filler (full_text /
region_text / table_html), how many rows are missing the label and which sources they come from —
no model is loaded. The actual fill (``docvlm_eval.unified.pseudo_label.apply`` with a got-ocr2 /
paddleocr-vl labeler) is deliberately left as the GPU follow-up; provenance (`pseudo_json`) and
never-overwrite-gold semantics are already enforced by the pipeline.

    python scripts/pseudo_label_udd.py                 # plan on data/udd/hf/_all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified.pseudo_label import FILLERS, plan  # noqa: E402

MD = ROOT / "docs" / "results" / "udd_pseudo_label_plan.md"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    args = p.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.src)
    rep = plan(ds)

    lines = ["# UDD pseudo-labeling plan (no inference run — future GPU work)", "",
             f"Corpus: {rep['total_rows']} image-rows. For each fillable column: how many rows a "
             "SOTA open-source OCR model could label, and where they come from. Provenance design: "
             "filled values land with a `pseudo_json` marker (`{column: labeler}`), gold is never "
             "overwritten.", "",
             "| filler | column | rows needing fill | share | suggested models | top sources |",
             "|---|---|---|---|---|---|"]
    for name in FILLERS:
        r = rep[name]
        srcs = ", ".join(f"{s} ({n})" for s, n in list(r["by_source_top"].items())[:4])
        lines.append(f"| {name} | `{r['column']}` | {r['rows_needing_fill']} | "
                     f"{100 * r['share']:.0f}% | {', '.join(r['suggested_models'])} | {srcs} |")
    lines += ["", "Run the fill (GPU): `docvlm_eval.unified.pseudo_label.apply(ds, '<filler>', "
              "labeler=..., name='got-ocr2')` — the repo's `docvlm_eval.models` already wraps "
              "got-ocr2 and paddleocr-vl for generation."]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2)[:1200])
    print(f"\n[ok] {MD}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
