#!/usr/bin/env python3
"""Regenerate data/benchmarks/README.md from configs/benchmark_catalog.yaml.

Groups all benchmarks by their capability-category number, shows status (image sample vs
documented-only), metric and purpose. Reproducible (run after editing the catalog or fetching).

    python scripts/build_benchmark_index.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmarks"
CATALOG = ROOT / "configs" / "benchmark_catalog.yaml"

CATNAMES = {
    1: "Text recognition (full-page / line / word)",
    2: "Scene-text detection & recognition",
    3: "Document / scene-text / diagram VQA",
    4: "Key Information Extraction (KIE)",
    5: "Table recognition & structure",
    6: "Chart / plot / figure reasoning",
    7: "Formula / math-expression recognition",
    8: "Comprehensive LMM OCR & figure suites",
    9: "End-to-end page parsing & layout",
    10: "Reliability: robustness / calibration / hallucination",
    11: "Custom capability axes (our probe)",
}


def cnum(e: dict) -> int:
    try:
        return int(str(e["category"]).split(".")[0])
    except Exception:
        return 99


def status(e: dict) -> str:
    png = BENCH / e["key"] / "sample.png"
    js = BENCH / e["key"] / "sample.json"
    src = (e.get("source") or "").lower()
    if png.exists():
        return "🖼️ sample (synthetic)" if ("synthetic" in src or "derived" in src) else "🖼️ sample (HF)"
    if js.exists():
        return "📄 label only"
    return "📝 documented" if not e.get("hf_id") else "📝 documented (HF n/a)"


def main() -> None:
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["benchmarks"]
    groups: "OrderedDict[int, list]" = OrderedDict()
    for e in sorted(cat, key=cnum):
        groups.setdefault(cnum(e), []).append(e)
    withimg = len(list(BENCH.glob("*/sample.png")))

    L = ["# Benchmark catalog & sample previews\n",
         "Every benchmark across the capability categories of",
         "[`../../docs/report/benchmark_taxonomy.md`](../../docs/report/benchmark_taxonomy.md) and",
         "[`../../docs/report/capability_axes.md`](../../docs/report/capability_axes.md), annotated with **what",
         "each one measures** (`purpose`). Source of truth:",
         "[`../../configs/benchmark_catalog.yaml`](../../configs/benchmark_catalog.yaml).\n",
         "- 🖼️ **sample** = image + `sample.json` (GT + metric + purpose) in `<key>/`.",
         "- 📝 **documented** = not cleanly streamable from HF; catalogued with purpose + source.\n",
         f"**Coverage: {withimg} image samples across {len(cat)} catalogued benchmarks.**\n"]
    for n, items in groups.items():
        L.append(f"### {n}. {CATNAMES.get(n, 'Other')}\n")
        L.append("| Benchmark | Status | Metric | Purpose (what it measures) |")
        L.append("|---|---|---|---|")
        for e in items:
            L.append(f"| [`{e['key']}`]({e['key']}/) | {status(e)} | {e.get('metric','-')} | {e.get('purpose','-')} |")
        L.append("")
    from docvlm_eval.report_md import prettify_tables
    (BENCH / "README.md").write_text(prettify_tables("\n".join(L)) + "\n", encoding="utf-8")
    print(f"[done] index: {withimg} image samples / {len(cat)} benchmarks / {len(groups)} categories")


if __name__ == "__main__":
    main()
