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
PROBES = ROOT / "data" / "probes"
CATALOG = ROOT / "configs" / "benchmark_catalog.yaml"


def entry_dir(e: dict) -> Path:
    """Where the entry's samples live: real public benchmarks under data/benchmarks/, our own
    synthetic/probe sets under data/probes/ (kind: synthetic|probe)."""
    root = PROBES if e.get("kind") in ("synthetic", "probe") else BENCH
    return root / e["key"]

# Capability families (see docs/report/benchmark_taxonomy.md). Each catalog entry's `category`
# is a code like "B2. …"; the leading letter is its family.
FAMILYNAMES = {
    "A": "Recognition / transcription (full-page · scene-text · end-to-end parsing)",
    "B": "Question answering & extraction (VQA · KIE · chart)",
    "C": "Structure recovery (tables · formulas)",
    "D": "Umbrella OCR & figure suites",
    "E": "Reliability: robustness / calibration / hallucination",
    "F": "Custom capability axes (our probes)",
}


def ccode(e: dict) -> str:
    """Task code, e.g. 'B2', from the category string 'B2. Name'."""
    return str(e.get("category", "")).split(".")[0].strip()


def family(e: dict) -> str:
    c = ccode(e)
    return c[0] if c else "Z"


def status(e: dict) -> str:
    d = entry_dir(e)
    src = (e.get("source") or "").lower()
    if (d / "sample.png").exists():
        return "🖼️ sample (synthetic)" if ("synthetic" in src or "derived" in src) else "🖼️ sample (HF)"
    if (d / "sample.json").exists():
        return "📄 label only"
    return "📝 documented" if not e.get("hf_id") else "📝 documented (HF n/a)"


def main() -> None:
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["benchmarks"]
    groups: "OrderedDict[str, list]" = OrderedDict()
    for e in sorted(cat, key=ccode):
        groups.setdefault(family(e), []).append(e)
    withimg = len(list(BENCH.glob("*/sample.png"))) + len(list(PROBES.glob("*/sample.png")))

    L = ["# Benchmark catalog & sample previews\n",
         "Every benchmark across the capability families of",
         "[`../../docs/report/benchmark_taxonomy.md`](../../docs/report/benchmark_taxonomy.md) and",
         "[`../../docs/report/capability_axes.md`](../../docs/report/capability_axes.md), annotated with **what",
         "each one measures** (`purpose`). Source of truth:",
         "[`../../configs/benchmark_catalog.yaml`](../../configs/benchmark_catalog.yaml).\n",
         "- 🖼️ **sample** = image + `sample.json` (GT + metric + purpose) in `<key>/`.",
         "- 📝 **documented** = not cleanly streamable from HF; catalogued with purpose + source.",
         "- Real public benchmarks live in `data/benchmarks/`; our own synthetic/probe sets "
         "(`kind: synthetic|probe`) live in `../probes/`.\n",
         f"**Coverage: {withimg} image samples across {len(cat)} catalogued benchmarks.**\n"]
    for fam in sorted(groups):
        L.append(f"### {fam}. {FAMILYNAMES.get(fam, 'Other')}\n")
        L.append("| Code | Benchmark | Status | Metric | Purpose (what it measures) |")
        L.append("|---|---|---|---|---|")
        for e in sorted(groups[fam], key=ccode):
            rel = (".." / entry_dir(e).relative_to(ROOT / "data")) if e.get("kind") in ("synthetic", "probe") else Path(e["key"])
            L.append(f"| {ccode(e)} | [`{e['key']}`]({rel}/) | {status(e)} | "
                     f"{e.get('metric','-')} | {e.get('purpose','-')} |")
        L.append("")
    from docvlm_eval.report_md import prettify_tables
    (BENCH / "README.md").write_text(prettify_tables("\n".join(L)) + "\n", encoding="utf-8")
    print(f"[done] index: {withimg} image samples / {len(cat)} benchmarks / {len(groups)} families")


if __name__ == "__main__":
    main()
