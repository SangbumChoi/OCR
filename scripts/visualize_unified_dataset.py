#!/usr/bin/env python3
"""Visualize unified-loader examples across ALL datasets in one montage.

Loads a few examples from every streamable benchmark through the unified loader (caching images to a
scratch dir), then renders a grid where each cell shows the image + task badge + a task-appropriate
overlay (KIE field boxes = green, localization regions = orange, table/recognition/vqa captions).
Writes to docs/report/figures/unified_examples.png.

    python scripts/visualize_unified_dataset.py --per-bench 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import UnifiedLoader, render_grid  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" / "unified_examples.png"))
    p.add_argument("--cache", default=str(ROOT / "data" / "unified_dataset" / "images"))
    p.add_argument("--per-bench", type=int, default=1, help="examples (distinct images) per dataset")
    p.add_argument("--only", default=None, help="comma-separated benchmark keys")
    p.add_argument("--skip", default=None, help="comma-separated benchmark keys to skip")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--max-scan", type=int, default=60)
    args = p.parse_args()

    loader = UnifiedLoader()
    only = [k.strip() for k in args.only.split(",")] if args.only else None
    skip = [k.strip() for k in args.skip.split(",")] if args.skip else None
    print("loading unified examples across datasets (this streams HF; a few minutes)...")
    by_key = loader.load_all(limit_per=args.per_bench, only=only, skip=skip,
                             max_scan=args.max_scan, cache_dir=args.cache)
    # one representative row per (dataset) — first record of each, plus extra rows up to per-bench
    rows = [r for k in by_key for r in by_key[k]]
    if not rows:
        print("No examples loaded (network?)."); return
    render_grid(rows, args.out, cols=args.cols,
                title=f"Unified loader — examples across {len(by_key)} datasets / "
                      f"{len({r.task for r in rows})} tasks")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort
