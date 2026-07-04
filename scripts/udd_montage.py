#!/usr/bin/env python3
"""Regenerate the UDD examples montage from the MERGED corpus: one cell per source.

``build_udd.py`` renders a montage of whatever it just built, so an incremental run (``--only
newkey``) leaves a montage with only the new cells. This script always renders **one example per
source from the full merged dataset** (KIE fields green, localization regions orange), writing the
card asset used by the Hub README and docs.

    python scripts/udd_montage.py                       # from data/udd/hf/_all
    python scripts/udd_montage.py --repo danelcsb/UDD   # from the Hub
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import render_grid, unified_from_hf_row  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    p.add_argument("--repo", default=None, help="HF repo to pull instead of --src")
    p.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" / "udd_examples.png"))
    p.add_argument("--tmp", default=str(ROOT / "data" / "udd" / "viz_imgs"))
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    if args.repo:
        from datasets import load_dataset
        ds = load_dataset(args.repo, split="train")
    else:
        from datasets import load_from_disk
        ds = load_from_disk(args.src)

    tmp = Path(args.tmp); tmp.mkdir(parents=True, exist_ok=True)
    sources = ds["source"]
    seen: set[str] = set()
    rows = []
    for i in range(len(ds)):
        s = sources[i]
        if s in seen:
            continue
        seen.add(s)
        row = {k: ds[k][i] for k in ds.column_names if k != "image"}
        ip = tmp / f"{s}_{i}.png"
        ds[i]["image"].convert("RGB").save(ip)
        rows.append(unified_from_hf_row(row, image_path=str(ip)))
    render_grid(rows, args.out, cols=args.cols,
                title=f"UDD — {len(rows)} datasets, one example each "
                      f"(KIE=green, localization=orange)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort
