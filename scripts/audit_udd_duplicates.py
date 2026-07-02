#!/usr/bin/env python3
"""Duplicate audit across the merged UDD: exact byte-dupes + perceptual near-dupes, by source.

Public benchmarks recycle each other's images (COCO photos appear in TextVQA *and* ST-VQA; DocVQA
pages come from the same IDL crawl as Docmatix), so a merged corpus needs a duplication report —
both to keep training honest (a val image hiding in another source's train rows) and to explain
metric transfer between sources. Two detectors:

* **exact** — md5 of decoded RGB pixels: identical stored images (same page, same crop, same size).
* **near**  — 64-bit difference hash (``enrich.dhash``) with Hamming distance ≤ ``--near``:
  re-encodes, resizes and small crops of the same underlying image that byte-hashes can't see.

Writes ``docs/results/udd_duplicates.md`` (cross-source pair counts + within-source counts + sample
collisions). Uses the ``phash`` column when present (enriched corpus) and computes it otherwise.

    python scripts/audit_udd_duplicates.py                 # audit data/udd/hf/_all
    python scripts/audit_udd_duplicates.py --near 4        # stricter near-dup threshold
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import dhash, hamming  # noqa: E402

MD = ROOT / "docs" / "results" / "udd_duplicates.md"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    p.add_argument("--near", type=int, default=2,
                   help="max Hamming distance for a near-duplicate. Documents are mostly-white "
                        "low-entropy images, so dhash saturates fast: at 6 (the usual photo "
                        "threshold) this corpus shows ~1.3k cross-source 'pairs' full of false "
                        "positives (diagram≈receipt); at 2 the survivors are real re-uses.")
    p.add_argument("--out", default=str(MD))
    args = p.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.src)
    n = len(ds)
    have_phash = "phash" in ds.column_names
    print(f"[audit] {n} rows, phash column: {have_phash}")

    # one entry per IMAGE (many rows share an image within a source — that's QA fan-out, not dupes):
    # key = (source, image-identity from sample_id without the QA suffix)
    per_image: dict[tuple, dict] = {}
    for i in range(n):
        sid = ds["sample_id"][i]
        key = (ds["source"][i], sid.rsplit("_", 1)[0])
        if key in per_image:
            continue
        img = ds[i]["image"]
        per_image[key] = {
            "md5": hashlib.md5(img.convert("RGB").tobytes()).hexdigest(),
            "phash": (ds["phash"][i] if have_phash else "") or dhash(img),
            "sid": sid,
        }
    print(f"[audit] {len(per_image)} distinct image slots across {len({k[0] for k in per_image})} sources")

    # ---- exact dupes: same pixel md5
    by_md5: dict[str, list[tuple]] = defaultdict(list)
    for key, v in per_image.items():
        by_md5[v["md5"]].append(key)
    exact_groups = {h: ks for h, ks in by_md5.items() if len(ks) > 1}
    exact_cross = {h: ks for h, ks in exact_groups.items() if len({k[0] for k in ks}) > 1}

    # ---- near dupes: dhash Hamming <= threshold (skip exact-identical pairs; O(n^2) ints is fine
    # at this scale — ~3-4k images)
    items = [(key, int(v["phash"], 16)) for key, v in per_image.items()]
    near_pairs = []
    for i in range(len(items)):
        ki, hi = items[i]
        for j in range(i + 1, len(items)):
            kj, hj = items[j]
            if per_image[ki]["md5"] == per_image[kj]["md5"]:
                continue
            if bin(hi ^ hj).count("1") <= args.near:
                near_pairs.append((ki, kj))
    cross_near = [(a, b) for a, b in near_pairs if a[0] != b[0]]
    pair_counts = Counter(tuple(sorted((a[0], b[0]))) for a, b in cross_near)
    within_near = Counter(a[0] for a, b in near_pairs if a[0] == b[0])

    # ---- report
    lines = ["# UDD duplicate audit", "",
             f"{len(per_image)} distinct image slots, {len({k[0] for k in per_image})} sources. "
             f"Exact = identical decoded pixels (md5); near = dhash Hamming ≤ {args.near}.", "",
             "Threshold note: documents are mostly-white, low-entropy images, so perceptual hashes "
             "saturate much faster than on photos — at the usual photo threshold (≤6) this corpus "
             "reports ~1.3k cross-source 'pairs' dominated by false positives (e.g. diagram ≈ "
             "receipt). At ≤2 the survivors are genuine re-uses; treat anything between as "
             "candidates needing eyeballing.", "",
             f"- **Exact duplicate groups:** {len(exact_groups)} "
             f"({len(exact_cross)} cross-source)",
             f"- **Near-duplicate pairs:** {len(near_pairs)} ({len(cross_near)} cross-source)", ""]
    if exact_cross:
        lines += ["## Cross-source EXACT duplicates (same stored image in two sources)", ""]
        for h, ks in list(exact_cross.items())[:20]:
            lines.append("- " + "  ↔  ".join(f"`{s}:{i}`" for s, i in ks))
        lines.append("")
    if pair_counts:
        lines += ["## Cross-source near-duplicate pair counts", "",
                  "| source A | source B | near-dup pairs |", "|---|---|---|"]
        for (a, b), c in pair_counts.most_common():
            lines.append(f"| {a} | {b} | {c} |")
        lines.append("")
        lines += ["Sample pairs:", ""]
        for (sa, ia), (sb, ib) in cross_near[:15]:
            lines.append(f"- `{sa}:{ia}`  ≈  `{sb}:{ib}`")
        lines.append("")
    if within_near:
        lines += ["## Within-source near-duplicates (template/render reuse — expected for synthetic "
                  "and chart sets)", "",
                  "| source | near-dup pairs |", "|---|---|"]
        for s, c in within_near.most_common():
            lines.append(f"| {s} | {c} |")
        lines.append("")
    if not exact_groups and not near_pairs:
        lines.append("_No duplicates found at this threshold._")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] exact groups={len(exact_groups)} (cross-source {len(exact_cross)}), "
          f"near pairs={len(near_pairs)} (cross-source {len(cross_near)}) -> {out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort
