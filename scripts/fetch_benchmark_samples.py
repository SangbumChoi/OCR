#!/usr/bin/env python3
"""Download ONE representative sample (image + ground-truth + metric + PURPOSE) for every
benchmark in configs/benchmark_catalog.yaml, so the whole suite is inspectable at a glance.

For each benchmark with an `hf_id` it writes:
    data/benchmarks/<key>/sample.png     # the image (if the record has one)
    data/benchmarks/<key>/sample.json    # GT + metric + PURPOSE ("what it measures") + source

Catalog entries without an `hf_id` are documented-only (not cleanly streamable from HF); they
are listed in data/benchmarks/README.md and, where relevant, realised by
scripts/make_synthetic_samples.py.

Uses HF `datasets` streaming (one example, no full download). Failures are skipped + reported.

    python scripts/fetch_benchmark_samples.py
    python scripts/fetch_benchmark_samples.py --only docvqa chartqa
    python scripts/fetch_benchmark_samples.py --refresh-meta   # update purpose/labels, no re-download
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks"
CATALOG = ROOT / "configs" / "benchmark_catalog.yaml"


def load_catalog() -> list[dict]:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["benchmarks"]


def _find_image(ex: dict):
    from PIL import Image

    if "image" in ex and isinstance(ex["image"], Image.Image):
        return ex["image"]
    for v in ex.values():
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, list) and v and isinstance(v[0], Image.Image):
            return v[0]
    return None


def _json_safe(ex: dict) -> dict:
    out = {}
    for k, v in ex.items():
        try:
            json.dumps(v)
            out[k] = v if not isinstance(v, str) else v[:2000]
        except (TypeError, ValueError):
            out[k] = f"<{type(v).__name__}>"
    return out


def _meta(e: dict, ground_truth: dict | None = None) -> dict:
    label = {
        "benchmark": e["key"],
        "name": e.get("name", e["key"]),
        "category": e.get("category", "-"),
        "metric": e.get("metric", "-"),
        "purpose": e.get("purpose", "-"),
        "hf_id": e.get("hf_id"),
        "config": e.get("config"),
        "split": e.get("split"),
        "source": e.get("source", "-"),
    }
    if ground_truth is not None:
        label["ground_truth"] = ground_truth
    return label


def fetch_one(e: dict, force: bool = False, refresh_meta: bool = False) -> str:
    key = e["key"]
    if not e.get("hf_id"):
        return "documented"  # no HF source; handled by catalog / synthetic generator
    folder = OUT / key
    img_path = folder / "sample.png"
    json_path = folder / "sample.json"

    # refresh metadata (purpose/category/...) without re-downloading
    if img_path.exists() and not force:
        if json_path.exists():
            try:
                old = json.loads(json_path.read_text(encoding="utf-8"))
                gt = old.get("ground_truth")
            except Exception:
                gt = None
        else:
            gt = None
        json_path.write_text(json.dumps(_meta(e, gt), indent=2, ensure_ascii=False), encoding="utf-8")
        return "refreshed" if refresh_meta else "skip"

    from datasets import load_dataset

    try:
        ds = load_dataset(e["hf_id"], e.get("config"), split=e["split"], streaming=True)
        ex = dict(next(iter(ds)))
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:120]}")
        return "fail"

    img = _find_image(ex)
    folder.mkdir(parents=True, exist_ok=True)
    if img is not None:
        img.convert("RGB").save(img_path)
    json_path.write_text(json.dumps(_meta(e, _json_safe(ex)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok]   {key}: image={'yes' if img is not None else 'NONE'} -> {folder}")
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", help="subset of benchmark keys")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--refresh-meta", action="store_true", help="rewrite purpose/labels without re-download")
    args = p.parse_args()

    entries = [e for e in load_catalog() if not args.only or e["key"] in args.only]
    stats: dict[str, int] = {}
    for e in entries:
        r = fetch_one(e, force=args.force, refresh_meta=args.refresh_meta)
        stats[r] = stats.get(r, 0) + 1
    print(f"\n[done] {stats} over {len(entries)} catalog entries -> {OUT}")
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort


if __name__ == "__main__":
    main()
