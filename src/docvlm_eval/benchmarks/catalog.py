"""Benchmark catalog access + compact source-example fetching.

The catalog (``configs/benchmark_catalog.yaml``) lists every benchmark across the 10
capability categories with its ``purpose``, metric and HF id. This module loads it and can
fetch source images and their annotation records per benchmark via HF streaming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_catalog_path() -> Path:
    """Locate configs/benchmark_catalog.yaml relative to cwd or the repo root."""
    cand = Path("configs/benchmark_catalog.yaml")
    if cand.exists():
        return cand
    # walk up from this file to find a repo root containing configs/
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / "configs" / "benchmark_catalog.yaml"
        if p.exists():
            return p
    return cand


def load_catalog(path: str | Path | None = None) -> list[dict]:
    import yaml

    path = Path(path) if path else default_catalog_path()
    return yaml.safe_load(path.read_text(encoding="utf-8"))["benchmarks"]


def find_image(ex: dict):
    """Return the first PIL image found in a record, or None."""
    from PIL import Image

    if "image" in ex and isinstance(ex["image"], Image.Image):
        return ex["image"]
    for v in ex.values():
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, list) and v and isinstance(v[0], Image.Image):
            return v[0]
    return None


def json_safe(ex: dict) -> dict:
    """Drop non-serialisable values (e.g. the image), truncate long strings."""
    out: dict[str, Any] = {}
    for k, v in ex.items():
        try:
            json.dumps(v)
            out[k] = v if not isinstance(v, str) else v[:2000]
        except (TypeError, ValueError):
            continue
    return out


def fetch_one(e: dict, out_dir: str | Path, force: bool = False) -> str:
    """Fetch a single sample for one catalog entry. Returns a status string."""
    key = e["key"]
    if not e.get("hf_id"):
        return "documented"
    folder = Path(out_dir) / key
    download_dir = folder / "download"
    img_path = download_dir / "00.png"
    annotation_path = folder / "annotations.jsonl"

    if annotation_path.exists() and not force:
        return "skip"

    from datasets import load_dataset

    try:
        ds = load_dataset(e["hf_id"], e.get("config"), split=e["split"], streaming=True)
        ex = dict(next(iter(ds)))
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:120]}")
        return "fail"

    img = find_image(ex)
    if img is None:
        print(f"[skip] {key}: source record has no image")
        return "no-image"
    download_dir.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(img_path)
    row = {"image": "download/00.png", "ground_truth": json_safe(ex)}
    annotation_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[ok]   {key}: downloaded one image -> {folder}")
    return "ok"


def _downscale(img, max_px: int = 1000):
    """Convert to RGB and shrink so the longest side <= max_px (keeps repo size sane)."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    return img


def fetch_many(e: dict, out_dir: str | Path, n: int = 10, force: bool = False,
               max_px: int = 1000, quality: int = 80) -> str:
    """Fetch up to ``n`` source examples into ``<key>/download/`` and ``annotations.jsonl``.

    Images are downscaled and JPEG-compressed to keep the checked-in examples compact. Each
    annotation line stores a relative image path and the raw source record.
    """
    key = e["key"]
    if not e.get("hf_id"):
        return "documented"
    folder = Path(out_dir) / key
    sdir = folder / "download"
    jsonl = folder / "annotations.jsonl"
    if jsonl.exists() and not force:
        return "skip"

    from datasets import load_dataset

    try:
        ds = load_dataset(e["hf_id"], e.get("config"), split=e["split"], streaming=True)
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:120]}")
        return "fail"

    sdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[str] = set()   # dedupe by image content (these sets often repeat an image per question)
    import hashlib
    try:
        for scanned, ex in enumerate(ds):
            if len(rows) >= n or scanned >= 1500:  # bound streaming cost
                break
            ex = dict(ex)
            img = find_image(ex)
            if img is None:
                continue
            small = _downscale(img, max_px)
            h = hashlib.md5(small.tobytes()).hexdigest()
            if h in seen:           # skip duplicate image (keep distinct images only)
                continue
            seen.add(h)
            fn = f"{len(rows):02d}.jpg"
            small.save(sdir / fn, quality=quality)
            rows.append({"image": f"download/{fn}", "ground_truth": json_safe(ex)})
    except Exception as exc:
        print(f"[warn] {key}: stopped after {len(rows)} ({type(exc).__name__})")

    if not rows:
        return "no-image"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[ok]   {key}: {len(rows)} source examples -> {sdir}")
    return "ok"
