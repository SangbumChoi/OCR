"""Benchmark catalog access + one-sample fetching.

The catalog (``configs/benchmark_catalog.yaml``) lists every benchmark across the 10
capability categories with its ``purpose``, metric and HF id. This module loads it and can
materialise a one-sample preview (image + GT + purpose) per benchmark via HF streaming.
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
            out[k] = f"<{type(v).__name__}>"
    return out


def meta(e: dict, ground_truth: dict | None = None) -> dict:
    """Build the sample.json payload (label + metric + purpose + source) for a catalog entry."""
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


def fetch_one(e: dict, out_dir: str | Path, force: bool = False, refresh_meta: bool = False) -> str:
    """Fetch a single sample for one catalog entry. Returns a status string."""
    key = e["key"]
    if not e.get("hf_id"):
        return "documented"
    folder = Path(out_dir) / key
    img_path = folder / "sample.png"
    json_path = folder / "sample.json"

    if img_path.exists() and not force:
        gt = None
        if json_path.exists():
            try:
                gt = json.loads(json_path.read_text(encoding="utf-8")).get("ground_truth")
            except Exception:
                gt = None
        json_path.write_text(json.dumps(meta(e, gt), indent=2, ensure_ascii=False), encoding="utf-8")
        return "refreshed" if refresh_meta else "skip"

    from datasets import load_dataset

    try:
        ds = load_dataset(e["hf_id"], e.get("config"), split=e["split"], streaming=True)
        ex = dict(next(iter(ds)))
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:120]}")
        return "fail"

    img = find_image(ex)
    folder.mkdir(parents=True, exist_ok=True)
    if img is not None:
        img.convert("RGB").save(img_path)
    json_path.write_text(json.dumps(meta(e, json_safe(ex)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok]   {key}: image={'yes' if img is not None else 'NONE'} -> {folder}")
    return "ok"
