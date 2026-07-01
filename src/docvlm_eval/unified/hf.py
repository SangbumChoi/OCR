"""UDD — the **Universal Document Dataset** HuggingFace layer.

Turns :class:`~docvlm_eval.unified.core.UnifiedSample` records into a HuggingFace ``datasets.Dataset``
with **one uniform schema for every task** (recognition / kie / vqa / localization / table /
reasoning), then shards + uploads it to the Hub. The structured payload (KIE fields, localization
regions with boxes) is JSON-encoded into string columns so a single schema covers all datasets while
losing nothing — decode `fields_json` / `regions_json` to recover the typed payload.

Workflow (see ``scripts/build_udd.py``):
  1. per-dataset **converter**: UnifiedLoader.load(key) → list[UnifiedSample]
  2. ``to_hf_dataset(rows)`` → Dataset (image cast to the HF Image feature)
  3. ``safety_check(rows)`` → build + save + reload + verify nothing was lost (run BEFORE upload)
  4. ``push(ds, repo, config_name=key, ...)`` → sharded upload (one config per benchmark + an "all")
"""

from __future__ import annotations

import json
from typing import Any

from .core import UnifiedSample


def udd_features():
    """The single, uniform UDD schema (every task fits it; structured payload is JSON-encoded)."""
    from datasets import Features, Image, Sequence, Value
    return Features({
        "image": Image(),
        "sample_id": Value("string"),
        "source": Value("string"),          # benchmark key
        "task": Value("string"),            # recognition/kie/vqa/localization/table/reasoning
        "instruction": Value("string"),
        "answers": Sequence(Value("string")),
        "fields_json": Value("string"),     # json [{key,value,bbox:[x1,y1,x2,y2,normalized]|null}]
        "regions_json": Value("string"),    # json [{label,text,bbox:[...]|null}]
        "full_text": Value("string"),
        "table_html": Value("string"),
        "language": Value("string"),
        "metric": Value("string"),
        # provenance / origin
        "hf_id": Value("string"),
        "split": Value("string"),
        "hf_config": Value("string"),
    })


def _row_to_record(r: UnifiedSample) -> dict[str, Any]:
    d = r.to_dict()                         # bbox already flattened to [x1,y1,x2,y2,normalized]
    return {
        "image": r.image_path,              # path -> cast to Image() reads the bytes
        "sample_id": r.sample_id,
        "source": r.source,
        "task": r.task,
        "instruction": r.prompt(),
        "answers": [str(a) for a in r.answers],
        "fields_json": json.dumps(d["fields"], ensure_ascii=False),
        "regions_json": json.dumps(d["regions"], ensure_ascii=False),
        "full_text": r.full_text or "",
        "table_html": r.table_html or "",
        "language": r.language or "",
        "metric": r.metric or "anls",
        "hf_id": r.hf_id or "",
        "split": r.split or "",
        "hf_config": r.hf_config or "",
    }


def to_hf_dataset(rows: list[UnifiedSample]):
    """Build a ``datasets.Dataset`` (UDD schema) from unified rows that have a cached ``image_path``."""
    from datasets import Dataset
    recs = [_row_to_record(r) for r in rows if r.image_path]
    if not recs:
        raise ValueError("no rows with a cached image_path — load with cache_dir set")
    # cast the FULL uniform schema (not just image) so sources whose optional columns are all-empty
    # — e.g. localization has no `answers` — still get List(string), keeping every source's features
    # identical for a clean cross-dataset concat.
    ds = Dataset.from_list(recs)
    return ds.cast(udd_features())


def safety_check(rows: list[UnifiedSample], workdir: str) -> dict:
    """Build → save_to_disk → reload → verify nothing was lost. Returns a report; raises on mismatch.

    Confirms: row count preserved, the image decodes, and the JSON-encoded structured payload
    round-trips (field/region counts survive). Run this BEFORE any upload."""
    from pathlib import Path

    from datasets import load_from_disk

    ds = to_hf_dataset(rows)
    src = [r for r in rows if r.image_path]
    out = Path(workdir); out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    ds2 = load_from_disk(str(out))

    assert len(ds2) == len(src), f"row count changed: {len(src)} -> {len(ds2)}"
    img0 = ds2[0]["image"]
    assert getattr(img0, "size", None), "image did not decode after round-trip"
    # structured payload survived
    exp_fields = sum(len(r.fields) for r in src)
    got_fields = sum(len(json.loads(x or "[]")) for x in ds2["fields_json"])
    assert exp_fields == got_fields, f"fields lost: {exp_fields} -> {got_fields}"
    exp_reg = sum(len(r.regions) for r in src)
    got_reg = sum(len(json.loads(x or "[]")) for x in ds2["regions_json"])
    assert exp_reg == got_reg, f"regions lost: {exp_reg} -> {got_reg}"
    return {"rows": len(ds2), "fields": got_fields, "regions": got_reg,
            "image_ok": True, "columns": ds2.column_names}


def push(ds, repo: str, *, config_name: str | None = None, token: str | None = None,
         private: bool = True, max_shard_size: str = "500MB") -> str:
    """Push a UDD Dataset to the Hub as ``config_name`` (sharded). Returns the repo URL."""
    ds.push_to_hub(repo, config_name=config_name or "default", token=token,
                   private=private, max_shard_size=max_shard_size)
    return f"https://huggingface.co/datasets/{repo}"
