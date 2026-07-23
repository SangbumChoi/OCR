"""Reproducible acquisition and validation of public UDD training components."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_COLUMNS = {
    "image",
    "sample_id",
    "source",
    "task",
    "instructions",
    "answers",
    "language",
    "metric",
    "fold",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class HubComponentSpec:
    """Immutable Hub source plus deterministic row-selection controls."""

    repo_id: str
    revision: str
    split: str = "train"
    config_name: str | None = None
    fold: str | None = "train"
    sources: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    max_rows: int | None = None
    seed: int = 7
    decode_checks: int = 16

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError("Hub dataset repo_id must use namespace/name")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError("Hub dataset revision must be an immutable 40-character commit SHA")
        if not self.split:
            raise ValueError("Hub dataset split must be non-empty")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("Hub component max_rows must be positive or null")
        if self.seed < 0:
            raise ValueError("Hub component seed must be non-negative")
        if self.decode_checks <= 0:
            raise ValueError("Hub component decode_checks must be positive")


def _row_matches(row: dict[str, Any], spec: HubComponentSpec) -> bool:
    return bool(
        (spec.fold is None or str(row.get("fold") or "") == spec.fold)
        and (not spec.sources or str(row.get("source") or "") in spec.sources)
        and (not spec.tasks or str(row.get("task") or "") in spec.tasks)
        and (not spec.languages or str(row.get("language") or "") in spec.languages)
    )


def _selection_key(dataset: Any, index: int, seed: int) -> str:
    sample_id = str(dataset[index].get("sample_id") or f"row-{index}")
    return hashlib.sha256(f"{seed}:{sample_id}:{index}".encode("utf-8")).hexdigest()


def _validate_selected(dataset: Any, spec: HubComponentSpec) -> dict[str, Any]:
    from docvlm_eval.unified.hf import validate_payload_shapes

    try:
        validate_payload_shapes(dataset)
    except (AssertionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"public UDD payload validation failed: {exc}") from exc

    sample_ids: set[str] = set()
    image_keys: set[tuple[str, int, int]] = set()
    sources: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    qa_count = 0
    metadata_columns = [
        column
        for column in (
            "sample_id",
            "source",
            "task",
            "instructions",
            "answers",
            "language",
            "metric",
            "fold",
            "phash",
            "image_width",
            "image_height",
            "license",
        )
        if column in dataset.column_names
    ]
    metadata = dataset.select_columns(metadata_columns)
    for index in range(len(metadata)):
        row = metadata[index]
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"public UDD row {index} has an empty sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"public UDD contains duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if spec.fold is not None and str(row.get("fold") or "") != spec.fold:
            raise ValueError(f"public UDD selection contains a non-{spec.fold} fold row")
        answers = list(row.get("answers") or [])
        if any(not [str(value).strip() for value in variants if str(value).strip()] for variants in answers):
            raise ValueError(f"public UDD row {sample_id!r} contains an empty answer set")
        qa_count += len(answers)
        sources[str(row.get("source") or "unknown")] += 1
        tasks[str(row.get("task") or "unknown")] += 1
        languages[str(row.get("language") or "und")] += 1
        licenses[str(row.get("license") or "unspecified")] += 1
        phash = str(row.get("phash") or "")
        if phash:
            image_key = (
                phash,
                int(row.get("image_width") or 0),
                int(row.get("image_height") or 0),
            )
            if image_key in image_keys:
                raise ValueError(
                    f"public UDD contains duplicate phash/dimensions at {sample_id!r}"
                )
            image_keys.add(image_key)

    checks = min(spec.decode_checks, len(dataset))
    ranked = sorted(range(len(dataset)), key=lambda index: _selection_key(metadata, index, spec.seed))
    for index in ranked[:checks]:
        row = dataset[index]
        image = row["image"]
        size = getattr(image, "size", None)
        if not size or min(size) <= 0:
            raise ValueError(f"public UDD image failed to decode at row {index}")
        if "image_width" in dataset.column_names and int(row.get("image_width") or 0) != size[0]:
            raise ValueError(f"public UDD image_width mismatch at row {index}")
        if "image_height" in dataset.column_names and int(row.get("image_height") or 0) != size[1]:
            raise ValueError(f"public UDD image_height mismatch at row {index}")

    return {
        "rows": len(dataset),
        "qas": qa_count,
        "decoded_images_checked": checks,
        "unique_sample_ids": len(sample_ids),
        "unique_phash_dimensions": len(image_keys),
        "sources": dict(sorted(sources.items())),
        "tasks": dict(sorted(tasks.items())),
        "languages": dict(sorted(languages.items())),
        "licenses": dict(sorted(licenses.items())),
    }


def materialize_component(
    dataset: Any,
    output_dir: str | Path,
    spec: HubComponentSpec,
    *,
    resolved_revision: str,
) -> dict[str, Any]:
    """Filter, hash-sample, validate, and persist one public UDD component."""

    if resolved_revision != spec.revision:
        raise ValueError(
            f"Hub resolved revision {resolved_revision!r} does not match pinned "
            f"{spec.revision!r}"
        )
    missing = _REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"public UDD is missing required columns: {sorted(missing)}")
    metadata = dataset.select_columns(
        [
            column
            for column in ("sample_id", "source", "task", "language", "fold")
            if column in dataset.column_names
        ]
    )
    selected_indices = [
        index for index in range(len(metadata)) if _row_matches(metadata[index], spec)
    ]
    if not selected_indices:
        raise ValueError("public UDD filters selected zero rows")
    if spec.max_rows is not None and len(selected_indices) > spec.max_rows:
        selected_indices = sorted(
            selected_indices,
            key=lambda index: _selection_key(metadata, index, spec.seed),
        )[: spec.max_rows]
        selected_indices.sort()
    selected = dataset.select(selected_indices)
    validation = _validate_selected(selected, spec)

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty component output {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    selected.save_to_disk(str(output_dir))
    manifest = {
        "schema_version": 1,
        "kind": "huggingface_dataset",
        "spec": asdict(spec),
        "resolved_revision": resolved_revision,
        "source_rows": len(dataset),
        "selected_indices_fingerprint": _fingerprint(selected_indices),
        "dataset_fingerprint": getattr(selected, "_fingerprint", None),
        "validation": validation,
    }
    _atomic_write_json(output_dir / "component_manifest.json", manifest)
    return manifest


def acquire_hub_component(
    spec: HubComponentSpec,
    output_dir: str | Path,
    *,
    token: str | None = None,
    dataset_loader: Callable[..., Any] | None = None,
    hub_api: Any | None = None,
) -> dict[str, Any]:
    """Resolve a pinned Hub revision, download its split, and materialize the component."""

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi(token=token)
    info = hub_api.dataset_info(spec.repo_id, revision=spec.revision)
    resolved_revision = str(info.sha)
    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "revision": spec.revision,
        "token": token,
    }
    if spec.config_name is not None:
        kwargs["name"] = spec.config_name
    dataset = dataset_loader(**kwargs)
    return materialize_component(
        dataset,
        output_dir,
        spec,
        resolved_revision=resolved_revision,
    )
