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
_SAMPLING_STRATEGIES = {"global_hash", "task_stratified"}
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
    sampling_strategy: str = "global_hash"
    min_rows_per_task: int = 0

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
        if self.sampling_strategy not in _SAMPLING_STRATEGIES:
            raise ValueError(
                "Hub component sampling_strategy must be one of "
                f"{sorted(_SAMPLING_STRATEGIES)}"
            )
        if self.min_rows_per_task < 0:
            raise ValueError("Hub component min_rows_per_task must be non-negative")
        if self.sampling_strategy == "global_hash" and self.min_rows_per_task:
            raise ValueError(
                "min_rows_per_task requires sampling_strategy='task_stratified'"
            )
        if (
            self.sampling_strategy == "task_stratified"
            and self.min_rows_per_task <= 0
        ):
            raise ValueError(
                "task_stratified sampling requires a positive min_rows_per_task"
            )


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


def _allocate_task_quotas(
    counts: dict[str, int],
    *,
    max_rows: int,
    minimum: int,
) -> dict[str, int]:
    tasks = sorted(counts)
    required = sum(min(minimum, counts[task]) for task in tasks)
    if required > max_rows:
        raise ValueError(
            "max_rows cannot satisfy min_rows_per_task across the filtered "
            f"tasks: required={required}, max_rows={max_rows}"
        )
    quotas = {task: min(minimum, counts[task]) for task in tasks}
    remaining = max_rows - required
    while remaining:
        capacities = {
            task: counts[task] - quotas[task]
            for task in tasks
            if counts[task] > quotas[task]
        }
        if not capacities:
            break
        total_capacity = sum(capacities.values())
        shares = {
            task: remaining * capacity / total_capacity
            for task, capacity in capacities.items()
        }
        allocated = 0
        for task in sorted(capacities):
            addition = min(capacities[task], int(shares[task]))
            quotas[task] += addition
            allocated += addition
        remaining -= allocated
        if not remaining:
            break
        remainders = sorted(
            capacities,
            key=lambda task: (
                -(shares[task] - int(shares[task])),
                task,
            ),
        )
        for task in remainders:
            if remaining <= 0:
                break
            if quotas[task] < counts[task]:
                quotas[task] += 1
                remaining -= 1
    return quotas


def _select_indices(
    metadata: Any,
    eligible_indices: list[int],
    spec: HubComponentSpec,
) -> tuple[list[int], dict[str, Any]]:
    eligible_tasks = Counter(
        str(metadata[index].get("task") or "unknown")
        for index in eligible_indices
    )
    if spec.max_rows is None or len(eligible_indices) <= spec.max_rows:
        selected_indices = list(eligible_indices)
        quotas = dict(sorted(eligible_tasks.items()))
        applied_strategy = "all_filtered"
    elif spec.sampling_strategy == "global_hash":
        selected_indices = sorted(
            eligible_indices,
            key=lambda index: _selection_key(metadata, index, spec.seed),
        )[: spec.max_rows]
        quotas = {}
        applied_strategy = spec.sampling_strategy
    else:
        quotas = _allocate_task_quotas(
            dict(eligible_tasks),
            max_rows=spec.max_rows,
            minimum=spec.min_rows_per_task,
        )
        selected_indices = []
        for task in sorted(quotas):
            task_indices = [
                index
                for index in eligible_indices
                if str(metadata[index].get("task") or "unknown") == task
            ]
            task_indices.sort(
                key=lambda index: _selection_key(metadata, index, spec.seed)
            )
            selected_indices.extend(task_indices[: quotas[task]])
        applied_strategy = spec.sampling_strategy
    selected_indices.sort()
    selected_tasks = Counter(
        str(metadata[index].get("task") or "unknown")
        for index in selected_indices
    )
    return selected_indices, {
        "applied_strategy": applied_strategy,
        "eligible_rows": len(eligible_indices),
        "eligible_task_counts": dict(sorted(eligible_tasks.items())),
        "task_quotas": dict(sorted(quotas.items())),
        "selected_rows": len(selected_indices),
        "selected_task_counts": dict(sorted(selected_tasks.items())),
        "task_floor_satisfied": all(
            selected_tasks[task] >= min(spec.min_rows_per_task, count)
            for task, count in eligible_tasks.items()
        ),
    }


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
    eligible_indices = [
        index for index in range(len(metadata)) if _row_matches(metadata[index], spec)
    ]
    if not eligible_indices:
        raise ValueError("public UDD filters selected zero rows")
    selected_indices, selection = _select_indices(
        metadata,
        eligible_indices,
        spec,
    )
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
        "selection": selection,
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
