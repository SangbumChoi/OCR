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
    coverage_languages: tuple[str, ...] = ()
    min_rows_per_language: int = 0

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
        if len(set(self.coverage_languages)) != len(
            self.coverage_languages
        ) or any(not language for language in self.coverage_languages):
            raise ValueError(
                "coverage_languages must contain unique non-empty values"
            )
        if self.min_rows_per_language < 0:
            raise ValueError(
                "Hub component min_rows_per_language must be non-negative"
            )
        if bool(self.coverage_languages) != bool(
            self.min_rows_per_language
        ):
            raise ValueError(
                "coverage_languages and min_rows_per_language must be set "
                "together"
            )
        if (
            self.coverage_languages
            and self.sampling_strategy != "task_stratified"
        ):
            raise ValueError(
                "language coverage requires sampling_strategy='task_stratified'"
            )
        if self.languages and not set(self.coverage_languages).issubset(
            self.languages
        ):
            raise ValueError(
                "coverage_languages must be included by the language filters"
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


def _allocate_language_task_reservations(
    intersection_counts: dict[str, dict[str, int]],
    *,
    task_quotas: dict[str, int],
    languages: tuple[str, ...],
    minimum: int,
) -> dict[str, dict[str, int]]:
    intersections = {
        language: {
            task: int(count)
            for task, count in intersection_counts.get(language, {}).items()
            if int(count) > 0
        }
        for language in languages
    }
    for language, by_task in intersections.items():
        available = sum(by_task.values())
        if available < minimum:
            raise ValueError(
                "filtered rows cannot satisfy min_rows_per_language for "
                f"{language!r}: available={available}, required={minimum}"
            )

    source = ("source", "")
    sink = ("sink", "")
    residual: dict[tuple[str, str], dict[tuple[str, str], int]] = {}

    def add_edge(
        start: tuple[str, str],
        end: tuple[str, str],
        capacity: int,
    ) -> None:
        residual.setdefault(start, {})[end] = capacity
        residual.setdefault(end, {}).setdefault(start, 0)

    language_nodes = {
        language: ("language", language) for language in sorted(languages)
    }
    task_nodes = {
        task: ("task", task) for task in sorted(task_quotas)
    }
    for language, node in language_nodes.items():
        add_edge(source, node, minimum)
        for task, count in sorted(intersections[language].items()):
            if task in task_nodes:
                add_edge(node, task_nodes[task], count)
    for task, node in task_nodes.items():
        add_edge(node, sink, task_quotas[task])

    required_flow = minimum * len(languages)
    total_flow = 0
    while total_flow < required_flow:
        parent: dict[tuple[str, str], tuple[str, str] | None] = {
            source: None
        }
        queue = [source]
        for node in queue:
            for neighbor in sorted(residual[node]):
                if residual[node][neighbor] > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if sink not in parent:
            break
        increment = required_flow - total_flow
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            increment = min(increment, residual[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= increment
            residual[node][previous] += increment
            node = previous
        total_flow += increment
    if total_flow != required_flow:
        raise ValueError(
            "task quotas cannot jointly satisfy the requested language floors: "
            f"required={required_flow}, feasible={total_flow}"
        )

    reservation_counts: dict[str, dict[str, int]] = {}
    for language, language_node in language_nodes.items():
        for task, task_node in task_nodes.items():
            capacity = intersections[language].get(task, 0)
            count = capacity - residual.get(language_node, {}).get(
                task_node,
                0,
            )
            if count <= 0:
                continue
            reservation_counts.setdefault(language, {})[task] = count
    return reservation_counts


def plan_stratified_coverage(
    task_counts: dict[str, int],
    task_language_counts: dict[str, dict[str, int]],
    *,
    max_rows: int,
    min_rows_per_task: int,
    coverage_languages: tuple[str, ...],
    min_rows_per_language: int,
) -> dict[str, Any]:
    """Plan deterministic task quotas and feasible language reservations."""

    task_quotas = _allocate_task_quotas(
        task_counts,
        max_rows=max_rows,
        minimum=min_rows_per_task,
    )
    language_task_reservations = _allocate_language_task_reservations(
        task_language_counts,
        task_quotas=task_quotas,
        languages=coverage_languages,
        minimum=min_rows_per_language,
    )
    return {
        "max_rows": max_rows,
        "min_rows_per_task": min_rows_per_task,
        "coverage_languages": list(coverage_languages),
        "min_rows_per_language": min_rows_per_language,
        "task_quotas": dict(sorted(task_quotas.items())),
        "language_task_reservations": language_task_reservations,
        "feasible": True,
    }


def _reserve_language_rows(
    metadata: Any,
    eligible_indices: list[int],
    *,
    task_quotas: dict[str, int],
    languages: tuple[str, ...],
    minimum: int,
    seed: int,
) -> tuple[set[int], dict[str, dict[str, int]]]:
    intersections: dict[str, dict[str, list[int]]] = {
        language: {} for language in languages
    }
    for index in eligible_indices:
        row = metadata[index]
        language = str(row.get("language") or "und")
        if language not in intersections:
            continue
        task = str(row.get("task") or "unknown")
        intersections[language].setdefault(task, []).append(index)
    reservation_counts = _allocate_language_task_reservations(
        {
            language: {
                task: len(indices)
                for task, indices in by_task.items()
            }
            for language, by_task in intersections.items()
        },
        task_quotas=task_quotas,
        languages=languages,
        minimum=minimum,
    )
    reserved: set[int] = set()
    for language, by_task in reservation_counts.items():
        for task, count in by_task.items():
            ranked = sorted(
                intersections[language][task],
                key=lambda index: _selection_key(metadata, index, seed),
            )
            reserved.update(ranked[:count])
    return reserved, reservation_counts


def _select_indices(
    metadata: Any,
    eligible_indices: list[int],
    spec: HubComponentSpec,
) -> tuple[list[int], dict[str, Any]]:
    eligible_tasks = Counter(
        str(metadata[index].get("task") or "unknown")
        for index in eligible_indices
    )
    eligible_languages = Counter(
        str(metadata[index].get("language") or "und")
        for index in eligible_indices
    )
    reservations: dict[str, dict[str, int]] = {}
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
        reserved, reservations = _reserve_language_rows(
            metadata,
            eligible_indices,
            task_quotas=quotas,
            languages=spec.coverage_languages,
            minimum=spec.min_rows_per_language,
            seed=spec.seed,
        )
        selected_indices = list(reserved)
        selected_by_task = Counter(
            str(metadata[index].get("task") or "unknown")
            for index in reserved
        )
        for task in sorted(quotas):
            task_indices = [
                index
                for index in eligible_indices
                if index not in reserved
                and str(metadata[index].get("task") or "unknown") == task
            ]
            task_indices.sort(
                key=lambda index: _selection_key(metadata, index, spec.seed)
            )
            selected_indices.extend(
                task_indices[: quotas[task] - selected_by_task[task]]
            )
        applied_strategy = spec.sampling_strategy
    selected_indices.sort()
    selected_tasks = Counter(
        str(metadata[index].get("task") or "unknown")
        for index in selected_indices
    )
    selected_languages = Counter(
        str(metadata[index].get("language") or "und")
        for index in selected_indices
    )
    language_floor_satisfied = all(
        selected_languages[language] >= spec.min_rows_per_language
        for language in spec.coverage_languages
    )
    if not language_floor_satisfied:
        raise ValueError(
            "selected rows do not satisfy the language coverage floor"
        )
    return selected_indices, {
        "applied_strategy": applied_strategy,
        "eligible_rows": len(eligible_indices),
        "eligible_task_counts": dict(sorted(eligible_tasks.items())),
        "eligible_language_counts": dict(
            sorted(eligible_languages.items())
        ),
        "task_quotas": dict(sorted(quotas.items())),
        "language_task_reservations": reservations,
        "selected_rows": len(selected_indices),
        "selected_task_counts": dict(sorted(selected_tasks.items())),
        "selected_language_counts": dict(
            sorted(selected_languages.items())
        ),
        "task_floor_satisfied": all(
            selected_tasks[task] >= min(spec.min_rows_per_task, count)
            for task, count in eligible_tasks.items()
        ),
        "language_floor_satisfied": language_floor_satisfied,
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
