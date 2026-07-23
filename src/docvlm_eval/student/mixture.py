"""Materialize heterogeneous UDD components for weighted student pretraining."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_COMPONENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class MixtureComponent:
    """One on-disk UDD dataset and its target sampling probability."""

    name: str
    path: str
    weight: float
    fold: str | None = None

    def __post_init__(self) -> None:
        if not _COMPONENT_NAME.fullmatch(self.name):
            raise ValueError(
                "mixture component names must match [A-Za-z0-9][A-Za-z0-9_.-]*"
            )
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("mixture component weights must be finite and positive")


def validate_components(components: Iterable[MixtureComponent]) -> tuple[MixtureComponent, ...]:
    """Validate uniqueness and normalize weights to an exact probability distribution."""

    components = tuple(components)
    if not components:
        raise ValueError("at least one mixture component is required")
    names = [component.name for component in components]
    if len(set(names)) != len(names):
        raise ValueError("mixture component names must be unique")
    total = sum(component.weight for component in components)
    return tuple(
        MixtureComponent(
            component.name,
            component.path,
            component.weight / total,
            component.fold,
        )
        for component in components
    )


def mixture_features():
    """Canonical UDD superset consumed by ``UDDStudentDataset``."""

    from datasets import Features, Image, Sequence, Value

    return Features(
        {
            "image": Image(),
            "sample_id": Value("string"),
            "source": Value("string"),
            "task": Value("string"),
            "instructions": Sequence(Value("string")),
            "answers": Sequence(Sequence(Value("string"))),
            "elements_json": Value("string"),
            "fields_json": Value("string"),
            "regions_json": Value("string"),
            "full_text": Value("string"),
            "table_html": Value("string"),
            "language": Value("string"),
            "metric": Value("string"),
            "fold": Value("string"),
            "mixture_component": Value("string"),
            "teacher_answers": Sequence(Value("string")),
            "teacher_scores": Sequence(Value("float32")),
            "teacher_provenance_json": Value("string"),
        }
    )


_STRING_DEFAULTS = {
    "sample_id": "",
    "source": "",
    "task": "",
    "elements_json": "",
    "fields_json": "",
    "regions_json": "",
    "full_text": "",
    "table_html": "",
    "language": "",
    "metric": "anls",
    "fold": "",
    "teacher_provenance_json": "{}",
}


def _normalize_dataset(dataset: Any, component: MixtureComponent):
    columns = set(dataset.column_names)
    required = {"image", "instructions", "answers"}
    missing = required - columns
    if missing:
        raise ValueError(
            f"component {component.name!r} is missing required columns: {sorted(missing)}"
        )
    if component.fold is not None:
        if "fold" not in columns:
            raise ValueError(
                f"component {component.name!r} requested fold {component.fold!r}, "
                "but the dataset has no fold column"
            )
        dataset = dataset.filter(
            lambda row: row["fold"] == component.fold,
            desc=f"mixture:{component.name}:{component.fold}",
        )
    if len(dataset) == 0:
        raise ValueError(f"component {component.name!r} has no selected rows")

    for name, default in _STRING_DEFAULTS.items():
        if name not in dataset.column_names:
            dataset = dataset.add_column(name, [default] * len(dataset))
    for name in ("teacher_answers", "teacher_scores"):
        if name not in dataset.column_names:
            dataset = dataset.add_column(name, [[] for _ in range(len(dataset))])
    if "fold" in dataset.column_names:
        dataset = dataset.remove_columns(["fold"])
    dataset = dataset.add_column("fold", ["train"] * len(dataset))
    if "mixture_component" in dataset.column_names:
        dataset = dataset.remove_columns(["mixture_component"])
    dataset = dataset.add_column("mixture_component", [component.name] * len(dataset))
    canonical = list(mixture_features())
    dataset = dataset.select_columns(canonical)
    return dataset.cast(mixture_features())


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


def _manifest_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_weighted_mixture(
    components: Iterable[MixtureComponent],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Normalize and concatenate components without duplicating rows.

    Sampling weights are persisted in ``mixture_manifest.json``. The training sampler applies them
    at runtime, avoiding materialized duplicate rows and allowing weights to change independently
    of the underlying Arrow data.
    """

    from datasets import concatenate_datasets, load_from_disk

    components = validate_components(components)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty mixture output {output_dir}")
    normalized = []
    records = []
    for component in components:
        source = load_from_disk(component.path)
        dataset = _normalize_dataset(source, component)
        normalized.append(dataset)
        upstream_path = Path(component.path) / "component_manifest.json"
        upstream_manifest = (
            json.loads(upstream_path.read_text(encoding="utf-8"))
            if upstream_path.is_file()
            else None
        )
        records.append(
            {
                **asdict(component),
                "path": str(Path(component.path).resolve()),
                "rows": len(dataset),
                "source_fingerprint": getattr(source, "_fingerprint", None),
                "normalized_fingerprint": getattr(dataset, "_fingerprint", None),
                "upstream_manifest_fingerprint": (
                    _manifest_fingerprint(upstream_manifest)
                    if upstream_manifest is not None
                    else None
                ),
            }
        )
    combined = concatenate_datasets(normalized)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(output_dir))
    manifest = {
        "schema_version": 1,
        "rows": len(combined),
        "components": records,
        "weights": {component.name: component.weight for component in components},
        "dataset_fingerprint": getattr(combined, "_fingerprint", None),
        "columns": combined.column_names,
    }
    _atomic_write_json(output_dir / "mixture_manifest.json", manifest)
    return manifest
