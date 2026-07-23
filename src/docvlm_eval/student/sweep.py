"""Matched multi-run ablations for the native sub-1B document VLM."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..architecture import estimate_parameters, load_blueprint
from .experiment import ExperimentPlan, ExperimentRunner, build_experiment_plan


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DOCUMENTS = {"experiment", "blueprint"}
_RESERVED_EXPERIMENT_PATHS = {"/name", "/output_root", "/blueprint"}


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


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _pointer_parts(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise ValueError("JSON Patch paths must be non-root JSON pointers")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _list_index(token: str, length: int, *, allow_end: bool) -> int:
    if token == "-" and allow_end:
        return length
    if not token.isdigit():
        raise ValueError(f"invalid JSON Patch list index {token!r}")
    index = int(token)
    upper = length if allow_end else length - 1
    if not 0 <= index <= upper:
        raise ValueError(f"JSON Patch list index {index} is out of range")
    return index


def _pointer_parent(document: Any, pointer: str) -> tuple[Any, str]:
    parts = _pointer_parts(pointer)
    parent = document
    for token in parts[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                raise ValueError(f"JSON Patch path does not exist: {pointer}")
            parent = parent[token]
        elif isinstance(parent, list):
            parent = parent[_list_index(token, len(parent), allow_end=False)]
        else:
            raise ValueError(f"JSON Patch path crosses a scalar: {pointer}")
    return parent, parts[-1]


def apply_json_patch(document: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the add, replace, and remove subset of RFC 6902 to a copy."""

    result = copy.deepcopy(document)
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError(f"patch operation {index} must be a mapping")
        op = str(operation.get("op") or "")
        path = str(operation.get("path") or "")
        if op not in {"add", "replace", "remove"}:
            raise ValueError(f"patch operation {index} has unsupported op {op!r}")
        if op in {"add", "replace"} and "value" not in operation:
            raise ValueError(f"patch operation {index} requires value")
        parent, token = _pointer_parent(result, path)
        if isinstance(parent, dict):
            exists = token in parent
            if op in {"replace", "remove"} and not exists:
                raise ValueError(f"JSON Patch path does not exist: {path}")
            if op == "remove":
                del parent[token]
            else:
                parent[token] = copy.deepcopy(operation["value"])
        elif isinstance(parent, list):
            item_index = _list_index(token, len(parent), allow_end=op == "add")
            if op == "remove":
                parent.pop(item_index)
            elif op == "replace":
                parent[item_index] = copy.deepcopy(operation["value"])
            else:
                parent.insert(item_index, copy.deepcopy(operation["value"]))
        else:
            raise ValueError(f"JSON Patch path has scalar parent: {path}")
    return result


def _pointer_get(document: Any, pointer: str) -> Any:
    value = document
    for token in _pointer_parts(pointer):
        if isinstance(value, dict):
            if token not in value:
                raise ValueError(f"matched control path does not exist: {pointer}")
            value = value[token]
        elif isinstance(value, list):
            value = value[_list_index(token, len(value), allow_end=False)]
        else:
            raise ValueError(f"matched control path crosses a scalar: {pointer}")
    return value


def _patch_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"sweep.{key} must be a list")
    return value


def _validate_reserved_patches(operations: list[dict[str, Any]]) -> None:
    for operation in operations:
        path = str(operation.get("path") or "")
        if path in _RESERVED_EXPERIMENT_PATHS:
            raise ValueError(f"experiment patch path {path} is reserved by the sweep compiler")


@dataclass(frozen=True)
class MatchedControl:
    document: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.document}:{self.path}"


@dataclass(frozen=True)
class CompiledVariant:
    id: str
    hypothesis: str
    experiment_patches: tuple[dict[str, Any], ...]
    blueprint_patches: tuple[dict[str, Any], ...]
    experiment_path: str
    blueprint_path: str
    plan: ExperimentPlan
    parameters: dict[str, int]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "experiment_patches": list(self.experiment_patches),
            "blueprint_patches": list(self.blueprint_patches),
            "experiment_path": self.experiment_path,
            "blueprint_path": self.blueprint_path,
            "experiment_root": self.plan.root,
            "experiment_fingerprint": self.plan.fingerprint,
            "parameters": self.parameters,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class SweepPlan:
    name: str
    root: str
    baseline: str
    base_experiment: str
    controls: tuple[MatchedControl, ...]
    control_values: dict[str, Any]
    variants: tuple[CompiledVariant, ...]
    fingerprint: str
    raw_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "root": self.root,
            "baseline": self.baseline,
            "base_experiment": self.base_experiment,
            "matched_controls": [asdict(control) for control in self.controls],
            "control_values": self.control_values,
            "variants": [variant.to_dict() for variant in self.variants],
            "fingerprint": self.fingerprint,
        }


def _load_sweep(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("sweep root must be a mapping")
    return raw


def _parse_controls(raw: dict[str, Any]) -> tuple[MatchedControl, ...]:
    controls_raw = raw.get("matched_controls")
    if not isinstance(controls_raw, list) or not controls_raw:
        raise ValueError("sweep.matched_controls must be a non-empty list")
    controls = []
    seen = set()
    for item in controls_raw:
        if not isinstance(item, dict):
            raise ValueError("every matched control must be a mapping")
        control = MatchedControl(
            document=str(item.get("document") or ""),
            path=str(item.get("path") or ""),
        )
        if control.document not in _DOCUMENTS:
            raise ValueError("matched control document must be experiment or blueprint")
        _pointer_parts(control.path)
        if control.key in seen:
            raise ValueError(f"duplicate matched control {control.key}")
        seen.add(control.key)
        controls.append(control)
    return tuple(controls)


def compile_sweep_plan(
    sweep_path: str | Path,
    *,
    repo_root: str | Path,
    python: str,
    compile_root: str | Path | None = None,
) -> SweepPlan:
    """Compile a matched sweep into validated, independently resumable experiments."""

    repo = Path(repo_root).resolve()
    source = _resolve_path(repo, sweep_path)
    raw = _load_sweep(source)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("sweep schema_version must be 1")
    name = str(raw.get("name") or "")
    if not _NAME.fullmatch(name):
        raise ValueError("sweep name must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    root = _resolve_path(repo, raw.get("output_root") or f"outputs/sweeps/{name}")
    compiled_root = (
        _resolve_path(repo, compile_root) if compile_root is not None else root / "compiled"
    )
    base_experiment_path = _resolve_path(repo, raw.get("base_experiment") or "")
    if not base_experiment_path.is_file():
        raise ValueError(f"base experiment does not exist: {base_experiment_path}")
    with base_experiment_path.open(encoding="utf-8") as handle:
        base_experiment = yaml.safe_load(handle) or {}
    if not isinstance(base_experiment, dict):
        raise ValueError("base experiment root must be a mapping")
    base_blueprint_path = _resolve_path(repo, base_experiment.get("blueprint") or "")
    base_blueprint = load_blueprint(base_blueprint_path)

    shared_experiment = _patch_list(raw, "shared_experiment_patches")
    shared_blueprint = _patch_list(raw, "shared_blueprint_patches")
    _validate_reserved_patches(shared_experiment)
    controls = _parse_controls(raw)
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or len(variants_raw) < 2:
        raise ValueError("sweep.variants must contain at least two variants")
    baseline = str(raw.get("baseline") or "")
    ids: set[str] = set()
    compiled: list[CompiledVariant] = []
    control_values: dict[str, Any] | None = None

    for item in variants_raw:
        if not isinstance(item, dict):
            raise ValueError("every sweep variant must be a mapping")
        variant_id = str(item.get("id") or "")
        if not _NAME.fullmatch(variant_id) or variant_id in ids:
            raise ValueError("variant ids must be unique safe names")
        ids.add(variant_id)
        variant_experiment_patches = _patch_list(item, "experiment_patches")
        variant_blueprint_patches = _patch_list(item, "blueprint_patches")
        experiment_patches = [*shared_experiment, *variant_experiment_patches]
        blueprint_patches = [*shared_blueprint, *variant_blueprint_patches]
        _validate_reserved_patches(experiment_patches)
        if (
            variant_id != baseline
            and not variant_experiment_patches
            and not variant_blueprint_patches
        ):
            raise ValueError(f"non-baseline variant {variant_id!r} has no patches")

        experiment = apply_json_patch(base_experiment, experiment_patches)
        blueprint = apply_json_patch(base_blueprint, blueprint_patches)
        variant_dir = compiled_root / variant_id
        blueprint_path = variant_dir / "blueprint.yaml"
        experiment_path = variant_dir / "experiment.yaml"
        stable_blueprint_path = root / "compiled" / variant_id / "blueprint.yaml"
        experiment["name"] = f"{name}--{variant_id}"
        experiment["output_root"] = str(root / "runs" / variant_id)
        experiment["blueprint"] = str(blueprint_path)
        evaluation = experiment.setdefault("evaluation", {})
        evaluation["wandb_group"] = name
        evaluation["wandb_run"] = f"{name}--{variant_id}"
        tags = [str(tag) for tag in evaluation.get("wandb_tags") or []]
        evaluation["wandb_tags"] = list(
            dict.fromkeys([*tags, "native-student-sweep", f"variant:{variant_id}"])
        )
        _atomic_write_yaml(blueprint_path, blueprint)
        _atomic_write_yaml(experiment_path, experiment)

        plan = build_experiment_plan(experiment_path, repo_root=repo, python=python)
        canonical_experiment = copy.deepcopy(experiment)
        canonical_experiment["blueprint"] = str(stable_blueprint_path)
        variant_fingerprint = _fingerprint(
            {
                "sweep": name,
                "variant": variant_id,
                "experiment": canonical_experiment,
                "blueprint": blueprint,
            }
        )
        plan = replace(
            plan,
            raw_spec=canonical_experiment,
            fingerprint=variant_fingerprint,
        )
        documents = {"experiment": canonical_experiment, "blueprint": blueprint}
        values = {
            control.key: copy.deepcopy(_pointer_get(documents[control.document], control.path))
            for control in controls
        }
        if control_values is None:
            control_values = values
        elif values != control_values:
            mismatches = [
                key for key in sorted(values) if values[key] != control_values.get(key)
            ]
            raise ValueError(
                f"variant {variant_id!r} violates matched controls: {mismatches}"
            )
        compiled.append(
            CompiledVariant(
                id=variant_id,
                hypothesis=str(item.get("hypothesis") or ""),
                experiment_patches=tuple(copy.deepcopy(experiment_patches)),
                blueprint_patches=tuple(copy.deepcopy(blueprint_patches)),
                experiment_path=str(experiment_path),
                blueprint_path=str(blueprint_path),
                plan=plan,
                parameters=estimate_parameters(plan.resolved_blueprint),
                fingerprint=variant_fingerprint,
            )
        )
    if baseline not in ids:
        raise ValueError("sweep baseline must name one compiled variant")
    sweep_fingerprint = _fingerprint(
        {
            "spec": raw,
            "base_experiment": base_experiment,
            "base_blueprint": base_blueprint,
            "variants": [variant.fingerprint for variant in compiled],
        }
    )
    return SweepPlan(
        name=name,
        root=str(root),
        baseline=baseline,
        base_experiment=str(base_experiment_path),
        controls=controls,
        control_values=control_values or {},
        variants=tuple(compiled),
        fingerprint=sweep_fingerprint,
        raw_spec=raw,
    )


def _summary_metrics(comparison: dict[str, Any]) -> dict[str, float]:
    splits = comparison["splits"]
    train = splits["train"]
    heldout = splits["heldout"]
    gap = comparison["train_minus_heldout"]["headline"]
    return {
        "train_score": float(train["score"]),
        "heldout_score": float(heldout["score"]),
        "heldout_reward": float(heldout["reward"]),
        "heldout_valid_structure": float(heldout["valid_structure_fraction"]),
        "heldout_answer_rate": float(heldout["answer_rate"]),
        "train_minus_heldout_score": float(gap["score"]),
        "heldout_milliseconds_per_sample": float(heldout["milliseconds_per_sample"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _evaluation_set_fingerprint(run_root: Path, split: str) -> str:
    path = run_root / "artifacts" / "samples" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {split} evaluation samples: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = Path(str(row.pop("image_path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{path}:{line_number} references missing image {image_path}"
            )
        row["image_sha256"] = _file_sha256(image_path)
        rows.append(row)
    if not rows:
        raise ValueError(f"evaluation sample file is empty: {path}")
    rows.sort(key=lambda row: str(row.get("sample_id") or ""))
    return _fingerprint(rows)


def _axis_scores(comparison: dict[str, Any], split: str) -> dict[str, float]:
    axes = comparison["splits"][split].get("by_answer_type", {})
    return {name: float(values["score"]) for name, values in sorted(axes.items())}


def aggregate_sweep_results(plan: SweepPlan) -> dict[str, Any]:
    """Aggregate completed train/heldout comparisons and deltas against the baseline."""

    records: dict[str, dict[str, Any]] = {}
    artifact_controls: dict[str, str] | None = None
    for variant in plan.variants:
        run_root = Path(variant.plan.root)
        fingerprints = {
            split: _evaluation_set_fingerprint(run_root, split)
            for split in ("train", "heldout")
        }
        if artifact_controls is None:
            artifact_controls = fingerprints
        elif fingerprints != artifact_controls:
            mismatches = [
                split
                for split in sorted(fingerprints)
                if fingerprints[split] != artifact_controls.get(split)
            ]
            raise ValueError(
                f"variant {variant.id!r} has mismatched evaluation artifacts: {mismatches}"
            )
        comparison_path = (
            run_root / "artifacts" / "evaluation" / "comparison.json"
        )
        if not comparison_path.is_file():
            raise FileNotFoundError(
                f"variant {variant.id!r} has no evaluation comparison: {comparison_path}"
            )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        records[variant.id] = {
            "hypothesis": variant.hypothesis,
            "fingerprint": variant.fingerprint,
            "parameters": variant.parameters,
            "metrics": _summary_metrics(comparison),
            "heldout_by_answer_type": _axis_scores(comparison, "heldout"),
            "train_by_answer_type": _axis_scores(comparison, "train"),
            "comparison": str(comparison_path),
        }
    baseline = records[plan.baseline]
    baseline_metrics = baseline["metrics"]
    baseline_axes = baseline["heldout_by_answer_type"]
    for record in records.values():
        record["delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_metrics[name]), 8)
            for name, value in record["metrics"].items()
        }
        record["heldout_axis_delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_axes[name]), 8)
            for name, value in record["heldout_by_answer_type"].items()
            if name in baseline_axes
        }
    ranking = sorted(
        records,
        key=lambda variant_id: (
            -records[variant_id]["metrics"]["heldout_score"],
            records[variant_id]["metrics"]["train_minus_heldout_score"],
            variant_id,
        ),
    )
    result = {
        "schema_version": 1,
        "sweep": plan.name,
        "sweep_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "matched_controls": plan.control_values,
        "matched_evaluation_artifacts": artifact_controls or {},
        "ranking": ranking,
        "variants": records,
    }
    root = Path(plan.root)
    _atomic_write_json(root / "comparison.json", result)
    _write_comparison_markdown(root / "comparison.md", result)
    return result


def _write_comparison_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['sweep']} matched sweep",
        "",
        f"Baseline: `{result['baseline']}`",
        "",
        "| Rank | Variant | Parameters | Heldout score | Delta | Train-heldout gap | ms/sample |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, variant_id in enumerate(result["ranking"], start=1):
        record = result["variants"][variant_id]
        metrics = record["metrics"]
        delta = record["delta_vs_baseline"]["heldout_score"]
        lines.append(
            f"| {rank} | `{variant_id}` | {record['parameters']['total']:,} | "
            f"{metrics['heldout_score']:.6f} | {delta:+.6f} | "
            f"{metrics['train_minus_heldout_score']:+.6f} | "
            f"{metrics['heldout_milliseconds_per_sample']:.3f} |"
        )
    lines.extend(["", "## Matched controls", ""])
    for key, value in sorted(result["matched_controls"].items()):
        lines.append(f"- `{key}`: `{_stable_json(value)}`")
    _atomic_write_text(path, "\n".join(lines) + "\n")


class SweepRunner:
    """Run matched variants sequentially and publish a single comparison."""

    def __init__(self, plan: SweepPlan, *, repo_root: str | Path):
        self.plan = plan
        self.repo_root = Path(repo_root).resolve()

    def _write_plan(self) -> None:
        root = Path(self.plan.root)
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(root / "sweep_plan.json", self.plan.to_dict())
        _atomic_write_json(root / "sweep_spec.json", self.plan.raw_spec)

    def run(
        self,
        *,
        dry_run: bool = False,
        resume: bool = True,
        variant_ids: set[str] | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> dict[str, Any]:
        known = {variant.id for variant in self.plan.variants}
        selected = known if variant_ids is None else set(variant_ids)
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown sweep variants: {sorted(unknown)}")
        outcomes = []
        if dry_run:
            for variant in self.plan.variants:
                if variant.id not in selected:
                    continue
                result = ExperimentRunner(
                    variant.plan,
                    repo_root=self.repo_root,
                ).run(
                    dry_run=True,
                    resume=resume,
                    start=start,
                    stop=stop,
                )
                outcomes.append({"variant": variant.id, "result": result})
        response: dict[str, Any] = {
            "dry_run": dry_run,
            "sweep": self.plan.name,
            "fingerprint": self.plan.fingerprint,
            "variants": outcomes,
        }
        if dry_run:
            return response
        self._write_plan()
        summary_path = Path(self.plan.root) / "sweep_run_summary.json"
        response["status"] = "running"
        _atomic_write_json(summary_path, response)
        for variant in self.plan.variants:
            if variant.id not in selected:
                continue
            try:
                result = ExperimentRunner(
                    variant.plan,
                    repo_root=self.repo_root,
                ).run(
                    resume=resume,
                    start=start,
                    stop=stop,
                )
            except Exception as error:
                outcomes.append(
                    {
                        "variant": variant.id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                response["variants"] = outcomes
                response["status"] = "failed"
                _atomic_write_json(summary_path, response)
                raise
            outcomes.append(
                {
                    "variant": variant.id,
                    "status": "completed",
                    "result": result,
                }
            )
            response["variants"] = outcomes
            _atomic_write_json(summary_path, response)
        complete_suite = selected == known and stop in {None, "evaluate"}
        if complete_suite:
            comparison = aggregate_sweep_results(self.plan)
            response["comparison"] = str(Path(self.plan.root) / "comparison.json")
            response["ranking"] = comparison["ranking"]
        response["status"] = "completed"
        _atomic_write_json(summary_path, response)
        return response
