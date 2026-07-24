"""Matched multi-run ablations for the native sub-1B document VLM."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..architecture import estimate_parameters, load_blueprint
from .experiment import ExperimentPlan, ExperimentRunner, build_experiment_plan
from .gates import (
    evaluate_deployment_gates,
    load_evaluation_artifacts,
    write_gate_report,
)


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
    arm_id: str
    replicate_id: str
    hypothesis: str
    replicate_experiment_patches: tuple[dict[str, Any], ...]
    replicate_blueprint_patches: tuple[dict[str, Any], ...]
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
            "arm_id": self.arm_id,
            "replicate_id": self.replicate_id,
            "hypothesis": self.hypothesis,
            "replicate_experiment_patches": list(self.replicate_experiment_patches),
            "replicate_blueprint_patches": list(self.replicate_blueprint_patches),
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
class PromotionRule:
    primary_metric: str
    direction: str
    minimum_effect: float
    minimum_replicates: int
    familywise_alpha: float
    max_promotions: int
    required_gates: tuple[str, ...]
    required_axis_deltas: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "minimum_effect": self.minimum_effect,
            "minimum_replicates": self.minimum_replicates,
            "familywise_alpha": self.familywise_alpha,
            "max_promotions": self.max_promotions,
            "required_gates": list(self.required_gates),
            "required_axis_deltas": self.required_axis_deltas,
        }


@dataclass(frozen=True)
class SweepPlan:
    name: str
    root: str
    baseline: str
    base_experiment: str
    controls: tuple[MatchedControl, ...]
    replicate_controls: tuple[MatchedControl, ...]
    control_values_by_replicate: dict[str, dict[str, Any]]
    replicates: tuple[str, ...]
    variants: tuple[CompiledVariant, ...]
    promotion: PromotionRule | None
    fingerprint: str
    raw_spec: dict[str, Any]

    @property
    def control_values(self) -> dict[str, Any]:
        if len(self.control_values_by_replicate) == 1:
            return next(iter(self.control_values_by_replicate.values()))
        return self.control_values_by_replicate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "root": self.root,
            "baseline": self.baseline,
            "base_experiment": self.base_experiment,
            "matched_controls": [asdict(control) for control in self.controls],
            "replicate_controls": [
                asdict(control) for control in self.replicate_controls
            ],
            "control_values_by_replicate": self.control_values_by_replicate,
            "replicates": list(self.replicates),
            "variants": [variant.to_dict() for variant in self.variants],
            "promotion": (
                None if self.promotion is None else self.promotion.to_dict()
            ),
            "fingerprint": self.fingerprint,
        }


def _load_sweep(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("sweep root must be a mapping")
    return raw


def _parse_controls(
    raw: dict[str, Any],
    key: str = "matched_controls",
) -> tuple[MatchedControl, ...]:
    controls_raw = raw.get(key)
    if not isinstance(controls_raw, list) or not controls_raw:
        raise ValueError(f"sweep.{key} must be a non-empty list")
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
            raise ValueError(f"{key} document must be experiment or blueprint")
        _pointer_parts(control.path)
        if control.key in seen:
            raise ValueError(f"duplicate {key} entry {control.key}")
        seen.add(control.key)
        controls.append(control)
    return tuple(controls)


def _parse_replicates(
    raw: dict[str, Any],
    replicate_controls: tuple[MatchedControl, ...],
) -> tuple[dict[str, Any], ...]:
    replicates_raw = raw.get("replicates")
    if replicates_raw is None:
        return ({"id": "default", "explicit": False},)
    if not isinstance(replicates_raw, list) or not replicates_raw:
        raise ValueError("sweep.replicates must be a non-empty list when set")
    replicates = []
    seen = set()
    allowed = {
        (control.document, control.path)
        for control in replicate_controls
    }
    for item in replicates_raw:
        if not isinstance(item, dict):
            raise ValueError("every sweep replicate must be a mapping")
        replicate_id = str(item.get("id") or "")
        if not _NAME.fullmatch(replicate_id) or replicate_id in seen:
            raise ValueError("replicate ids must be unique safe names")
        seen.add(replicate_id)
        experiment_patches = _patch_list(item, "experiment_patches")
        blueprint_patches = _patch_list(item, "blueprint_patches")
        _validate_reserved_patches(experiment_patches)
        if not experiment_patches and not blueprint_patches:
            raise ValueError(f"replicate {replicate_id!r} has no seed or control patches")
        patched = {
            ("experiment", str(operation.get("path") or ""))
            for operation in experiment_patches
        } | {
            ("blueprint", str(operation.get("path") or ""))
            for operation in blueprint_patches
        }
        undeclared = sorted(patched - allowed)
        if undeclared:
            raise ValueError(
                f"replicate {replicate_id!r} patches undeclared replicate controls: "
                f"{undeclared}"
            )
        missing = sorted(allowed - patched)
        if missing:
            raise ValueError(
                f"replicate {replicate_id!r} does not set every replicate control: "
                f"{missing}"
            )
        replicates.append(
            {
                "id": replicate_id,
                "explicit": True,
                "experiment_patches": experiment_patches,
                "blueprint_patches": blueprint_patches,
            }
        )
    return tuple(replicates)


def _parse_promotion(raw: dict[str, Any]) -> PromotionRule | None:
    value = raw.get("promotion")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("sweep.promotion must be a mapping")
    primary_metric = str(value.get("primary_metric") or "")
    if primary_metric.startswith("axis."):
        if not primary_metric.removeprefix("axis."):
            raise ValueError(
                "promotion.primary_metric axis name must be non-empty"
            )
    elif not _NAME.fullmatch(primary_metric):
        raise ValueError("promotion.primary_metric must be a safe metric name")
    direction = str(value.get("direction") or "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("promotion.direction must be maximize or minimize")
    minimum_effect = value.get("minimum_effect", 0.0)
    if (
        not isinstance(minimum_effect, (int, float))
        or isinstance(minimum_effect, bool)
        or not math.isfinite(float(minimum_effect))
        or float(minimum_effect) < 0
    ):
        raise ValueError("promotion.minimum_effect must be finite and non-negative")
    minimum_replicates = value.get("minimum_replicates", 3)
    if (
        not isinstance(minimum_replicates, int)
        or isinstance(minimum_replicates, bool)
        or minimum_replicates < 2
    ):
        raise ValueError("promotion.minimum_replicates must be an integer >= 2")
    familywise_alpha = value.get("familywise_alpha", 0.05)
    if (
        not isinstance(familywise_alpha, (int, float))
        or isinstance(familywise_alpha, bool)
        or not math.isfinite(float(familywise_alpha))
        or not 0 < float(familywise_alpha) < 0.5
    ):
        raise ValueError("promotion.familywise_alpha must be within (0, 0.5)")
    max_promotions = value.get("max_promotions", 1)
    if (
        not isinstance(max_promotions, int)
        or isinstance(max_promotions, bool)
        or max_promotions < 1
    ):
        raise ValueError("promotion.max_promotions must be an integer >= 1")
    required_gates = value.get("required_gates", [])
    normalized_gates = (
        [str(gate) for gate in required_gates]
        if isinstance(required_gates, list)
        else []
    )
    if (
        not isinstance(required_gates, list)
        or any(not _NAME.fullmatch(gate) for gate in normalized_gates)
        or len(set(normalized_gates)) != len(normalized_gates)
    ):
        raise ValueError("promotion.required_gates must contain unique safe names")
    required_axis_deltas = value.get("required_axis_deltas", {})
    if not isinstance(required_axis_deltas, dict):
        raise ValueError("promotion.required_axis_deltas must be a mapping")
    normalized_axes: dict[str, float] = {}
    for axis, threshold in required_axis_deltas.items():
        if not isinstance(axis, str) or not axis:
            raise ValueError("promotion axis names must be non-empty strings")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(
                "promotion.required_axis_deltas values must be finite"
            )
        normalized_axes[axis] = float(threshold)
    return PromotionRule(
        primary_metric=primary_metric,
        direction=direction,
        minimum_effect=float(minimum_effect),
        minimum_replicates=minimum_replicates,
        familywise_alpha=float(familywise_alpha),
        max_promotions=max_promotions,
        required_gates=tuple(normalized_gates),
        required_axis_deltas=normalized_axes,
    )


def compile_sweep_plan(
    sweep_path: str | Path,
    *,
    repo_root: str | Path,
    python: str,
    compile_root: str | Path | None = None,
) -> SweepPlan:
    """Compile matched arms and replicate blocks into resumable experiments."""

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
    if raw.get("replicates") is None:
        replicate_controls: tuple[MatchedControl, ...] = ()
    else:
        replicate_controls = _parse_controls(raw, "replicate_controls")
        matched_keys = {control.key for control in controls}
        unpaired = [
            control.key
            for control in replicate_controls
            if control.key not in matched_keys
        ]
        if unpaired:
            raise ValueError(
                f"replicate controls must also be matched controls: {sorted(unpaired)}"
            )
    replicates = _parse_replicates(raw, replicate_controls)
    promotion = _parse_promotion(raw)
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or len(variants_raw) < 2:
        raise ValueError("sweep.variants must contain at least two variants")
    baseline = str(raw.get("baseline") or "")
    ids: set[str] = set()
    arm_specs = []
    for item in variants_raw:
        if not isinstance(item, dict):
            raise ValueError("every sweep variant must be a mapping")
        variant_id = str(item.get("id") or "")
        if not _NAME.fullmatch(variant_id) or variant_id in ids:
            raise ValueError("variant ids must be unique safe names")
        ids.add(variant_id)
        variant_experiment_patches = _patch_list(item, "experiment_patches")
        variant_blueprint_patches = _patch_list(item, "blueprint_patches")
        _validate_reserved_patches(variant_experiment_patches)
        if (
            variant_id != baseline
            and not variant_experiment_patches
            and not variant_blueprint_patches
        ):
            raise ValueError(f"non-baseline variant {variant_id!r} has no patches")
        arm_specs.append(
            (
                item,
                variant_id,
                variant_experiment_patches,
                variant_blueprint_patches,
            )
        )
    if baseline not in ids:
        raise ValueError("sweep baseline must name one compiled variant")

    compiled: list[CompiledVariant] = []
    control_values_by_replicate: dict[str, dict[str, Any]] = {}
    for replicate in replicates:
        replicate_id = str(replicate["id"])
        replicate_experiment_patches = list(replicate.get("experiment_patches") or [])
        replicate_blueprint_patches = list(replicate.get("blueprint_patches") or [])
        replicate_control_values: dict[str, Any] | None = None
        for (
            item,
            variant_id,
            variant_experiment_patches,
            variant_blueprint_patches,
        ) in arm_specs:
            experiment_patches = [
                *shared_experiment,
                *replicate_experiment_patches,
                *variant_experiment_patches,
            ]
            blueprint_patches = [
                *shared_blueprint,
                *replicate_blueprint_patches,
                *variant_blueprint_patches,
            ]
            _validate_reserved_patches(experiment_patches)
            experiment = apply_json_patch(base_experiment, experiment_patches)
            blueprint = apply_json_patch(base_blueprint, blueprint_patches)
            run_id = (
                f"{variant_id}--{replicate_id}"
                if bool(replicate["explicit"])
                else variant_id
            )
            variant_dir = compiled_root / run_id
            blueprint_path = variant_dir / "blueprint.yaml"
            experiment_path = variant_dir / "experiment.yaml"
            stable_blueprint_path = root / "compiled" / run_id / "blueprint.yaml"
            experiment["name"] = f"{name}--{run_id}"
            experiment["output_root"] = str(root / "runs" / run_id)
            experiment["blueprint"] = str(blueprint_path)
            evaluation = experiment.setdefault("evaluation", {})
            evaluation["wandb_group"] = name
            evaluation["wandb_run"] = f"{name}--{run_id}"
            tags = [str(tag) for tag in evaluation.get("wandb_tags") or []]
            evaluation["wandb_tags"] = list(
                dict.fromkeys(
                    [
                        *tags,
                        "native-student-sweep",
                        f"variant:{variant_id}",
                        f"replicate:{replicate_id}",
                    ]
                )
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
                    "replicate": replicate_id,
                    "experiment": canonical_experiment,
                    "blueprint": blueprint,
                    "input_fingerprints": plan.input_fingerprints,
                }
            )
            plan = replace(
                plan,
                raw_spec=canonical_experiment,
                fingerprint=variant_fingerprint,
            )
            documents = {"experiment": canonical_experiment, "blueprint": blueprint}
            values = {
                control.key: copy.deepcopy(
                    _pointer_get(documents[control.document], control.path)
                )
                for control in controls
            }
            if replicate_control_values is None:
                replicate_control_values = values
            elif values != replicate_control_values:
                mismatches = [
                    key
                    for key in sorted(values)
                    if values[key] != replicate_control_values.get(key)
                ]
                raise ValueError(
                    f"variant {variant_id!r} violates matched controls in "
                    f"replicate {replicate_id!r}: {mismatches}"
                )
            compiled.append(
                CompiledVariant(
                    id=run_id,
                    arm_id=variant_id,
                    replicate_id=replicate_id,
                    hypothesis=str(item.get("hypothesis") or ""),
                    replicate_experiment_patches=tuple(
                        copy.deepcopy(replicate_experiment_patches)
                    ),
                    replicate_blueprint_patches=tuple(
                        copy.deepcopy(replicate_blueprint_patches)
                    ),
                    experiment_patches=tuple(copy.deepcopy(experiment_patches)),
                    blueprint_patches=tuple(copy.deepcopy(blueprint_patches)),
                    experiment_path=str(experiment_path),
                    blueprint_path=str(blueprint_path),
                    plan=plan,
                    parameters=estimate_parameters(plan.resolved_blueprint),
                    fingerprint=variant_fingerprint,
                )
            )
        control_values_by_replicate[replicate_id] = replicate_control_values or {}
    sweep_fingerprint = _fingerprint(
        {
            "spec": raw,
            "promotion": (
                None if promotion is None else promotion.to_dict()
            ),
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
        replicate_controls=replicate_controls,
        control_values_by_replicate=control_values_by_replicate,
        replicates=tuple(str(replicate["id"]) for replicate in replicates),
        variants=tuple(compiled),
        promotion=promotion,
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


def _pretraining_efficiency(run_root: Path) -> dict[str, float]:
    pointer = run_root / "artifacts" / "pretrain" / "latest_checkpoint.txt"
    if not pointer.is_file():
        return {}
    checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    dense = int(state.get("dense_visual_tokens_seen", 0))
    executed = int(state.get("executed_visual_tokens_seen", dense))
    valid = int(state.get("valid_visual_tokens_seen", 0))
    samples = int(state.get("visual_samples_seen", 0))
    if (
        dense < 0
        or executed < 0
        or valid < 0
        or samples < 0
        or valid > executed
    ):
        raise ValueError(f"invalid visual efficiency counters in {state_path}")
    return {
        "student_flops": float(state.get("student_flops_seen", 0)),
        "dense_visual_tokens_per_sample": (
            dense / samples if samples else 0.0
        ),
        "executed_visual_tokens_per_sample": (
            executed / samples if samples else 0.0
        ),
        "valid_visual_token_fraction": (
            valid / executed if executed else 0.0
        ),
    }


def _pretraining_attention_backend(run_root: Path) -> str:
    pointer = run_root / "artifacts" / "pretrain" / "latest_checkpoint.txt"
    if not pointer.is_file():
        return "missing"
    checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        return "missing"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return str(state.get("visual_attention_backend") or "legacy")


def _axis_scores(comparison: dict[str, Any], split: str) -> dict[str, float]:
    axes = comparison["splits"][split].get("by_answer_type", {})
    return {name: float(values["score"]) for name, values in sorted(axes.items())}


def _robustness_scores(
    comparison: dict[str, Any],
    split: str,
) -> dict[str, float]:
    axes = comparison["splits"][split].get("by_robustness_axis", {})
    return {
        f"{axis}/{value}": float(summary["score"])
        for axis, slices in sorted(axes.items())
        for value, summary in sorted(slices.items())
    }


def _nested_slice_values(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nested: dict[str, dict[str, Any]] = {}
    for key, value in sorted(values.items()):
        axis, slice_value = key.split("/", 1)
        nested.setdefault(axis, {})[slice_value] = value
    return nested


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _distribution(values: list[float], *, key: str) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    mean = _mean(values)
    if len(values) == 1:
        standard_deviation = 0.0
        ci95 = None
    else:
        standard_deviation = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        bootstrap_means = sorted(
            _mean([values[rng.randrange(len(values))] for _ in values])
            for _ in range(10_000)
        )
        lower = bootstrap_means[math.floor(0.025 * (len(bootstrap_means) - 1))]
        upper = bootstrap_means[math.ceil(0.975 * (len(bootstrap_means) - 1))]
        ci95 = [round(lower, 8), round(upper, 8)]
    return {
        "n": len(values),
        "mean": round(mean, 8),
        "standard_deviation": round(standard_deviation, 8),
        "minimum": round(min(values), 8),
        "maximum": round(max(values), 8),
        "ci95": ci95,
    }


def _one_sided_lower_bound(
    values: list[float],
    *,
    alpha: float,
    key: str,
) -> float | None:
    if len(values) < 2:
        return None
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    bootstrap_means = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(10_000)
    )
    index = math.floor(alpha * (len(bootstrap_means) - 1))
    return round(bootstrap_means[index], 8)


def _promotion_delta(
    record: dict[str, Any],
    primary_metric: str,
) -> float | None:
    if primary_metric.startswith("axis."):
        axis = primary_metric.removeprefix("axis.")
        value = record.get("heldout_axis_delta_vs_baseline", {}).get(axis)
    else:
        value = record.get("delta_vs_baseline", {}).get(primary_metric)
    return None if value is None else float(value)


def _promotion_decision(
    plan: SweepPlan,
    arm_records: dict[str, dict[str, Any]],
    records_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    rule = plan.promotion
    if rule is None:
        return {
            "status": "not_configured",
            "selected_variants": [],
            "baseline_retained": True,
            "candidates": {},
        }
    candidate_ids = sorted(set(arm_records) - {plan.baseline})
    if not candidate_ids:
        raise ValueError("promotion requires at least one non-baseline arm")
    comparisons_per_candidate = 1 + len(rule.required_axis_deltas)
    comparison_count = len(candidate_ids) * comparisons_per_candidate
    corrected_alpha = rule.familywise_alpha / comparison_count
    direction_sign = 1.0 if rule.direction == "maximize" else -1.0
    candidates: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for arm_id in candidate_ids:
        ordered = [
            records_by_arm[arm_id][replicate_id]
            for replicate_id in plan.replicates
        ]
        missing_metric = [
            record["replicate_id"]
            for record in ordered
            if _promotion_delta(record, rule.primary_metric) is None
        ]
        benefits = (
            []
            if missing_metric
            else [
                direction_sign
                * float(_promotion_delta(record, rule.primary_metric))
                for record in ordered
            ]
        )
        lower_bound = (
            _one_sided_lower_bound(
                benefits,
                alpha=corrected_alpha,
                key=(
                    f"{plan.fingerprint}:{arm_id}:{rule.primary_metric}:"
                    "promotion"
                ),
            )
            if benefits
            else None
        )
        gate_statuses = arm_records[arm_id]["gate_statuses"]
        failed_gates = sorted(
            gate
            for gate in rule.required_gates
            if gate_statuses.get(gate) == "fail"
        )
        incomplete_gates = sorted(
            gate
            for gate in rule.required_gates
            if gate_statuses.get(gate) not in {"pass", "fail"}
        )
        axis_evidence: dict[str, dict[str, Any]] = {}
        missing_axes = []
        regressed_axes = []
        for axis, minimum_delta in sorted(
            rule.required_axis_deltas.items()
        ):
            if any(
                axis not in record["heldout_axis_delta_vs_baseline"]
                for record in ordered
            ):
                missing_axes.append(axis)
                continue
            values = [
                float(record["heldout_axis_delta_vs_baseline"][axis])
                for record in ordered
            ]
            axis_lower_bound = _one_sided_lower_bound(
                values,
                alpha=corrected_alpha,
                key=f"{plan.fingerprint}:{arm_id}:{axis}:promotion-axis",
            )
            axis_evidence[axis] = {
                "mean_delta": round(_mean(values), 8),
                "simultaneous_lower_bound": axis_lower_bound,
                "minimum_delta": minimum_delta,
            }
            if (
                axis_lower_bound is not None
                and axis_lower_bound < minimum_delta
            ):
                regressed_axes.append(axis)
        evidence = {
            "replicate_count": len(ordered),
            "primary_metric": rule.primary_metric,
            "direction": rule.direction,
            "mean_benefit": (
                None if not benefits else round(_mean(benefits), 8)
            ),
            "simultaneous_lower_bound": lower_bound,
            "minimum_effect": rule.minimum_effect,
            "failed_gates": failed_gates,
            "incomplete_gates": incomplete_gates,
            "missing_metric_replicates": missing_metric,
            "missing_axes": missing_axes,
            "regressed_axes": regressed_axes,
            "axis_guardrails": axis_evidence,
            "parameters": arm_records[arm_id]["parameters"],
        }
        if failed_gates or regressed_axes:
            decision = "reject"
        elif (
            len(ordered) < rule.minimum_replicates
            or lower_bound is None
            or missing_metric
            or incomplete_gates
            or missing_axes
            or any(
                item["simultaneous_lower_bound"] is None
                for item in axis_evidence.values()
            )
        ):
            decision = "insufficient_evidence"
        elif lower_bound <= rule.minimum_effect:
            decision = "retain_baseline"
        else:
            decision = "eligible"
            eligible.append(arm_id)
        candidates[arm_id] = {
            "decision": decision,
            "evidence": evidence,
        }
    eligible.sort(
        key=lambda arm_id: (
            -float(
                candidates[arm_id]["evidence"]["simultaneous_lower_bound"]
            ),
            -float(candidates[arm_id]["evidence"]["mean_benefit"]),
            int(candidates[arm_id]["evidence"]["parameters"]["total"]),
            arm_id,
        )
    )
    selected = eligible[: rule.max_promotions]
    for arm_id in eligible:
        candidates[arm_id]["decision"] = (
            "promote" if arm_id in selected else "eligible_not_selected"
        )
    if selected:
        status = "promote"
    elif any(
        record["decision"] == "insufficient_evidence"
        for record in candidates.values()
    ):
        status = "insufficient_evidence"
    else:
        status = "retain_baseline"
    return {
        "status": status,
        "selected_variants": selected,
        "baseline_retained": not bool(selected),
        "contract": rule.to_dict(),
        "multiple_comparisons": {
            "method": "bonferroni_one_sided_percentile_bootstrap",
            "candidate_count": len(candidate_ids),
            "guardrails_per_candidate": len(rule.required_axis_deltas),
            "comparison_count": comparison_count,
            "familywise_alpha": rule.familywise_alpha,
            "per_comparison_alpha": corrected_alpha,
            "simultaneous_confidence_level": 1.0 - corrected_alpha,
            "resamples": 10_000,
        },
        "candidates": candidates,
    }


def _paired_conclusion(statistics: dict[str, Any]) -> str:
    ci95 = statistics["ci95"]
    if ci95 is None:
        return "insufficient_replicates"
    if ci95[0] > 0.0:
        return "improved"
    if ci95[1] < 0.0:
        return "degraded"
    return "inconclusive"


def _evaluation_parameter_counts(
    evaluation_root: Path,
    fallback: dict[str, int],
) -> tuple[dict[str, int], str]:
    gate_path = evaluation_root / "gates.json"
    if gate_path.is_file():
        report = json.loads(gate_path.read_text(encoding="utf-8"))
        for gate in report.get("gates", ()):
            if gate.get("id") != "parameter_budget":
                continue
            evidence = gate.get("evidence", {})
            if "actual_parameters" not in evidence:
                break
            counts = {
                str(key): int(value)
                for key, value in evidence.get("components", {}).items()
            }
            counts["total"] = int(evidence["actual_parameters"])
            return counts, "evaluation_actual"
    return {key: int(value) for key, value in fallback.items()}, "blueprint_estimate"


def _monolingual_control_comparison(
    evaluation_root: Path,
) -> dict[str, Any] | None:
    manifest_path = evaluation_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured = manifest.get("monolingual_control_evaluation")
    if not configured:
        return None
    comparison, _ = load_evaluation_artifacts(configured)
    return comparison


def _aggregate_gate_reports(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot aggregate an empty gate-report block")
    ordered_replicates = sorted(reports)
    first = reports[ordered_replicates[0]]
    gate_ids = [gate["id"] for gate in first["gates"]]
    gates = []
    for gate_id in gate_ids:
        replicate_gates = {
            replicate: next(
                gate
                for gate in reports[replicate]["gates"]
                if gate["id"] == gate_id
            )
            for replicate in ordered_replicates
        }
        statuses = {
            replicate: gate["status"]
            for replicate, gate in replicate_gates.items()
        }
        if "fail" in statuses.values():
            status = "fail"
        elif all(value == "pass" for value in statuses.values()):
            status = "pass"
        else:
            status = "insufficient_evidence"
        gates.append(
            {
                "id": gate_id,
                "requirement": next(iter(replicate_gates.values()))[
                    "requirement"
                ],
                "status": status,
                "replicate_statuses": statuses,
            }
        )
    statuses = [gate["status"] for gate in gates]
    overall = (
        "fail"
        if "fail" in statuses
        else "insufficient_evidence"
        if "insufficient_evidence" in statuses
        else "pass"
    )
    return {
        "schema_version": 1,
        "overall_status": overall,
        "counts": {
            status: statuses.count(status)
            for status in ("pass", "fail", "insufficient_evidence")
        },
        "replicates": ordered_replicates,
        "gates": gates,
    }


def aggregate_sweep_results(plan: SweepPlan) -> dict[str, Any]:
    """Aggregate run metrics and paired replicate deltas against the baseline arm."""

    run_records: dict[str, dict[str, Any]] = {}
    blueprints_by_run = {
        variant.id: variant.plan.resolved_blueprint
        for variant in plan.variants
    }
    artifact_controls: dict[str, dict[str, str]] = {}
    for variant in plan.variants:
        run_root = Path(variant.plan.root)
        fingerprints = {
            split: _evaluation_set_fingerprint(run_root, split)
            for split in ("train", "heldout")
        }
        expected_artifacts = artifact_controls.get(variant.replicate_id)
        if expected_artifacts is None:
            artifact_controls[variant.replicate_id] = fingerprints
        elif fingerprints != expected_artifacts:
            mismatches = [
                split
                for split in sorted(fingerprints)
                if fingerprints[split] != expected_artifacts.get(split)
            ]
            raise ValueError(
                f"variant {variant.arm_id!r} has mismatched evaluation artifacts "
                f"in replicate {variant.replicate_id!r}: {mismatches}"
            )
        comparison_path = (
            run_root / "artifacts" / "evaluation" / "comparison.json"
        )
        if not comparison_path.is_file():
            raise FileNotFoundError(
                f"variant {variant.id!r} has no evaluation comparison: {comparison_path}"
            )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        evaluation_root = comparison_path.parent
        run_records[variant.id] = {
            "arm_id": variant.arm_id,
            "replicate_id": variant.replicate_id,
            "hypothesis": variant.hypothesis,
            "fingerprint": variant.fingerprint,
            "parameters": variant.parameters,
            "metrics": _summary_metrics(comparison),
            "heldout_by_answer_type": _axis_scores(comparison, "heldout"),
            "train_by_answer_type": _axis_scores(comparison, "train"),
            "heldout_slice_scores": _robustness_scores(
                comparison,
                "heldout",
            ),
            "train_slice_scores": _robustness_scores(comparison, "train"),
            "pretraining_efficiency": _pretraining_efficiency(run_root),
            "pretraining_visual_attention_backend": (
                _pretraining_attention_backend(run_root)
            ),
            "comparison": str(comparison_path),
            "evaluation_root": str(evaluation_root),
        }

    records_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id, record in run_records.items():
        arm_runs = records_by_arm.setdefault(record["arm_id"], {})
        replicate_id = record["replicate_id"]
        if replicate_id in arm_runs:
            raise ValueError(
                f"arm {record['arm_id']!r} has duplicate replicate {replicate_id!r}"
            )
        record["run_id"] = run_id
        arm_runs[replicate_id] = record
    expected_replicates = set(plan.replicates)
    for arm_id, arm_runs in records_by_arm.items():
        observed = set(arm_runs)
        if observed != expected_replicates:
            raise ValueError(
                f"arm {arm_id!r} has incomplete replicate block: "
                f"expected {sorted(expected_replicates)}, observed {sorted(observed)}"
            )
    if plan.baseline not in records_by_arm:
        raise ValueError(f"baseline arm {plan.baseline!r} has no completed runs")

    baseline_runs = records_by_arm[plan.baseline]
    run_gate_reports: dict[str, dict[str, Any]] = {}
    for record in run_records.values():
        baseline = baseline_runs[record["replicate_id"]]
        baseline_metrics = baseline["metrics"]
        baseline_axes = baseline["heldout_by_answer_type"]
        baseline_slices = baseline["heldout_slice_scores"]
        baseline_efficiency = baseline["pretraining_efficiency"]
        record["delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_metrics[name]), 8)
            for name, value in record["metrics"].items()
        }
        record["heldout_axis_delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_axes[name]), 8)
            for name, value in record["heldout_by_answer_type"].items()
            if name in baseline_axes
        }
        record["heldout_slice_delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_slices[name]), 8)
            for name, value in record["heldout_slice_scores"].items()
            if name in baseline_slices
        }
        record["pretraining_efficiency_delta_vs_baseline"] = {
            name: round(float(value) - float(baseline_efficiency[name]), 8)
            for name, value in record["pretraining_efficiency"].items()
            if name in baseline_efficiency
        }
        evaluation_root = Path(record["evaluation_root"])
        current_comparison, current_rows = load_evaluation_artifacts(
            evaluation_root
        )
        parameter_counts, parameter_count_source = (
            _evaluation_parameter_counts(
                evaluation_root,
                record["parameters"],
            )
        )
        if record["arm_id"] == plan.baseline:
            baseline_comparison = None
            baseline_rows = None
        else:
            baseline_root = Path(baseline["evaluation_root"])
            baseline_comparison, baseline_rows = load_evaluation_artifacts(
                baseline_root
            )
        gate_report = evaluate_deployment_gates(
            blueprints_by_run[record["run_id"]],
            parameter_counts,
            current_comparison,
            current_rows,
            baseline_comparison=baseline_comparison,
            baseline_rows=baseline_rows,
            monolingual_control_comparison=_monolingual_control_comparison(
                evaluation_root
            ),
        )
        gate_report["parameter_count_source"] = parameter_count_source
        gate_path = (
            Path(plan.root) / "gates" / f"{record['run_id']}.json"
        )
        write_gate_report(gate_path, gate_report)
        record["gate_report"] = str(gate_path)
        record["gate_status"] = gate_report["overall_status"]
        record["gate_statuses"] = {
            gate["id"]: gate["status"]
            for gate in gate_report["gates"]
        }
        run_gate_reports[record["run_id"]] = gate_report

    arm_records: dict[str, dict[str, Any]] = {}
    for arm_id, arm_runs in records_by_arm.items():
        ordered = [arm_runs[replicate_id] for replicate_id in plan.replicates]
        parameters = [record["parameters"] for record in ordered]
        if any(value != parameters[0] for value in parameters[1:]):
            raise ValueError(f"arm {arm_id!r} changes architecture across replicates")
        metric_names = sorted(ordered[0]["metrics"])
        metric_statistics = {
            name: _distribution(
                [float(record["metrics"][name]) for record in ordered],
                key=f"{plan.fingerprint}:{arm_id}:{name}:metric",
            )
            for name in metric_names
        }
        paired_statistics = {
            name: _distribution(
                [float(record["delta_vs_baseline"][name]) for record in ordered],
                key=f"{plan.fingerprint}:{arm_id}:{name}:paired",
            )
            for name in metric_names
        }
        heldout_axes = sorted(
            set.intersection(
                *[
                    set(record["heldout_by_answer_type"])
                    for record in ordered
                ]
            )
        )
        train_axes = sorted(
            set.intersection(
                *[
                    set(record["train_by_answer_type"])
                    for record in ordered
                ]
            )
        )
        heldout_axis_statistics = {
            axis: _distribution(
                [
                    float(record["heldout_by_answer_type"][axis])
                    for record in ordered
                ],
                key=f"{plan.fingerprint}:{arm_id}:{axis}:heldout-axis",
            )
            for axis in heldout_axes
        }
        heldout_axis_delta_statistics = {
            axis: _distribution(
                [
                    float(record["heldout_axis_delta_vs_baseline"][axis])
                    for record in ordered
                ],
                key=f"{plan.fingerprint}:{arm_id}:{axis}:paired-axis",
            )
            for axis in heldout_axes
            if all(
                axis in record["heldout_axis_delta_vs_baseline"]
                for record in ordered
            )
        }
        heldout_slices = sorted(
            set.intersection(
                *[
                    set(record["heldout_slice_scores"])
                    for record in ordered
                ]
            )
        )
        train_slices = sorted(
            set.intersection(
                *[
                    set(record["train_slice_scores"])
                    for record in ordered
                ]
            )
        )
        heldout_slice_statistics = {
            slice_name: _distribution(
                [
                    float(record["heldout_slice_scores"][slice_name])
                    for record in ordered
                ],
                key=(
                    f"{plan.fingerprint}:{arm_id}:{slice_name}:"
                    "heldout-robustness-slice"
                ),
            )
            for slice_name in heldout_slices
        }
        heldout_slice_delta_statistics = {
            slice_name: _distribution(
                [
                    float(record["heldout_slice_delta_vs_baseline"][slice_name])
                    for record in ordered
                ],
                key=(
                    f"{plan.fingerprint}:{arm_id}:{slice_name}:"
                    "paired-robustness-slice"
                ),
            )
            for slice_name in heldout_slices
            if all(
                slice_name in record["heldout_slice_delta_vs_baseline"]
                for record in ordered
            )
        }
        efficiency_names = sorted(
            set.intersection(
                *[
                    set(record["pretraining_efficiency"])
                    for record in ordered
                ]
            )
        )
        efficiency_statistics = {
            name: _distribution(
                [
                    float(record["pretraining_efficiency"][name])
                    for record in ordered
                ],
                key=f"{plan.fingerprint}:{arm_id}:{name}:pretraining-efficiency",
            )
            for name in efficiency_names
        }
        efficiency_delta_statistics = {
            name: _distribution(
                [
                    float(
                        record["pretraining_efficiency_delta_vs_baseline"][name]
                    )
                    for record in ordered
                ],
                key=(
                    f"{plan.fingerprint}:{arm_id}:{name}:"
                    "paired-pretraining-efficiency"
                ),
            )
            for name in efficiency_names
            if all(
                name in record["pretraining_efficiency_delta_vs_baseline"]
                for record in ordered
            )
        }
        arm_gate_report = _aggregate_gate_reports(
            {
                record["replicate_id"]: run_gate_reports[record["run_id"]]
                for record in ordered
            }
        )
        arm_gate_path = Path(plan.root) / "gates" / f"{arm_id}.json"
        write_gate_report(arm_gate_path, arm_gate_report)
        arm_records[arm_id] = {
            "hypothesis": ordered[0]["hypothesis"],
            "parameters": parameters[0],
            "replicate_count": len(ordered),
            "runs": {
                record["replicate_id"]: record["run_id"]
                for record in ordered
            },
            "metrics": {
                name: statistics["mean"]
                for name, statistics in metric_statistics.items()
            },
            "metric_statistics": metric_statistics,
            "delta_vs_baseline": {
                name: statistics["mean"]
                for name, statistics in paired_statistics.items()
            },
            "paired_delta_statistics": paired_statistics,
            "heldout_by_answer_type": {
                axis: statistics["mean"]
                for axis, statistics in heldout_axis_statistics.items()
            },
            "train_by_answer_type": {
                axis: round(
                    _mean(
                        [
                            float(record["train_by_answer_type"][axis])
                            for record in ordered
                        ]
                    ),
                    8,
                )
                for axis in train_axes
            },
            "heldout_axis_delta_vs_baseline": {
                axis: statistics["mean"]
                for axis, statistics in heldout_axis_delta_statistics.items()
            },
            "heldout_axis_delta_statistics": heldout_axis_delta_statistics,
            "heldout_by_robustness_axis": _nested_slice_values(
                {
                    name: statistics["mean"]
                    for name, statistics in heldout_slice_statistics.items()
                }
            ),
            "train_by_robustness_axis": _nested_slice_values(
                {
                    name: round(
                        _mean(
                            [
                                float(record["train_slice_scores"][name])
                                for record in ordered
                            ]
                        ),
                        8,
                    )
                    for name in train_slices
                }
            ),
            "heldout_robustness_statistics": _nested_slice_values(
                heldout_slice_statistics
            ),
            "heldout_robustness_delta_vs_baseline": _nested_slice_values(
                {
                    name: statistics["mean"]
                    for name, statistics
                    in heldout_slice_delta_statistics.items()
                }
            ),
            "heldout_robustness_delta_statistics": _nested_slice_values(
                heldout_slice_delta_statistics
            ),
            "pretraining_efficiency": {
                name: statistics["mean"]
                for name, statistics in efficiency_statistics.items()
            },
            "pretraining_efficiency_statistics": efficiency_statistics,
            "pretraining_efficiency_delta_vs_baseline": {
                name: statistics["mean"]
                for name, statistics in efficiency_delta_statistics.items()
            },
            "pretraining_efficiency_delta_statistics": (
                efficiency_delta_statistics
            ),
            "heldout_score_conclusion": (
                "reference"
                if arm_id == plan.baseline
                else _paired_conclusion(paired_statistics["heldout_score"])
            ),
            "gate_report": str(arm_gate_path),
            "gate_status": arm_gate_report["overall_status"],
            "gate_statuses": {
                gate["id"]: gate["status"]
                for gate in arm_gate_report["gates"]
            },
        }
    ranking = sorted(
        arm_records,
        key=lambda variant_id: (
            -arm_records[variant_id]["metrics"]["heldout_score"],
            arm_records[variant_id]["metrics"]["train_minus_heldout_score"],
            variant_id,
        ),
    )
    matched_artifacts: dict[str, Any]
    if len(artifact_controls) == 1:
        matched_artifacts = next(iter(artifact_controls.values()))
    else:
        matched_artifacts = artifact_controls
    promotion = _promotion_decision(
        plan,
        arm_records,
        records_by_arm,
    )
    result = {
        "schema_version": 4,
        "sweep": plan.name,
        "sweep_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "replicates": list(plan.replicates),
        "replicate_count": len(plan.replicates),
        "confidence_interval": {
            "method": "paired_percentile_bootstrap",
            "level": 0.95,
            "resamples": 10_000,
            "deterministic_seed_source": "sweep_fingerprint_arm_metric",
        },
        "matched_controls": plan.control_values,
        "matched_controls_by_replicate": plan.control_values_by_replicate,
        "matched_evaluation_artifacts": matched_artifacts,
        "matched_evaluation_artifacts_by_replicate": artifact_controls,
        "ranking": ranking,
        "promotion": promotion,
        "variants": arm_records,
        "runs": run_records,
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
        f"Paired replicates: **{result['replicate_count']}**",
        "",
        "| Rank | Variant | Parameters | Heldout mean ± SD | Paired delta [95% CI] | "
        "Train-heldout gap | Conclusion | Gates |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, variant_id in enumerate(result["ranking"], start=1):
        record = result["variants"][variant_id]
        metrics = record["metrics"]
        heldout = record["metric_statistics"]["heldout_score"]
        delta = record["paired_delta_statistics"]["heldout_score"]
        ci95 = (
            "unavailable"
            if delta["ci95"] is None
            else f"{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}"
        )
        lines.append(
            f"| {rank} | `{variant_id}` | {record['parameters']['total']:,} | "
            f"{heldout['mean']:.6f} ± {heldout['standard_deviation']:.6f} | "
            f"{delta['mean']:+.6f} [{ci95}] | "
            f"{metrics['train_minus_heldout_score']:+.6f} | "
            f"`{record['heldout_score_conclusion']}` | "
            f"`{record['gate_status']}` |"
        )
    promotion = result["promotion"]
    lines.extend(
        [
            "",
            "## Promotion decision",
            "",
            f"- Status: `{promotion['status']}`",
            "- Selected variants: "
            + (
                ", ".join(
                    f"`{variant}`"
                    for variant in promotion["selected_variants"]
                )
                if promotion["selected_variants"]
                else "none"
            ),
        ]
    )
    if promotion["status"] != "not_configured":
        multiplicity = promotion["multiple_comparisons"]
        lines.extend(
            [
                "- Multiplicity control: "
                f"`{multiplicity['method']}` over "
                f"{multiplicity['comparison_count']} comparisons",
                "",
                "| Candidate | Decision | Mean benefit | Simultaneous lower bound |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for variant_id, candidate in sorted(
            promotion["candidates"].items()
        ):
            evidence = candidate["evidence"]
            mean_benefit = evidence["mean_benefit"]
            lower_bound = evidence["simultaneous_lower_bound"]
            mean_text = (
                "unavailable"
                if mean_benefit is None
                else f"{mean_benefit:+.6f}"
            )
            lower_text = (
                "unavailable"
                if lower_bound is None
                else f"{lower_bound:+.6f}"
            )
            lines.append(
                f"| `{variant_id}` | `{candidate['decision']}` | "
                f"{mean_text} | {lower_text} |"
            )
    lines.extend(["", "## Matched controls by replicate", ""])
    for replicate_id, controls in sorted(
        result["matched_controls_by_replicate"].items()
    ):
        lines.append(f"### {replicate_id}")
        lines.append("")
        for key, value in sorted(controls.items()):
            lines.append(f"- `{key}`: `{_stable_json(value)}`")
        lines.append("")
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
        replicate_ids: set[str] | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> dict[str, Any]:
        known = {variant.id for variant in self.plan.variants}
        known_arms = {variant.arm_id for variant in self.plan.variants}
        selected_arms = known_arms if variant_ids is None else set(variant_ids)
        unknown_arms = selected_arms - known_arms
        if unknown_arms:
            raise ValueError(f"unknown sweep variants: {sorted(unknown_arms)}")
        known_replicates = set(self.plan.replicates)
        selected_replicates = (
            known_replicates if replicate_ids is None else set(replicate_ids)
        )
        unknown_replicates = selected_replicates - known_replicates
        if unknown_replicates:
            raise ValueError(
                f"unknown sweep replicates: {sorted(unknown_replicates)}"
            )
        selected = {
            variant.id
            for variant in self.plan.variants
            if variant.arm_id in selected_arms
            and variant.replicate_id in selected_replicates
        }
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
                outcomes.append(
                    {
                        "run": variant.id,
                        "variant": variant.arm_id,
                        "replicate": variant.replicate_id,
                        "result": result,
                    }
                )
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
                        "run": variant.id,
                        "variant": variant.arm_id,
                        "replicate": variant.replicate_id,
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
                    "run": variant.id,
                    "variant": variant.arm_id,
                    "replicate": variant.replicate_id,
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
            response["promotion"] = {
                "status": comparison["promotion"]["status"],
                "selected_variants": comparison["promotion"][
                    "selected_variants"
                ],
                "baseline_retained": comparison["promotion"][
                    "baseline_retained"
                ],
            }
        response["status"] = "completed"
        _atomic_write_json(summary_path, response)
        return response
