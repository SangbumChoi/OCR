"""Resumable end-to-end orchestration for the native document VLM student."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..architecture import estimate_parameters, load_blueprint, validate_blueprint
from .acquisition import HubComponentSpec
from .checkpoint_acquisition import (
    HubCheckpointSpec,
    checkpoint_manifest_valid,
    checkpoint_path_from_manifest,
)
from .config import StudentConfig
from .mixture import MixtureComponent, validate_components


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CHECKPOINT_STAGES = {"pretrain", "sft", "rlvr"}
_OUTPUT_FLAGS = {"--out", "--output", "--save"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()}"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def _evaluation_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"evaluation root does not exist: {path}")
    files = [
        candidate
        for pattern in ("comparison.json", "*/summary.json", "*/per_sample.jsonl")
        for candidate in path.glob(pattern)
        if candidate.is_file()
    ]
    if not any(candidate.name == "comparison.json" for candidate in files):
        raise ValueError(f"evaluation root has no comparison.json: {path}")
    records = []
    for candidate in sorted(set(files)):
        record = _file_fingerprint(candidate)
        record["path"] = str(candidate.relative_to(path))
        records.append(record)
    return {
        "path": str(path),
        "files": len(records),
        "sha256": _fingerprint(records),
    }


def _source_fingerprint(
    repo_root: Path,
    stages: Iterable["ExperimentStage"],
) -> dict[str, Any]:
    files = set((repo_root / "src" / "docvlm_eval").rglob("*.py"))
    for stage in stages:
        for argument in stage.command:
            path = Path(argument)
            if path.suffix == ".py" and path.is_file():
                files.add(path.resolve())
    records = []
    for path in sorted(files):
        record = _file_fingerprint(path)
        try:
            record["path"] = str(path.relative_to(repo_root))
        except ValueError:
            record["path"] = str(path)
        records.append(record)
    return {
        "files": len(records),
        "sha256": _fingerprint(records),
    }


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
class Artifact:
    path: str
    kind: str = "file"

    def validate(self) -> bool:
        path = Path(self.path)
        if self.kind == "file":
            return path.is_file() and path.stat().st_size > 0
        if self.kind == "directory":
            return path.is_dir() and any(path.iterdir())
        if self.kind == "checkpoint_manifest":
            return checkpoint_manifest_valid(path)
        raise ValueError(f"unknown artifact kind {self.kind!r}")


@dataclass(frozen=True)
class ExperimentStage:
    name: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    artifacts: tuple[Artifact, ...]

    def signature(self, experiment_fingerprint: str, dependency_signatures: dict[str, str]) -> str:
        return _fingerprint(
            {
                "experiment": experiment_fingerprint,
                "name": self.name,
                "command": self.command,
                "dependencies": {
                    dependency: dependency_signatures[dependency]
                    for dependency in self.dependencies
                },
                "artifacts": [asdict(artifact) for artifact in self.artifacts],
            }
        )

    def artifacts_valid(self) -> bool:
        return bool(self.artifacts) and all(artifact.validate() for artifact in self.artifacts)


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    root: str
    blueprint: str
    resolved_blueprint: dict[str, Any]
    raw_spec: dict[str, Any]
    components: tuple[MixtureComponent, ...]
    stages: tuple[ExperimentStage, ...]
    fingerprint: str
    input_fingerprints: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "root": self.root,
            "blueprint": self.blueprint,
            "fingerprint": self.fingerprint,
            "input_fingerprints": self.input_fingerprints,
            "components": [asdict(component) for component in self.components],
            "stages": [
                {
                    "name": stage.name,
                    "command": list(stage.command),
                    "dependencies": list(stage.dependencies),
                    "artifacts": [asdict(artifact) for artifact in stage.artifacts],
                }
                for stage in self.stages
            ],
        }


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _checkpoint_path_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return _file_fingerprint(path)
    if not path.is_dir():
        raise ValueError(f"initialization checkpoint does not exist: {path}")
    names = {
        "config.json",
        "student_config.json",
        "model.pt",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (
            candidate.name in names
            or candidate.name.startswith("model-")
            or candidate.name.startswith("pytorch_model-")
        )
    ]
    if not files:
        raise ValueError(
            f"initialization checkpoint has no supported files: {path}"
        )
    records = []
    for candidate in sorted(files):
        record = _file_fingerprint(candidate)
        record["path"] = str(candidate.relative_to(path))
        records.append(record)
    return {
        "path": str(path),
        "files": len(records),
        "sha256": _fingerprint(records),
    }


def _checkpoint_sources(
    initialization: dict[str, Any],
    repo_root: Path,
    artifacts: Path,
) -> tuple[
    dict[str, str],
    dict[str, HubCheckpointSpec],
    dict[str, Path],
    dict[str, Any],
]:
    resolved: dict[str, str] = {}
    hub_specs: dict[str, HubCheckpointSpec] = {}
    manifests: dict[str, Path] = {}
    fingerprints: dict[str, Any] = {}
    for component in ("vision", "language"):
        source = initialization.get(f"{component}_source")
        if source is None:
            continue
        family = str(
            initialization.get(f"{component}_family")
            or ("siglip" if component == "vision" else "llama")
        )
        if isinstance(source, (str, Path)):
            path = _resolve_path(repo_root, source)
            resolved[component] = str(path)
            fingerprints[f"initialization_{component}_source"] = (
                _checkpoint_path_fingerprint(path)
            )
            continue
        if not isinstance(source, dict) or not isinstance(source.get("hub"), dict):
            raise ValueError(
                f"initialization.{component}_source must be a path or a hub mapping"
            )
        hub = source["hub"]
        checkpoint_kwargs: dict[str, Any] = {}
        if hub.get("allow_patterns") is not None:
            checkpoint_kwargs["allow_patterns"] = tuple(
                str(pattern) for pattern in hub["allow_patterns"]
            )
        spec = HubCheckpointSpec(
            repo_id=str(hub.get("repo_id") or ""),
            revision=str(hub.get("revision") or ""),
            family=family,
            **checkpoint_kwargs,
        )
        manifest = (
            artifacts
            / "initialization_sources"
            / f"{component}_checkpoint.json"
        )
        resolved[component] = f"@checkpoint:{component}"
        hub_specs[component] = spec
        manifests[component] = manifest
    return resolved, hub_specs, manifests, fingerprints


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"experiment.{key} must be a mapping")
    return value


def _synthetic_count(synthetic: dict[str, Any], split: str) -> int:
    value = synthetic.get(f"{split}_count")
    return int(synthetic.get("count", 0) if value is None else value)


def _validate_spec(raw: dict[str, Any], repo_root: Path) -> tuple[str, Path, Path]:
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("experiment schema_version must be 1")
    name = str(raw.get("name") or "")
    if not _NAME.fullmatch(name):
        raise ValueError("experiment name must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    output_root = _resolve_path(repo_root, raw.get("output_root") or f"outputs/{name}")
    blueprint = _resolve_path(
        repo_root,
        raw.get("blueprint") or "configs/sub1b_architecture.yaml",
    )
    if not blueprint.is_file():
        raise ValueError(f"experiment blueprint does not exist: {blueprint}")
    synthetic = _require_mapping(raw, "synthetic")
    if not bool(synthetic.get("enabled", True)):
        raise ValueError("synthetic.enabled=false is not supported by the end-to-end schema")
    train_count = _synthetic_count(synthetic, "train")
    heldout_count = _synthetic_count(synthetic, "heldout")
    difficulty = int(synthetic.get("difficulty_level", 0))
    train_seed = int(synthetic.get("train_seed", 0))
    heldout_seed = int(synthetic.get("heldout_seed", 0))
    cases = synthetic.get("cases")
    if min(train_count, heldout_count) <= 0 or not 1 <= difficulty <= 5:
        raise ValueError(
            "synthetic train/heldout counts must be positive and "
            "difficulty_level within [1, 5]"
        )
    if train_seed == heldout_seed:
        raise ValueError("synthetic train_seed and heldout_seed must differ")
    if not isinstance(cases, list) or not cases:
        raise ValueError("synthetic.cases must be a non-empty list")
    synth_config = _resolve_path(
        repo_root,
        synthetic.get("config") or "configs/synth_data.yaml",
    )
    if not synth_config.is_file():
        raise ValueError(f"synthetic config does not exist: {synth_config}")
    data = _require_mapping(raw, "data")
    if not isinstance(data.get("components"), list) or not data["components"]:
        raise ValueError("data.components must be a non-empty list")
    sequence_teacher = raw.get("sequence_teacher") or {}
    if not isinstance(sequence_teacher, dict):
        raise ValueError("experiment.sequence_teacher must be a mapping")
    if bool(sequence_teacher.get("enabled", False)):
        if not str(sequence_teacher.get("model") or "").strip():
            raise ValueError("enabled sequence_teacher requires a model")
        for field in ("min_score", "min_acceptance_rate", "target_probability"):
            value = float(sequence_teacher.get(field, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"sequence_teacher.{field} must be within [0, 1]")
        if sequence_teacher.get("target_format", "answer") not in {"answer", "response"}:
            raise ValueError("sequence_teacher.target_format must be answer or response")
    initialization = raw.get("initialization") or {}
    if not isinstance(initialization, dict):
        raise ValueError("experiment.initialization must be a mapping")
    if int(initialization.get("seed", 0)) < 0:
        raise ValueError("initialization.seed must be non-negative")
    return name, output_root, blueprint


def _component_specs(
    raw: dict[str, Any],
    repo_root: Path,
    synthetic_udd: Path,
    *,
    synthetic_enabled: bool,
    acquired_paths: dict[str, Path],
) -> tuple[MixtureComponent, ...]:
    components = []
    for item in raw["data"]["components"]:
        if not isinstance(item, dict):
            raise ValueError("every data component must be a mapping")
        name = str(item.get("name") or "")
        configured_path = str(item.get("path") or "")
        hub = item.get("hub")
        if configured_path and hub:
            raise ValueError(f"data component {name!r} cannot set both path and hub")
        if hub:
            if name not in acquired_paths:
                raise ValueError(f"data component {name!r} has no compiled Hub acquisition")
            path = acquired_paths[name]
        elif configured_path == "@synthetic":
            if not synthetic_enabled:
                raise ValueError("@synthetic requires synthetic.enabled=true")
            path = synthetic_udd
        else:
            path = _resolve_path(repo_root, configured_path)
            if not path.is_dir():
                raise ValueError(f"data component path does not exist: {path}")
        components.append(
            MixtureComponent(
                name=name,
                path=str(path),
                weight=float(item.get("weight", 0.0)),
                fold=(str(item["fold"]) if item.get("fold") is not None else None),
            )
        )
    components = validate_components(components)
    supplied_total = sum(float(item.get("weight", 0.0)) for item in raw["data"]["components"])
    if abs(supplied_total - 1.0) > 1e-6:
        raise ValueError(f"data component weights sum to {supplied_total:.6f}, expected 1.0")
    return components


def _resolved_blueprint(
    blueprint_path: Path,
    components: Iterable[MixtureComponent],
    *,
    tiny: bool,
    tiny_vocab_size: int,
    sequence_target_probability: float,
    sequence_target_min_score: float,
    sequence_target_seed: int,
) -> dict[str, Any]:
    blueprint = copy.deepcopy(load_blueprint(blueprint_path))
    if tiny:
        if tiny_vocab_size < 260:
            raise ValueError("tiny tokenizer vocab_size must be at least 260")
        tiny_config = asdict(StudentConfig.tiny(vocab_size=tiny_vocab_size))
        blueprint["student"] = {
            "parameter_budget": "<1B",
            **tiny_config,
        }
        blueprint["tokenizer"]["vocab_size"] = tiny_config["language"]["vocab_size"]
        pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
        pipeline["max_text_tokens"] = 128
        pipeline["max_image_long_side"] = tiny_config["vision"]["image_size"]
        pipeline["rotation_probability"] = 0.0
        distillation = blueprint["training"]["pretraining"]["distillation"]
        distillation["logit_top_k"] = min(
            int(distillation["logit_top_k"]),
            tiny_config["language"]["vocab_size"] - 1,
        )
        distillation["vision_layer_pairs"] = []
        distillation["language_layer_pairs"] = []
        blueprint["name"] = f"{blueprint['name']}-tiny"
        blueprint["budget"]["target_parameters"] = estimate_parameters(blueprint)["total"]
    weights = {component.name: component.weight for component in components}
    pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
    pipeline["balance_by"] = "component"
    pipeline["group_weights"] = weights
    blueprint["training"]["pretraining"]["data_mix"] = weights
    sequence_targets = blueprint["training"]["pretraining"]["distillation"][
        "sequence_targets"
    ]
    sequence_targets["probability"] = sequence_target_probability
    sequence_targets["min_score"] = sequence_target_min_score
    sequence_targets["seed"] = sequence_target_seed
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise ValueError("resolved blueprint is invalid:\n" + "\n".join(errors))
    return blueprint


def _add_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_experiment_plan(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    python: str,
) -> ExperimentPlan:
    """Load, validate, and compile one experiment YAML into an ordered stage DAG."""

    repo_root = Path(repo_root).resolve()
    config_path = _resolve_path(repo_root, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment root must be a mapping")
    name, output_root, blueprint_path = _validate_spec(raw, repo_root)
    synthetic = raw["synthetic"]
    synth_config_path = _resolve_path(
        repo_root,
        synthetic.get("config") or "configs/synth_data.yaml",
    )
    input_fingerprints: dict[str, Any] = {
        "synthetic_config": _file_fingerprint(synth_config_path),
    }
    configured_replay = (
        ((raw.get("posttraining") or {}).get("rlvr") or {}).get(
            "replay_samples"
        )
    )
    if configured_replay:
        input_fingerprints["rlvr_replay_samples"] = _file_fingerprint(
            _resolve_path(repo_root, configured_replay)
        )
    evaluation_spec = raw.get("evaluation") or {}
    for key in ("baseline_evaluation", "monolingual_control_evaluation"):
        if evaluation_spec.get(key):
            input_fingerprints[key] = _evaluation_fingerprint(
                _resolve_path(repo_root, evaluation_spec[key])
            )
    synthetic_enabled = bool(synthetic.get("enabled", True))
    artifacts = output_root / "artifacts"
    train_cases = artifacts / "synthetic" / "train"
    heldout_cases = artifacts / "synthetic" / "heldout"
    synthetic_udd_root = artifacts / "data" / "synthetic_train"
    synthetic_udd = synthetic_udd_root / "hf"
    mixed_data = artifacts / "data" / "mixture"
    sample_dir = artifacts / "samples"
    train_samples = sample_dir / "train.jsonl"
    heldout_samples = sample_dir / "heldout.jsonl"
    tokenizer_dir = artifacts / "tokenizer"
    initial_dir = artifacts / "initial"
    pretrain_dir = artifacts / "pretrain"
    sft_dir = artifacts / "sft"
    rlvr_dir = artifacts / "rlvr"
    eval_dir = artifacts / "evaluation"
    leakage_report = artifacts / "data" / "split_leakage.json"
    resolved_blueprint_path = output_root / "resolved_blueprint.yaml"
    component_root = artifacts / "data" / "components"
    acquired_paths: dict[str, Path] = {}
    hub_specs: dict[str, HubComponentSpec] = {}
    for item in raw["data"]["components"]:
        hub = item.get("hub") if isinstance(item, dict) else None
        if not hub:
            continue
        if not isinstance(hub, dict):
            raise ValueError("data component hub must be a mapping")
        name = str(item.get("name") or "")
        spec = HubComponentSpec(
            repo_id=str(hub.get("repo_id") or ""),
            revision=str(hub.get("revision") or ""),
            split=str(hub.get("split") or "train"),
            config_name=(
                str(hub["config_name"]) if hub.get("config_name") is not None else None
            ),
            fold=(str(hub["fold"]) if hub.get("fold") is not None else None),
            sources=tuple(str(value) for value in (hub.get("sources") or [])),
            tasks=tuple(str(value) for value in (hub.get("tasks") or [])),
            languages=tuple(str(value) for value in (hub.get("languages") or [])),
            max_rows=(int(hub["max_rows"]) if hub.get("max_rows") is not None else None),
            seed=int(hub.get("seed", 7)),
            decode_checks=int(hub.get("decode_checks", 16)),
        )
        hub_specs[name] = spec
        acquired_paths[name] = component_root / name
    components = _component_specs(
        raw,
        repo_root,
        synthetic_udd,
        synthetic_enabled=synthetic_enabled,
        acquired_paths=acquired_paths,
    )
    initialization = raw.get("initialization") or {}
    (
        initialization_sources,
        checkpoint_specs,
        checkpoint_manifests,
        checkpoint_fingerprints,
    ) = _checkpoint_sources(
        initialization,
        repo_root,
        artifacts,
    )
    input_fingerprints.update(checkpoint_fingerprints)
    tokenizer = raw.get("tokenizer") or {}
    sequence_teacher = raw.get("sequence_teacher") or {}
    sequence_teacher_enabled = bool(sequence_teacher.get("enabled", False))
    if sequence_teacher.get("predictions"):
        predictions_input = _resolve_path(repo_root, sequence_teacher["predictions"])
        input_fingerprints["sequence_teacher_predictions"] = _file_fingerprint(
            predictions_input
        )
    token_map = initialization.get("token_map")
    if token_map:
        input_fingerprints["initialization_token_map"] = _file_fingerprint(
            _resolve_path(repo_root, token_map)
        )
    blueprint = _resolved_blueprint(
        blueprint_path,
        components,
        tiny=bool(initialization.get("tiny", False)),
        tiny_vocab_size=int(tokenizer.get("vocab_size") or 512),
        sequence_target_probability=(
            float(sequence_teacher.get("target_probability", 0.0))
            if sequence_teacher_enabled
            else 0.0
        ),
        sequence_target_min_score=float(sequence_teacher.get("min_score", 0.8)),
        sequence_target_seed=int(sequence_teacher.get("seed", 7)),
    )
    for component, allowed in (
        ("vision", {"student", "siglip"}),
        ("language", {"student", "llama"}),
    ):
        family = str(
            initialization.get(f"{component}_family")
            or (
                checkpoint_specs[component].family
                if component in checkpoint_specs
                else "student" if component in initialization_sources else ""
            )
        )
        if family and family not in allowed:
            raise ValueError(
                f"initialization.{component}_family must be one of "
                f"{sorted(allowed)}"
            )
    arms = {
        str(arm["id"]): arm
        for arm in blueprint["initialization_arms"]
    }
    arm_id = str(initialization.get("arm") or "I0_random")
    if arm_id not in arms:
        raise ValueError(f"unknown initialization arm {arm_id!r}")
    required_sources = [
        component
        for component in ("vision", "language")
        if float(arms[arm_id].get(f"{component}_transfer", 0.0)) > 0
        and component not in initialization_sources
    ]
    if required_sources:
        raise ValueError(
            f"initialization arm {arm_id!r} requires sources for "
            f"{required_sources}"
        )
    runtime = raw.get("runtime") or {}
    device = str(runtime.get("device") or "auto")
    stages: list[ExperimentStage] = []

    def script(name: str) -> str:
        return str((repo_root / "scripts" / name).resolve())

    checkpoint_stage_names: list[str] = []
    for component, spec in checkpoint_specs.items():
        stage_name = f"acquire_{component}_checkpoint"
        manifest = checkpoint_manifests[component]
        stages.append(
            ExperimentStage(
                stage_name,
                (
                    python,
                    script("acquire_student_checkpoint.py"),
                    "--repo-id",
                    spec.repo_id,
                    "--revision",
                    spec.revision,
                    "--family",
                    spec.family,
                    "--output",
                    str(manifest),
                ),
                (),
                (Artifact(str(manifest), "checkpoint_manifest"),),
            )
        )
        checkpoint_stage_names.append(stage_name)

    if synthetic_enabled:
        common = [
            python,
            script("make_realistic_cases.py"),
            "--config",
            str(_resolve_path(repo_root, synthetic.get("config") or "configs/synth_data.yaml")),
            "--only",
            *[str(case) for case in synthetic["cases"]],
            "--difficulty-level",
            str(int(synthetic["difficulty_level"])),
        ]
        if bool(synthetic.get("no_degrade", False)):
            common.append("--no-degrade")
        for split, seed, count, output in (
            (
                "train",
                synthetic["train_seed"],
                _synthetic_count(synthetic, "train"),
                train_cases,
            ),
            (
                "heldout",
                synthetic["heldout_seed"],
                _synthetic_count(synthetic, "heldout"),
                heldout_cases,
            ),
        ):
            stages.append(
                ExperimentStage(
                    f"synthetic_{split}",
                    tuple(
                        [
                            *common,
                            "--count",
                            str(count),
                            "--seed",
                            str(int(seed)),
                            "--split-name",
                            split,
                            "--out",
                            str(output),
                        ]
                    ),
                    (),
                    (
                        Artifact(str(output / "index.json")),
                        Artifact(str(output / "gen_config.json")),
                    ),
                )
            )
        stages.append(
            ExperimentStage(
                "validate_synthetic_splits",
                (
                    python,
                    script("validate_synth_splits.py"),
                    "--split",
                    f"train={train_cases}",
                    "--split",
                    f"heldout={heldout_cases}",
                    "--output",
                    str(leakage_report),
                ),
                ("synthetic_train", "synthetic_heldout"),
                (Artifact(str(leakage_report)),),
            )
        )
        variant = str(synthetic.get("variant") or "clean")
        stages.extend(
            [
                ExperimentStage(
                    "build_synthetic_udd",
                    (
                        python,
                        script("build_udd_synthetic.py"),
                        "--root",
                        str(train_cases),
                        "--out",
                        str(synthetic_udd_root),
                        "--variant",
                        variant,
                    ),
                    ("validate_synthetic_splits",),
                    (Artifact(str(synthetic_udd), "directory"),),
                ),
                ExperimentStage(
                    "build_train_samples",
                    (
                        python,
                        script("build_realistic_benchmark.py"),
                        "--root",
                        str(train_cases),
                        "--variant",
                        variant,
                        "--out",
                        str(train_samples),
                    ),
                    ("validate_synthetic_splits",),
                    (Artifact(str(train_samples)),),
                ),
                ExperimentStage(
                    "build_heldout_samples",
                    (
                        python,
                        script("build_realistic_benchmark.py"),
                        "--root",
                        str(heldout_cases),
                        "--variant",
                        variant,
                        "--out",
                        str(heldout_samples),
                    ),
                    ("validate_synthetic_splits",),
                    (Artifact(str(heldout_samples)),),
                ),
            ]
        )

    for name, spec in hub_specs.items():
        output = acquired_paths[name]
        command = [
            python,
            script("acquire_student_data.py"),
            "--repo-id",
            spec.repo_id,
            "--revision",
            spec.revision,
            "--split",
            spec.split,
            "--seed",
            str(spec.seed),
            "--decode-checks",
            str(spec.decode_checks),
            "--output",
            str(output),
        ]
        _add_optional(command, "--config-name", spec.config_name)
        _add_optional(command, "--fold", spec.fold)
        _add_optional(command, "--max-rows", spec.max_rows)
        for value in spec.sources:
            command.extend(["--source", value])
        for value in spec.tasks:
            command.extend(["--task", value])
        for value in spec.languages:
            command.extend(["--language", value])
        stages.append(
            ExperimentStage(
                f"acquire_component_{name}",
                tuple(command),
                (),
                (
                    Artifact(str(output), "directory"),
                    Artifact(str(output / "component_manifest.json")),
                ),
            )
        )

    mix_command = [python, script("mix_student_data.py")]
    mix_dependencies: list[str] = []
    for component in components:
        mix_command.extend(["--component", f"{component.name}={component.path}"])
        mix_command.extend(["--weight", f"{component.name}={component.weight:.17g}"])
        if component.fold:
            mix_command.extend(["--fold", f"{component.name}={component.fold}"])
        if Path(component.path) == synthetic_udd:
            mix_dependencies.append("build_synthetic_udd")
        if component.name in hub_specs:
            mix_dependencies.append(f"acquire_component_{component.name}")
    mix_command.extend(["--output", str(mixed_data)])
    stages.append(
        ExperimentStage(
            "mix_pretraining_data",
            tuple(mix_command),
            tuple(mix_dependencies),
            (Artifact(str(mixed_data / "mixture_manifest.json")),),
        )
    )

    training_data = mixed_data
    training_data_dependency = "mix_pretraining_data"
    if sequence_teacher_enabled:
        teacher_requests = artifacts / "data" / "teacher_requests"
        generated_predictions = artifacts / "data" / "teacher_predictions.jsonl"
        distilled_data = artifacts / "data" / "distilled_mixture"
        stages.append(
            ExperimentStage(
                "export_teacher_requests",
                (
                    python,
                    script("build_teacher_targets.py"),
                    "export",
                    "--src",
                    str(mixed_data),
                    "--output",
                    str(teacher_requests),
                ),
                ("mix_pretraining_data",),
                (
                    Artifact(str(teacher_requests / "requests.jsonl")),
                    Artifact(str(teacher_requests / "manifest.json")),
                ),
            )
        )
        configured_predictions = sequence_teacher.get("predictions")
        if configured_predictions:
            predictions_path = _resolve_path(repo_root, configured_predictions)
            if not predictions_path.is_file():
                raise ValueError(
                    f"sequence teacher predictions do not exist: {predictions_path}"
                )
            apply_dependencies = ("export_teacher_requests",)
        else:
            predictions_path = generated_predictions
            generate_command = [
                python,
                script("build_teacher_targets.py"),
                "generate",
                "--requests",
                str(teacher_requests / "requests.jsonl"),
                "--output",
                str(predictions_path),
                "--model",
                str(sequence_teacher["model"]),
                "--device",
                str(sequence_teacher.get("device") or device),
                "--dtype",
                str(sequence_teacher.get("dtype") or "bfloat16"),
                "--max-new-tokens",
                str(int(sequence_teacher.get("max_new_tokens", 128))),
                "--temperature",
                str(float(sequence_teacher.get("temperature", 0.0))),
            ]
            stages.append(
                ExperimentStage(
                    "generate_teacher_predictions",
                    tuple(generate_command),
                    ("export_teacher_requests",),
                    (
                        Artifact(str(predictions_path)),
                        Artifact(str(predictions_path) + ".manifest.json"),
                    ),
                )
            )
            apply_dependencies = (
                "export_teacher_requests",
                "generate_teacher_predictions",
            )
        stages.append(
            ExperimentStage(
                "apply_teacher_targets",
                (
                    python,
                    script("build_teacher_targets.py"),
                    "apply",
                    "--src",
                    str(mixed_data),
                    "--requests",
                    str(teacher_requests / "requests.jsonl"),
                    "--predictions",
                    str(predictions_path),
                    "--output",
                    str(distilled_data),
                    "--min-score",
                    str(float(sequence_teacher.get("min_score", 0.8))),
                    "--min-acceptance-rate",
                    str(float(sequence_teacher.get("min_acceptance_rate", 0.0))),
                    "--target-format",
                    str(sequence_teacher.get("target_format") or "answer"),
                ),
                apply_dependencies,
                (
                    Artifact(str(distilled_data), "directory"),
                    Artifact(str(distilled_data / "teacher_target_manifest.json")),
                ),
            )
        )
        training_data = distilled_data
        training_data_dependency = "apply_teacher_targets"

    tokenizer_command = [
        python,
        script("train_student_tokenizer.py"),
        "--config",
        str(resolved_blueprint_path),
        "--src",
        str(training_data),
        "--output",
        str(tokenizer_dir),
        "--min-frequency",
        str(int(tokenizer.get("min_frequency", 2))),
    ]
    _add_optional(tokenizer_command, "--vocab-size", tokenizer.get("vocab_size"))
    if bool(tokenizer.get("no_progress", True)):
        tokenizer_command.append("--no-progress")
    stages.append(
        ExperimentStage(
            "train_tokenizer",
            tuple(tokenizer_command),
            (training_data_dependency,),
            (
                Artifact(str(tokenizer_dir / "tokenizer.json")),
                Artifact(str(tokenizer_dir / "tokenizer_config.json")),
            ),
        )
    )

    init_command = [
        python,
        script("build_sub1b_student.py"),
        "--config",
        str(resolved_blueprint_path),
        "--device",
        str(initialization.get("device") or ("cpu" if initialization.get("tiny") else device)),
        "--seed",
        str(int(initialization.get("seed", 0))),
        "--init-arm",
        str(initialization.get("arm") or "I0_random"),
        "--save",
        str(initial_dir),
    ]
    if bool(initialization.get("tiny", False)):
        init_command.extend(
            [
                "--tiny",
                "--tiny-vocab-size",
                str(blueprint["student"]["language"]["vocab_size"]),
            ]
        )
    if bool(initialization.get("allow_full_memory", False)):
        init_command.append("--allow-full-memory")
    for component in ("vision", "language"):
        source = initialization_sources.get(component)
        if source is not None:
            init_command.extend([f"--{component}-source", source])
        family = (
            initialization.get(f"{component}_family")
            or (
                checkpoint_specs[component].family
                if component in checkpoint_specs
                else None
            )
        )
        if family is not None:
            init_command.extend([f"--{component}-family", str(family)])
    if token_map is not None:
        init_command.extend(
            ["--token-map", str(_resolve_path(repo_root, token_map))]
        )
    stages.append(
        ExperimentStage(
            "initialize_student",
            tuple(init_command),
            ("train_tokenizer", *checkpoint_stage_names),
            (
                Artifact(str(initial_dir / "student_config.json")),
                Artifact(str(initial_dir / "model.pt")),
                Artifact(str(initial_dir / "metadata.json")),
            ),
        )
    )

    pretraining = raw.get("pretraining") or {}
    pretrain_command = [
        python,
        script("pretrain_student.py"),
        "--config",
        str(resolved_blueprint_path),
        "--src",
        str(training_data),
        "--tokenizer",
        str(tokenizer_dir),
        "--student-checkpoint",
        str(initial_dir),
        "--output",
        str(pretrain_dir),
        "--device",
        device,
        "--eval-group-by",
        str(pretraining.get("eval_group_by") or "component"),
    ]
    for key, flag in (
        ("epochs", "--epochs"),
        ("max_steps", "--max-steps"),
        ("batch_size", "--batch-size"),
        ("num_workers", "--num-workers"),
    ):
        _add_optional(pretrain_command, flag, pretraining.get(key))
    if pretraining.get("teacher_checkpoint"):
        pretrain_command.extend(
            [
                "--teacher-checkpoint",
                str(_resolve_path(repo_root, pretraining["teacher_checkpoint"])),
            ]
        )
    if bool(pretraining.get("no_grounding", False)):
        pretrain_command.append("--no-grounding")
    stages.append(
        ExperimentStage(
            "pretrain",
            tuple(pretrain_command),
            ("initialize_student",),
            (Artifact(str(pretrain_dir / "latest_checkpoint.txt")),),
        )
    )

    posttraining = raw.get("posttraining") or {}
    sft = posttraining.get("sft") or {}
    sft_command = [
        python,
        script("posttrain_student.py"),
        "sft",
        "--config",
        str(resolved_blueprint_path),
        "--samples",
        str(train_samples),
        "--tokenizer",
        str(tokenizer_dir),
        "--checkpoint",
        "@student:pretrain",
        "--output",
        str(sft_dir),
        "--device",
        device,
    ]
    for key, flag in (("max_steps", "--max-steps"), ("num_workers", "--num-workers")):
        _add_optional(sft_command, flag, sft.get(key))
    _add_optional(sft_command, "--target-mode", sft.get("target_mode"))
    stages.append(
        ExperimentStage(
            "sft",
            tuple(sft_command),
            ("pretrain", "build_train_samples"),
            (Artifact(str(sft_dir / "latest_checkpoint.txt")),),
        )
    )

    rlvr = posttraining.get("rlvr") or {}
    rlvr_command = [
        python,
        script("posttrain_student.py"),
        "rlvr",
        "--config",
        str(resolved_blueprint_path),
        "--samples",
        str(train_samples),
        "--tokenizer",
        str(tokenizer_dir),
        "--checkpoint",
        "@student:sft",
        "--output",
        str(rlvr_dir),
        "--device",
        device,
    ]
    _add_optional(rlvr_command, "--max-steps", rlvr.get("max_steps"))
    _add_optional(
        rlvr_command,
        "--replay-every-steps",
        rlvr.get("replay_every_steps"),
    )
    _add_optional(
        rlvr_command,
        "--replay-loss-coefficient",
        rlvr.get("replay_loss_coefficient"),
    )
    if rlvr.get("replay_samples"):
        rlvr_command.extend(
            [
                "--replay-samples",
                str(_resolve_path(repo_root, rlvr["replay_samples"])),
            ]
        )
    stages.append(
        ExperimentStage(
            "rlvr",
            tuple(rlvr_command),
            ("sft",),
            (Artifact(str(rlvr_dir / "latest_checkpoint.txt")),),
        )
    )

    evaluation = raw.get("evaluation") or {}
    eval_command = [
        python,
        script("eval_student.py"),
        "--config",
        str(resolved_blueprint_path),
        "--split",
        f"train={train_samples}",
        "--split",
        f"heldout={heldout_samples}",
        "--tokenizer",
        str(tokenizer_dir),
        "--checkpoint",
        "@student:rlvr",
        "--output",
        str(eval_dir),
        "--device",
        device,
        "--precision",
        str(evaluation.get("precision") or "bfloat16"),
        "--max-new-tokens",
        str(int(evaluation.get("max_new_tokens", 128))),
        "--seed",
        str(int(evaluation.get("seed", 0))),
    ]
    _add_optional(eval_command, "--max-samples", evaluation.get("max_samples"))
    for key, flag in (
        ("baseline_evaluation", "--baseline-evaluation"),
        (
            "monolingual_control_evaluation",
            "--monolingual-control-evaluation",
        ),
    ):
        if evaluation.get(key):
            eval_command.extend(
                [flag, str(_resolve_path(repo_root, evaluation[key]))]
            )
    for key, flag in (
        ("wandb_project", "--wandb-project"),
        ("wandb_entity", "--wandb-entity"),
        ("wandb_run", "--wandb-run"),
        ("wandb_group", "--wandb-group"),
    ):
        _add_optional(eval_command, flag, evaluation.get(key))
    tags = evaluation.get("wandb_tags") or []
    if tags:
        eval_command.extend(["--wandb-tags", *[str(tag) for tag in tags]])
    stages.append(
        ExperimentStage(
            "evaluate",
            tuple(eval_command),
            ("rlvr", "build_train_samples", "build_heldout_samples"),
            (
                Artifact(str(eval_dir / "manifest.json")),
                Artifact(str(eval_dir / "comparison.json")),
                Artifact(str(eval_dir / "gates.json")),
            ),
        )
    )

    input_fingerprints["python_source"] = _source_fingerprint(
        repo_root,
        stages,
    )
    fingerprint = _fingerprint(
        {
            "config": raw,
            "blueprint": blueprint,
            "components": [asdict(component) for component in components],
            "input_fingerprints": input_fingerprints,
        }
    )
    return ExperimentPlan(
        name=name,
        root=str(output_root),
        blueprint=str(resolved_blueprint_path),
        resolved_blueprint=blueprint,
        raw_spec=raw,
        components=components,
        stages=tuple(stages),
        fingerprint=fingerprint,
        input_fingerprints=input_fingerprints,
    )


def _checkpoint_student(output: Path) -> Path:
    pointer = output / "latest_checkpoint.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"missing checkpoint pointer {pointer}")
    checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    student = checkpoint / "student"
    if not (student / "model.pt").is_file():
        raise FileNotFoundError(f"checkpoint student is incomplete: {student}")
    return student


def _resolve_command(command: tuple[str, ...], root: Path) -> list[str]:
    stage_outputs = {
        "pretrain": root / "artifacts" / "pretrain",
        "sft": root / "artifacts" / "sft",
        "rlvr": root / "artifacts" / "rlvr",
    }
    resolved = []
    for argument in command:
        if argument.startswith("@student:"):
            stage = argument.split(":", 1)[1]
            if stage not in stage_outputs:
                raise ValueError(f"unknown checkpoint placeholder {argument!r}")
            resolved.append(str(_checkpoint_student(stage_outputs[stage])))
        elif argument.startswith("@checkpoint:"):
            component = argument.split(":", 1)[1]
            if component not in {"vision", "language"}:
                raise ValueError(f"unknown checkpoint placeholder {argument!r}")
            manifest = (
                root
                / "artifacts"
                / "initialization_sources"
                / f"{component}_checkpoint.json"
            )
            resolved.append(str(checkpoint_path_from_manifest(manifest)))
        else:
            resolved.append(argument)
    return resolved


def _with_training_resume(
    command: list[str],
    stage: str,
    root: Path,
    *,
    eligible: bool,
) -> list[str]:
    if not eligible or stage not in _CHECKPOINT_STAGES or "--resume" in command:
        return command
    pointer = root / "artifacts" / stage / "latest_checkpoint.txt"
    if pointer.is_file():
        return [*command, "--resume", "latest"]
    return command


def _owned_output_paths(
    stage: ExperimentStage,
    command: list[str],
    root: Path,
) -> tuple[Path, ...]:
    candidates = {Path(artifact.path).resolve() for artifact in stage.artifacts}
    for index, argument in enumerate(command[:-1]):
        if argument in _OUTPUT_FLAGS:
            candidates.add(Path(command[index + 1]).resolve())
    owned = []
    resolved_root = root.resolve()
    for path in sorted(candidates, key=lambda value: len(value.parts)):
        if path == resolved_root or resolved_root not in path.parents:
            raise ValueError(
                f"stage {stage.name!r} output is outside experiment root: {path}"
            )
        if any(parent == path or parent in path.parents for parent in owned):
            continue
        owned.append(path)
    return tuple(owned)


def _clear_stale_outputs(
    stage: ExperimentStage,
    command: list[str],
    root: Path,
) -> list[str]:
    removed = []
    for path in _owned_output_paths(stage, command, root):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
        elif path.exists():
            path.unlink()
            removed.append(str(path))
        for suffix in (".manifest.json", ".tmp"):
            sidecar = Path(str(path) + suffix)
            if sidecar.is_file():
                sidecar.unlink()
                removed.append(str(sidecar))
    return removed


class ExperimentRunner:
    """Execute a compiled plan with signature-aware, artifact-checked resume."""

    def __init__(self, plan: ExperimentPlan, *, repo_root: str | Path):
        self.plan = plan
        self.repo_root = Path(repo_root).resolve()
        self.root = Path(plan.root)
        self.state_dir = self.root / "state" / "stages"
        self.log_dir = self.root / "logs"

    def _state_path(self, stage: str) -> Path:
        return self.state_dir / f"{stage}.json"

    def _load_state(self, stage: str) -> dict[str, Any] | None:
        path = self._state_path(stage)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _write_static_manifests(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "resolved_blueprint.yaml").write_text(
            yaml.safe_dump(self.plan.resolved_blueprint, sort_keys=False),
            encoding="utf-8",
        )
        _atomic_write_json(self.root / "experiment_plan.json", self.plan.to_dict())
        _atomic_write_json(self.root / "experiment_spec.json", self.plan.raw_spec)

    def signatures(self) -> dict[str, str]:
        signatures: dict[str, str] = {}
        for stage in self.plan.stages:
            signatures[stage.name] = stage.signature(self.plan.fingerprint, signatures)
        return signatures

    def _is_complete(self, stage: ExperimentStage, signature: str) -> bool:
        state = self._load_state(stage.name)
        return bool(
            state
            and state.get("status") == "completed"
            and state.get("signature") == signature
            and stage.artifacts_valid()
        )

    def _selection(self, start: str | None, stop: str | None) -> tuple[ExperimentStage, ...]:
        names = self.plan.stage_names
        if start is not None and start not in names:
            raise ValueError(f"unknown --from-stage {start!r}")
        if stop is not None and stop not in names:
            raise ValueError(f"unknown --to-stage {stop!r}")
        first = names.index(start) if start else 0
        last = names.index(stop) if stop else len(names) - 1
        if first > last:
            raise ValueError("--from-stage occurs after --to-stage")
        return self.plan.stages[first : last + 1]

    def run(
        self,
        *,
        dry_run: bool = False,
        resume: bool = True,
        start: str | None = None,
        stop: str | None = None,
    ) -> dict[str, Any]:
        signatures = self.signatures()
        selected = self._selection(start, stop)
        if dry_run:
            return {
                "dry_run": True,
                "fingerprint": self.plan.fingerprint,
                "stages": [
                    {
                        "name": stage.name,
                        "command": list(stage.command),
                        "dependencies": list(stage.dependencies),
                        "would_skip": self._is_complete(stage, signatures[stage.name]),
                    }
                    for stage in selected
                ],
            }

        self._write_static_manifests()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        outcomes = []
        selected_names = {stage.name for stage in selected}
        for stage in selected:
            for dependency in stage.dependencies:
                dependency_stage = next(
                    candidate for candidate in self.plan.stages if candidate.name == dependency
                )
                if dependency not in selected_names and not self._is_complete(
                    dependency_stage,
                    signatures[dependency],
                ):
                    raise RuntimeError(
                        f"stage {stage.name!r} requires incomplete dependency {dependency!r}"
                    )
            signature = signatures[stage.name]
            if resume and self._is_complete(stage, signature):
                outcomes.append({"stage": stage.name, "status": "skipped"})
                continue
            prior_state = self._load_state(stage.name)
            resolved_command = _resolve_command(stage.command, self.root)
            resume_interrupted = bool(
                resume
                and prior_state
                and prior_state.get("signature") == signature
                and prior_state.get("status") in {"running", "failed"}
            )
            signature_changed = bool(
                prior_state
                and prior_state.get("signature") != signature
            )
            invalid_completed_artifacts = bool(
                prior_state
                and prior_state.get("status") == "completed"
                and not stage.artifacts_valid()
            )
            clear_outputs = (
                signature_changed
                or invalid_completed_artifacts
                or (
                    resume_interrupted
                    and stage.name not in _CHECKPOINT_STAGES
                )
            )
            removed_outputs = (
                _clear_stale_outputs(stage, resolved_command, self.root)
                if clear_outputs
                else []
            )
            if clear_outputs:
                resume_interrupted = False
            command = _with_training_resume(
                resolved_command,
                stage.name,
                self.root,
                eligible=resume_interrupted,
            )
            started = time.time()
            state = {
                "stage": stage.name,
                "status": "running",
                "signature": signature,
                "command": command,
                "started_at_unix": started,
            }
            if removed_outputs:
                state["invalidated_outputs"] = removed_outputs
            _atomic_write_json(self._state_path(stage.name), state)
            log_path = self.log_dir / f"{stage.name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            finished = time.time()
            state.update(
                {
                    "finished_at_unix": finished,
                    "duration_seconds": finished - started,
                    "return_code": return_code,
                    "log": str(log_path),
                    "status": "completed" if return_code == 0 else "failed",
                }
            )
            if return_code == 0 and not stage.artifacts_valid():
                state["status"] = "failed"
                state["error"] = "command succeeded but expected artifacts are missing"
            _atomic_write_json(self._state_path(stage.name), state)
            if state["status"] != "completed":
                raise RuntimeError(
                    f"stage {stage.name!r} failed; inspect {log_path}"
                )
            outcomes.append({"stage": stage.name, "status": "completed"})
        result = {
            "dry_run": False,
            "fingerprint": self.plan.fingerprint,
            "outcomes": outcomes,
        }
        _atomic_write_json(self.root / "run_summary.json", result)
        return result
