"""Compute-matched visual-resolution and latent-token architecture sweeps."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from ..architecture import load_blueprint
from .compute import compute_profile
from .config import StudentConfig
from .sweep import SweepPlan, SweepRunner, compile_sweep_plan


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STAGES = ("pretrain", "sft", "rlvr")


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_yaml(path: Path, value: Any) -> None:
    _atomic_write(path, yaml.safe_dump(value, sort_keys=False))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class ArchitectureProfile:
    id: str
    image_long_side: int
    latent_tokens: int
    compute: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_long_side": self.image_long_side,
            "latent_tokens": self.latent_tokens,
            "compute": self.compute,
        }


@dataclass(frozen=True)
class ArchitectureComputeBudgets:
    pretrain: int
    sft: int
    rlvr: int
    warmup_pretrain: int
    warmup_sft: int

    def to_dict(self) -> dict[str, int]:
        return {
            "pretrain": self.pretrain,
            "sft": self.sft,
            "rlvr": self.rlvr,
            "warmup_pretrain": self.warmup_pretrain,
            "warmup_sft": self.warmup_sft,
        }


@dataclass(frozen=True)
class ArchitectureSweepPlan:
    name: str
    root: str
    baseline: str
    profiles: tuple[ArchitectureProfile, ...]
    budgets: ArchitectureComputeBudgets
    overshoot_tolerance_fraction: float
    sweep: SweepPlan
    fingerprint: str
    raw_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "root": self.root,
            "baseline": self.baseline,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "budgets": self.budgets.to_dict(),
            "overshoot_tolerance_fraction": (
                self.overshoot_tolerance_fraction
            ),
            "sweep": self.sweep.to_dict(),
            "fingerprint": self.fingerprint,
        }


def _reference_budgets(
    profile: dict[str, Any],
    raw: dict[str, Any],
    latent_tokens: int,
) -> ArchitectureComputeBudgets:
    reference = raw.get("reference_budget") or {}
    text_tokens = int(reference.get("text_tokens", 0))
    pretrain_tokens = int(
        reference.get("pretraining_effective_tokens", 0)
    )
    sft_tokens = int(reference.get("sft_effective_tokens", 0))
    rlvr_steps = int(reference.get("rlvr_steps", 0))
    if min(text_tokens, pretrain_tokens, sft_tokens, rlvr_steps) <= 0:
        raise ValueError("architecture reference budgets must be positive")
    effective_tokens = text_tokens + latent_tokens
    pretrain_samples = math.ceil(pretrain_tokens / effective_tokens)
    sft_samples = math.ceil(sft_tokens / effective_tokens)
    pretrain = (
        int(profile["training_flops_per_sample"]) * pretrain_samples
    )
    sft = int(profile["training_flops_per_sample"]) * sft_samples
    rlvr = int(profile["rlvr"]["expected_total_per_step"]) * rlvr_steps
    pretrain_warmup = Fraction(
        str(reference.get("pretraining_warmup_fraction", 0))
    )
    sft_warmup = Fraction(
        str(reference.get("sft_warmup_fraction", 0))
    )
    if not 0 <= pretrain_warmup < 1 or not 0 <= sft_warmup < 1:
        raise ValueError("architecture warmup fractions must be within [0, 1)")
    return ArchitectureComputeBudgets(
        pretrain=pretrain,
        sft=sft,
        rlvr=rlvr,
        warmup_pretrain=pretrain * pretrain_warmup.numerator
        // pretrain_warmup.denominator,
        warmup_sft=sft * sft_warmup.numerator
        // sft_warmup.denominator,
    )


def _compute_patches(
    budgets: ArchitectureComputeBudgets,
) -> list[dict[str, Any]]:
    return [
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/epochs",
            "value": None,
        },
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/stop_at_total_tokens",
            "value": False,
        },
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/warmup_student_flops",
            "value": budgets.warmup_pretrain,
        },
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/total_student_flops",
            "value": budgets.pretrain,
        },
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/stop_at_student_flops",
            "value": True,
        },
        {
            "op": "replace",
            "path": "/training/pretraining/optimizer/schedule_unit",
            "value": "student_flops",
        },
        {
            "op": "replace",
            "path": "/training/pretraining/curriculum/unit",
            "value": "training_compute_fraction",
        },
        {
            "op": "replace",
            "path": "/training/posttraining/sft/optimizer/epochs",
            "value": None,
        },
        {
            "op": "replace",
            "path": (
                "/training/posttraining/sft/optimizer/"
                "warmup_student_flops"
            ),
            "value": budgets.warmup_sft,
        },
        {
            "op": "replace",
            "path": (
                "/training/posttraining/sft/optimizer/"
                "total_student_flops"
            ),
            "value": budgets.sft,
        },
        {
            "op": "replace",
            "path": (
                "/training/posttraining/sft/optimizer/"
                "stop_at_student_flops"
            ),
            "value": True,
        },
        {
            "op": "replace",
            "path": "/training/posttraining/sft/optimizer/schedule_unit",
            "value": "student_flops",
        },
        {
            "op": "replace",
            "path": "/training/posttraining/rlvr/optimizer/max_steps",
            "value": None,
        },
        {
            "op": "replace",
            "path": (
                "/training/posttraining/rlvr/optimizer/"
                "total_student_flops"
            ),
            "value": budgets.rlvr,
        },
        {
            "op": "replace",
            "path": (
                "/training/posttraining/rlvr/optimizer/"
                "stop_at_student_flops"
            ),
            "value": True,
        },
    ]


def compile_architecture_sweep(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    python: str,
    compile_root: str | Path | None = None,
) -> ArchitectureSweepPlan:
    """Compile paired architecture arms with one shared three-phase FLOP budget."""

    repo = Path(repo_root).resolve()
    source = _resolve_path(repo, config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
        raise ValueError("architecture sweep must be a schema_version 1 mapping")
    name = str(raw.get("name") or "")
    if not _NAME.fullmatch(name):
        raise ValueError("architecture sweep name must be a safe non-empty name")
    base_sweep_path = _resolve_path(repo, raw.get("base_sweep") or "")
    base_sweep = yaml.safe_load(
        base_sweep_path.read_text(encoding="utf-8")
    ) or {}
    base_experiment_path = _resolve_path(
        repo,
        base_sweep.get("base_experiment") or "",
    )
    base_experiment = yaml.safe_load(
        base_experiment_path.read_text(encoding="utf-8")
    ) or {}
    blueprint_path = _resolve_path(
        repo,
        base_experiment.get("blueprint") or "",
    )
    blueprint = load_blueprint(blueprint_path)
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, list) or len(profiles_raw) < 2:
        raise ValueError("architecture sweep requires at least two profiles")
    seen: set[str] = set()
    profiles: list[ArchitectureProfile] = []
    for item in profiles_raw:
        if not isinstance(item, dict):
            raise ValueError("every architecture profile must be a mapping")
        profile_id = str(item.get("id") or "")
        if not _NAME.fullmatch(profile_id) or profile_id in seen:
            raise ValueError("architecture profile ids must be unique safe names")
        seen.add(profile_id)
        image_long_side = int(item.get("image_long_side", 0))
        latent_tokens = int(item.get("latent_tokens", 0))
        if image_long_side <= 0 or latent_tokens <= 0:
            raise ValueError("architecture dimensions must be positive")
        candidate = copy.deepcopy(blueprint)
        candidate["student"]["vision"]["image_size"] = image_long_side
        candidate["student"]["connector"]["latent_tokens"] = latent_tokens
        candidate["training"]["pretraining"]["input_pipeline"][
            "max_image_long_side"
        ] = image_long_side
        model_config = StudentConfig.from_blueprint(candidate)
        reference = raw.get("reference_budget") or {}
        profile = compute_profile(
            model_config,
            image_long_side=image_long_side,
            text_tokens=int(reference.get("text_tokens", 0)),
            rlvr_prompt_tokens=int(
                reference.get("rlvr_prompt_tokens", 256)
            ),
            rlvr_completion_tokens=int(
                reference.get("rlvr_completion_tokens", 128)
            ),
            rlvr_group_size=int(reference.get("rlvr_group_size", 8)),
            rlvr_replay_every_steps=int(
                reference.get("rlvr_replay_every_steps", 20)
            ),
        )
        profiles.append(
            ArchitectureProfile(
                id=profile_id,
                image_long_side=image_long_side,
                latent_tokens=latent_tokens,
                compute=profile,
            )
        )
    baseline = str(raw.get("baseline") or "")
    if baseline not in seen:
        raise ValueError("architecture baseline must name a configured profile")
    baseline_profile = next(
        profile for profile in profiles if profile.id == baseline
    )
    budgets = _reference_budgets(
        baseline_profile.compute,
        raw,
        baseline_profile.latent_tokens,
    )
    root = _resolve_path(
        repo,
        raw.get("output_root") or f"outputs/sweeps/{name}",
    )
    generated_root = (
        _resolve_path(repo, compile_root)
        if compile_root is not None
        else root / "compiled"
    )
    child = copy.deepcopy(base_sweep)
    child["name"] = name
    child["output_root"] = str(root)
    child["baseline"] = baseline
    shared_blueprint = list(child.get("shared_blueprint_patches") or [])
    shared_blueprint.extend(_compute_patches(budgets))
    shared_blueprint.append(
        {
            "op": "replace",
            "path": (
                "/training/pretraining/input_pipeline/visual_canvas_mode"
            ),
            "value": "fixed_square",
        }
    )
    shared_blueprint.append(
        {
            "op": "replace",
            "path": (
                "/training/pretraining/input_pipeline/aspect_ratio_bucketing"
            ),
            "value": False,
        }
    )
    child["shared_blueprint_patches"] = shared_blueprint
    controls = list(child.get("matched_controls") or [])
    for path in (
        "/training/pretraining/optimizer/total_student_flops",
        "/training/posttraining/sft/optimizer/total_student_flops",
        "/training/posttraining/rlvr/optimizer/total_student_flops",
        "/training/pretraining/input_pipeline/visual_canvas_mode",
        "/training/pretraining/input_pipeline/aspect_ratio_bucketing",
    ):
        controls.append({"document": "blueprint", "path": path})
    child["matched_controls"] = controls
    child["variants"] = [
        {
            "id": profile.id,
            "hypothesis": (
                f"Tests {profile.image_long_side}px input with "
                f"{profile.latent_tokens} visual latents at matched student FLOPs."
            ),
            "experiment_patches": [
                {
                    "op": "add",
                    "path": "/evaluation/wandb_tags/-",
                    "value": f"resolution:{profile.image_long_side}",
                },
                {
                    "op": "add",
                    "path": "/evaluation/wandb_tags/-",
                    "value": f"visual-latents:{profile.latent_tokens}",
                },
                {
                    "op": "add",
                    "path": "/evaluation/wandb_tags/-",
                    "value": "compute-matched-architecture",
                },
            ],
            "blueprint_patches": [
                {
                    "op": "replace",
                    "path": "/student/vision/image_size",
                    "value": profile.image_long_side,
                },
                {
                    "op": "replace",
                    "path": (
                        "/training/pretraining/input_pipeline/"
                        "max_image_long_side"
                    ),
                    "value": profile.image_long_side,
                },
                {
                    "op": "replace",
                    "path": "/student/connector/latent_tokens",
                    "value": profile.latent_tokens,
                },
            ],
        }
        for profile in profiles
    ]
    child_path = generated_root / "architecture_sweep.yaml"
    _write_yaml(child_path, child)
    sweep = compile_sweep_plan(
        child_path,
        repo_root=repo,
        python=python,
        compile_root=generated_root / "runs",
    )
    tolerance = float(raw.get("overshoot_tolerance_fraction", 0.02))
    if not 0 <= tolerance < 1:
        raise ValueError("overshoot_tolerance_fraction must be within [0, 1)")
    fingerprint = _fingerprint(
        {
            "spec": raw,
            "profiles": [profile.to_dict() for profile in profiles],
            "budgets": budgets.to_dict(),
            "sweep": sweep.fingerprint,
        }
    )
    return ArchitectureSweepPlan(
        name=name,
        root=str(root),
        baseline=baseline,
        profiles=tuple(profiles),
        budgets=budgets,
        overshoot_tolerance_fraction=tolerance,
        sweep=sweep,
        fingerprint=fingerprint,
        raw_spec=raw,
    )


def compute_budget_report(plan: ArchitectureSweepPlan) -> dict[str, Any]:
    """Verify realized checkpoint FLOPs against each phase's shared budget."""

    budgets = plan.budgets.to_dict()
    runs: dict[str, Any] = {}
    overall = "pass"
    for variant in plan.sweep.variants:
        stages: dict[str, Any] = {}
        for stage in _STAGES:
            pointer = (
                Path(variant.plan.root)
                / "artifacts"
                / stage
                / "latest_checkpoint.txt"
            )
            if not pointer.is_file():
                raise FileNotFoundError(f"missing {stage} checkpoint: {pointer}")
            checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
            state = json.loads(
                (checkpoint / "trainer_state.json").read_text(
                    encoding="utf-8"
                )
            )
            seen = int(state.get("student_flops_seen", -1))
            budget = int(budgets[stage])
            overshoot = seen - budget
            fraction = overshoot / budget
            status = (
                "pass"
                if seen >= budget
                and fraction <= plan.overshoot_tolerance_fraction
                else "fail"
            )
            if status == "fail":
                overall = "fail"
            stages[stage] = {
                "status": status,
                "budget": budget,
                "seen": seen,
                "overshoot": overshoot,
                "overshoot_fraction": fraction,
            }
        runs[variant.id] = stages
    return {
        "schema_version": 1,
        "status": overall,
        "tolerance_fraction": plan.overshoot_tolerance_fraction,
        "budgets": budgets,
        "runs": runs,
    }


class ArchitectureSweepRunner:
    def __init__(
        self,
        plan: ArchitectureSweepPlan,
        *,
        repo_root: str | Path,
    ):
        self.plan = plan
        self.repo_root = Path(repo_root).resolve()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        dry_run = bool(kwargs.get("dry_run", False))
        result = SweepRunner(
            self.plan.sweep,
            repo_root=self.repo_root,
        ).run(**kwargs)
        result["architecture_plan"] = self.plan.to_dict()
        if not dry_run and result.get("comparison"):
            report = compute_budget_report(self.plan)
            report_path = Path(self.plan.root) / "compute_budget_report.json"
            _write_json(report_path, report)
            result["compute_budget_report"] = str(report_path)
            result["compute_budget_status"] = report["status"]
            if report["status"] != "pass":
                raise RuntimeError("architecture compute-budget gate failed")
        return result
