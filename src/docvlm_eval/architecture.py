"""Validation and parameter estimation for the adjustable sub-1B VLM blueprint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_blueprint(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError("blueprint root must be a mapping")
    return document


def estimate_parameters(blueprint: dict[str, Any]) -> dict[str, int]:
    student = blueprint["student"]
    vision = student["vision"]
    language = student["language"]

    vd = int(vision["width"])
    vlayers = int(vision["layers"])
    patch = int(vision["patch_size"])
    vratio = float(vision["mlp_ratio"])
    vpositions = int(vision["max_position_tokens"])
    vision_patch = 3 * patch * patch * vd + vd
    vision_blocks = vlayers * int((4.0 + 2.0 * vratio) * vd * vd + 4 * vd)
    vision_params = vision_patch + vpositions * vd + vision_blocks

    ld = int(language["width"])
    llayers = int(language["layers"])
    mlp_width = int(language["mlp_width"])
    vocab = int(language["vocab_size"])
    embeddings = vocab * ld
    if not bool(language.get("tied_embeddings", True)):
        embeddings *= 2
    language_blocks = llayers * (4 * ld * ld + 3 * ld * mlp_width + 4 * ld)
    language_params = embeddings + language_blocks

    connector_params = int(student["connector"]["estimated_parameters"])
    task_head_params = int(student["task_heads"]["estimated_parameters"])
    total = vision_params + language_params + connector_params + task_head_params
    return {
        "vision": vision_params,
        "language": language_params,
        "connector": connector_params,
        "task_heads": task_head_params,
        "total": total,
    }


def _validate_mix(name: str, values: Any, errors: list[str]) -> None:
    if not isinstance(values, dict) or not values:
        errors.append(f"{name} must be a non-empty mapping")
        return
    numeric = [float(value) for value in values.values()]
    if any(value < 0 for value in numeric):
        errors.append(f"{name} contains a negative weight")
    if abs(sum(numeric) - 1.0) > 1e-6:
        errors.append(f"{name} weights sum to {sum(numeric):.6f}, expected 1.0")


def validate_blueprint(blueprint: dict[str, Any]) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    required = {"schema_version", "budget", "student", "initialization_arms", "training"}
    missing = sorted(required - blueprint.keys())
    if missing:
        return {}, [f"missing top-level keys: {', '.join(missing)}"]

    estimates = estimate_parameters(blueprint)
    budget = blueprint["budget"]
    maximum = int(budget["max_parameters"])
    if estimates["total"] >= maximum:
        errors.append(
            f"estimated deployment size {estimates['total']:,} is not below {maximum:,}"
        )

    target = int(budget["target_parameters"])
    tolerance = float(budget.get("tolerance_fraction", 0.0))
    relative_error = abs(estimates["total"] - target) / target
    if relative_error > tolerance:
        errors.append(
            f"estimated size differs from target by {relative_error:.1%}, "
            f"above tolerance {tolerance:.1%}"
        )

    arm_ids: set[str] = set()
    for arm in blueprint["initialization_arms"]:
        arm_id = str(arm.get("id", ""))
        if not arm_id or arm_id in arm_ids:
            errors.append(f"initialization arm id is empty or duplicated: {arm_id!r}")
        arm_ids.add(arm_id)
        for key in ("vision_transfer", "language_transfer", "connector_transfer"):
            value = float(arm.get(key, -1.0))
            if not 0.0 <= value <= 1.0:
                errors.append(f"{arm_id}.{key} must be between 0 and 1")

    training = blueprint["training"]
    _validate_mix("training.pretraining.data_mix", training["pretraining"]["data_mix"], errors)
    _validate_mix(
        "training.posttraining.sft.data_mix",
        training["posttraining"]["sft"]["data_mix"],
        errors,
    )
    _validate_mix(
        "training.posttraining.rlvr.reward_mix",
        training["posttraining"]["rlvr"]["reward_mix"],
        errors,
    )
    return estimates, errors
