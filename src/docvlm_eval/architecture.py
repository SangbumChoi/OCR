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
    vision_mlp = int(vratio * vd)
    vision_attention = 4 * (vd * vd + vd)
    vision_ffn = vd * vision_mlp + vision_mlp + vision_mlp * vd + vd
    vision_blocks = vlayers * (vision_attention + vision_ffn + 4 * vd)
    vision_params = vision_patch + vpositions * vd + vision_blocks + 2 * vd

    ld = int(language["width"])
    llayers = int(language["layers"])
    attention_heads = int(language["attention_heads"])
    kv_heads = int(language["kv_heads"])
    mlp_width = int(language["mlp_width"])
    vocab = int(language["vocab_size"])
    embeddings = vocab * ld
    if not bool(language.get("tied_embeddings", True)):
        embeddings *= 2
    head_dim = ld // attention_heads
    kv_width = head_dim * kv_heads
    language_attention = (
        ld * ld + ld
        + 2 * (ld * kv_width + kv_width)
        + ld * ld + ld
    )
    language_ffn = 2 * (ld * mlp_width + mlp_width) + mlp_width * ld + ld
    language_blocks = llayers * (language_attention + language_ffn + 2 * ld)
    language_params = embeddings + language_blocks + ld

    connector = student["connector"]
    connector_in = int(connector["input_width"])
    connector_out = int(connector["output_width"])
    connector_layers = int(connector["layers"])
    connector_mlp = int(connector["mlp_width"])
    connector_latents = int(connector["latent_tokens"])
    cross_attention = (
        connector_out * connector_out + connector_out
        + 2 * (connector_in * connector_out + connector_out)
        + connector_out * connector_out + connector_out
    )
    connector_ffn = (
        2 * (connector_out * connector_mlp + connector_mlp)
        + connector_mlp * connector_out + connector_out
    )
    connector_params = (
        connector_latents * connector_out
        + connector_layers * (2 * connector_out + cross_attention + connector_ffn + 1)
    )

    heads = student["task_heads"]
    contrastive_width = int(heads["contrastive_width"])
    task_head_params = 0
    if bool(heads.get("region_text_contrastive")):
        task_head_params += (
            vd * contrastive_width + contrastive_width
            + ld * contrastive_width + contrastive_width
        )
    if bool(heads.get("orientation")):
        task_head_params += vd * 4 + 4
    if bool(heads.get("box_regression")):
        task_head_params += ld * 4 + 4
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
    required = {
        "schema_version",
        "budget",
        "tokenizer",
        "student",
        "initialization_arms",
        "training",
    }
    missing = sorted(required - blueprint.keys())
    if missing:
        return {}, [f"missing top-level keys: {', '.join(missing)}"]

    estimates = estimate_parameters(blueprint)
    student = blueprint["student"]
    vision = student["vision"]
    language = student["language"]
    connector = student["connector"]
    if int(vision["width"]) % int(vision["attention_heads"]):
        errors.append("student.vision.width must be divisible by attention_heads")
    if int(language["width"]) % int(language["attention_heads"]):
        errors.append("student.language.width must be divisible by attention_heads")
    if int(language["attention_heads"]) % int(language["kv_heads"]):
        errors.append("student.language.attention_heads must be divisible by kv_heads")
    if int(connector["output_width"]) % int(connector["attention_heads"]):
        errors.append("student.connector.output_width must be divisible by attention_heads")
    if int(connector["input_width"]) != int(vision["width"]):
        errors.append("student.connector.input_width must match student.vision.width")
    if int(connector["output_width"]) != int(language["width"]):
        errors.append("student.connector.output_width must match student.language.width")
    tokenizer = blueprint["tokenizer"]
    if int(tokenizer.get("vocab_size", 0)) != int(language["vocab_size"]):
        errors.append("tokenizer.vocab_size must match student.language.vocab_size")
    if tokenizer.get("normalization") != "NFC":
        errors.append("tokenizer.normalization must be NFC for exact document transcription")
    input_pipeline = blueprint["training"]["pretraining"].get("input_pipeline", {})
    if int(input_pipeline.get("max_text_tokens", 0)) <= 0:
        errors.append("training.pretraining.input_pipeline.max_text_tokens must be positive")
    if int(input_pipeline.get("max_image_long_side", 0)) <= 0:
        errors.append("training.pretraining.input_pipeline.max_image_long_side must be positive")
    rotation_probability = float(input_pipeline.get("rotation_probability", -1.0))
    if not 0.0 <= rotation_probability <= 1.0:
        errors.append(
            "training.pretraining.input_pipeline.rotation_probability must be between 0 and 1"
        )
    if input_pipeline.get("balance_by") not in {"task", "source", "language"}:
        errors.append(
            "training.pretraining.input_pipeline.balance_by must be task, source, or language"
        )
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
