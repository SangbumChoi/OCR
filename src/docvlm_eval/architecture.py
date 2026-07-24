"""Validation and parameter estimation for the adjustable sub-1B VLM blueprint."""

from __future__ import annotations

import math
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
    full_attention_layers = language.get("full_attention_layers")
    attention_layer_count = (
        llayers
        if full_attention_layers is None
        else (
            len(full_attention_layers)
            if isinstance(full_attention_layers, list)
            else llayers
        )
    )
    convolution_layer_count = llayers - attention_layer_count
    language_attention = (
        ld * ld + ld
        + 2 * (ld * kv_width + kv_width)
        + ld * ld + ld
    )
    conv_bias = bool(language.get("conv_bias", False))
    conv_kernel = int(language.get("conv_kernel_size", 3))
    language_convolution = (
        3 * ld * ld
        + (3 * ld if conv_bias else 0)
        + ld * conv_kernel
        + (ld if conv_bias else 0)
        + ld * ld
        + (ld if conv_bias else 0)
    )
    language_ffn = 2 * (ld * mlp_width + mlp_width) + mlp_width * ld + ld
    language_blocks = (
        attention_layer_count
        * (language_attention + language_ffn + 2 * ld)
        + convolution_layer_count
        * (language_convolution + language_ffn + 2 * ld)
    )
    language_params = embeddings + language_blocks + ld

    connector = student["connector"]
    connector_in = int(connector["input_width"])
    connector_out = int(connector["output_width"])
    connector_layers = int(connector["layers"])
    connector_mlp = int(connector["mlp_width"])
    connector_latents = int(connector["latent_tokens"])
    connector_family = str(connector.get("family", "gated_resampler"))
    if connector_family == "average_pool_projector":
        connector_params = connector_in * connector_out + connector_out
    else:
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
            + connector_layers * (
                2 * connector_out + cross_attention + connector_ffn + 1
            )
        )

    heads = student["task_heads"]
    contrastive_width = int(heads["contrastive_width"])
    task_head_params = 0
    if bool(heads.get("region_text_contrastive")):
        task_head_params += (
            vd * contrastive_width + contrastive_width
            + ld * contrastive_width + contrastive_width
            + 2
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
    visual_positions = int(vision["max_position_tokens"])
    if math.isqrt(visual_positions) ** 2 != visual_positions:
        errors.append(
            "student.vision.max_position_tokens must form a square "
            "two-dimensional grid"
        )
    if int(language["width"]) % int(language["attention_heads"]):
        errors.append("student.language.width must be divisible by attention_heads")
    if int(language["attention_heads"]) % int(language["kv_heads"]):
        errors.append("student.language.attention_heads must be divisible by kv_heads")
    full_attention_layers = language.get("full_attention_layers")
    if full_attention_layers is not None:
        if not isinstance(full_attention_layers, list):
            errors.append(
                "student.language.full_attention_layers must be null or a list"
            )
        else:
            if not all(
                type(index) is int
                for index in full_attention_layers
            ):
                indices = []
                errors.append(
                    "student.language.full_attention_layers must contain integers"
                )
            else:
                indices = list(full_attention_layers)
            if not indices:
                errors.append(
                    "student.language requires at least one full attention layer"
                )
            if len(set(indices)) != len(indices) or indices != sorted(indices):
                errors.append(
                    "student.language.full_attention_layers must be unique and sorted"
                )
            if any(index < 0 or index >= int(language["layers"]) for index in indices):
                errors.append(
                    "student.language.full_attention_layers contains an out-of-range index"
                )
    if int(language.get("conv_kernel_size", 3)) < 2:
        errors.append(
            "student.language.conv_kernel_size must be at least two"
        )
    if not isinstance(language.get("conv_bias", False), bool):
        errors.append("student.language.conv_bias must be a boolean")
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
    if input_pipeline.get("visual_canvas_mode", "fixed_square") not in {
        "fixed_square",
        "batch_adaptive",
    }:
        errors.append(
            "training.pretraining.input_pipeline.visual_canvas_mode must be "
            "fixed_square or batch_adaptive"
        )
    if input_pipeline.get("visual_sequence_mode", "dense") not in {
        "dense",
        "packed",
    }:
        errors.append(
            "training.pretraining.input_pipeline.visual_sequence_mode must be "
            "dense or packed"
        )
    if input_pipeline.get("packed_attention_backend", "auto") not in {
        "auto",
        "flex",
        "loop",
    }:
        errors.append(
            "training.pretraining.input_pipeline.packed_attention_backend must be "
            "auto, flex, or loop"
        )
    aspect_ratio_bucketing = input_pipeline.get(
        "aspect_ratio_bucketing",
        False,
    )
    if not isinstance(aspect_ratio_bucketing, bool):
        errors.append(
            "training.pretraining.input_pipeline.aspect_ratio_bucketing must be boolean"
        )
    if (
        input_pipeline.get("visual_sequence_mode", "dense") == "packed"
        and aspect_ratio_bucketing is True
    ):
        errors.append(
            "training.pretraining.input_pipeline.aspect_ratio_bucketing must be "
            "false for packed visual sequences"
        )
    if float(input_pipeline.get("aspect_ratio_bucket_log2_step", 0.5)) <= 0:
        errors.append(
            "training.pretraining.input_pipeline.aspect_ratio_bucket_log2_step "
            "must be positive"
        )
    rotation_probability = float(input_pipeline.get("rotation_probability", -1.0))
    if not 0.0 <= rotation_probability <= 1.0:
        errors.append(
            "training.pretraining.input_pipeline.rotation_probability must be between 0 and 1"
        )
    if input_pipeline.get("balance_by") not in {"task", "source", "language", "component"}:
        errors.append(
            "training.pretraining.input_pipeline.balance_by must be task, source, language, "
            "or component"
        )
    adaptive_mixture = input_pipeline.get("adaptive_mixture", {}) or {}
    if not isinstance(adaptive_mixture, dict):
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture must be a mapping"
        )
        adaptive_mixture = {}
    adaptive_enabled_raw = adaptive_mixture.get("enabled", False)
    if not isinstance(adaptive_enabled_raw, bool):
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture.enabled "
            "must be boolean"
        )
    adaptive_enabled = (
        adaptive_enabled_raw
        if isinstance(adaptive_enabled_raw, bool)
        else False
    )

    def adaptive_number(field: str, default: float) -> float:
        raw_value = adaptive_mixture.get(field, default)
        if isinstance(raw_value, bool):
            errors.append(
                "training.pretraining.input_pipeline.adaptive_mixture."
                f"{field} must be a finite number"
            )
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(
                "training.pretraining.input_pipeline.adaptive_mixture."
                f"{field} must be a finite number"
            )
            return default
        if not math.isfinite(value):
            errors.append(
                "training.pretraining.input_pipeline.adaptive_mixture."
                f"{field} must be a finite number"
            )
            return default
        return value

    if adaptive_number("step_size", 0.5) <= 0:
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture.step_size "
            "must be positive"
        )
    ema_decay = adaptive_number("ema_decay", 0.8)
    if not 0 <= ema_decay < 1:
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture.ema_decay "
            "must be within [0, 1)"
        )
    min_probability = adaptive_number("min_probability", 0.02)
    if not 0 <= min_probability < 1:
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture."
            "min_probability must be within [0, 1)"
        )
    warmup_evaluations = adaptive_mixture.get("warmup_evaluations", 1)
    if (
        not isinstance(warmup_evaluations, int)
        or isinstance(warmup_evaluations, bool)
        or warmup_evaluations < 0
    ):
        errors.append(
            "training.pretraining.input_pipeline.adaptive_mixture."
            "warmup_evaluations must be a non-negative integer"
        )
    distillation = blueprint["training"]["pretraining"].get("distillation", {})
    if float(distillation.get("temperature", 0.0)) <= 0:
        errors.append("training.pretraining.distillation.temperature must be positive")
    logit_top_k = int(distillation.get("logit_top_k", -1))
    if not 0 <= logit_top_k < int(language["vocab_size"]):
        errors.append(
            "training.pretraining.distillation.logit_top_k must be within the vocabulary"
        )
    sequence_targets = distillation.get("sequence_targets", {})
    for field in ("probability", "min_score"):
        value = float(sequence_targets.get(field, -1.0))
        if not 0.0 <= value <= 1.0:
            errors.append(
                f"training.pretraining.distillation.sequence_targets.{field} "
                "must be between 0 and 1"
            )
    if int(sequence_targets.get("seed", -1)) < 0:
        errors.append(
            "training.pretraining.distillation.sequence_targets.seed must be non-negative"
        )
    for name, maximum_layers in (
        ("vision_layer_pairs", int(vision["layers"])),
        ("language_layer_pairs", int(language["layers"])),
    ):
        for pair in distillation.get(name, []):
            if not isinstance(pair, list) or len(pair) != 2:
                errors.append(f"training.pretraining.distillation.{name} entries must be pairs")
                continue
            student_layer = int(pair[0])
            if student_layer != -1 and not 0 <= student_layer < maximum_layers:
                errors.append(
                    f"training.pretraining.distillation.{name} student layer "
                    f"{student_layer} is out of range"
                )
    optimizer = blueprint["training"]["pretraining"].get("optimizer", {})
    positive_optimizer_fields = (
        "micro_batch_size",
        "grad_accum_steps",
        "learning_rate",
        "max_grad_norm",
        "total_tokens",
    )
    for field in positive_optimizer_fields:
        if float(optimizer.get(field, 0)) <= 0:
            errors.append(f"training.pretraining.optimizer.{field} must be positive")
    epochs = optimizer.get("epochs")
    stop_at_total_tokens = bool(optimizer.get("stop_at_total_tokens", False))
    stop_at_student_flops = bool(
        optimizer.get("stop_at_student_flops", False)
    )
    total_student_flops = optimizer.get("total_student_flops")
    if epochs is None:
        if not (stop_at_total_tokens or stop_at_student_flops):
            errors.append(
                "training.pretraining.optimizer.epochs can be null only when "
                "a token or student-FLOP stop is active"
            )
    elif int(epochs) <= 0:
        errors.append("training.pretraining.optimizer.epochs must be positive when set")
    token_unit = str(optimizer.get("token_unit", "supervised"))
    if token_unit not in {"supervised", "text", "effective"}:
        errors.append(
            "training.pretraining.optimizer.token_unit must be supervised, text, or effective"
        )
    if int(optimizer.get("log_every_steps", 0)) <= 0:
        errors.append("training.pretraining.optimizer.log_every_steps must be positive")
    for field in ("checkpoint_every_steps", "eval_every_steps", "warmup_tokens"):
        if int(optimizer.get(field, -1)) < 0:
            errors.append(
                f"training.pretraining.optimizer.{field} must be non-negative"
            )
    if adaptive_enabled and int(optimizer.get("eval_every_steps", 0)) <= 0:
        errors.append(
            "adaptive mixture requires a positive pretraining eval_every_steps"
        )
    betas = optimizer.get("betas", ())
    if len(betas) != 2 or any(not 0 <= float(beta) < 1 for beta in betas):
        errors.append("training.pretraining.optimizer.betas must contain two values in [0, 1)")
    warmup_tokens = int(optimizer.get("warmup_tokens", -1))
    total_tokens = int(optimizer.get("total_tokens", 0))
    if not 0 <= warmup_tokens < total_tokens:
        errors.append(
            "training.pretraining.optimizer requires 0 <= warmup_tokens < total_tokens"
        )
    schedule_unit = str(optimizer.get("schedule_unit", "tokens"))
    if schedule_unit not in {"tokens", "student_flops"}:
        errors.append(
            "training.pretraining.optimizer.schedule_unit must be "
            "tokens or student_flops"
        )
    if total_student_flops is not None and int(total_student_flops) <= 0:
        errors.append(
            "training.pretraining.optimizer.total_student_flops must be positive"
        )
    if int(optimizer.get("warmup_student_flops", 0)) < 0:
        errors.append(
            "training.pretraining.optimizer.warmup_student_flops "
            "must be non-negative"
        )
    if stop_at_student_flops and total_student_flops is None:
        errors.append(
            "training.pretraining.optimizer.stop_at_student_flops requires "
            "total_student_flops"
        )
    if schedule_unit == "student_flops":
        if total_student_flops is None:
            errors.append(
                "student-FLOP scheduling requires total_student_flops"
            )
        elif not (
            0
            <= int(optimizer.get("warmup_student_flops", -1))
            < int(total_student_flops)
        ):
            errors.append(
                "student-FLOP scheduling requires "
                "0 <= warmup_student_flops < total_student_flops"
            )
    if optimizer.get("precision") not in {"auto", "float32", "bfloat16", "float16"}:
        errors.append("training.pretraining.optimizer.precision is invalid")
    curriculum = blueprint["training"]["pretraining"].get("curriculum") or {}
    curriculum_unit = str(curriculum.get("unit", "optimizer_step_fraction"))
    if (
        epochs is None
        and curriculum.get("stages")
        and curriculum_unit
        not in {"training_token_fraction", "training_compute_fraction"}
    ):
        errors.append(
            "unbounded pretraining requires a token- or "
            "compute-fraction curriculum"
        )
    if curriculum_unit == "training_token_fraction":
        if not stop_at_total_tokens:
            errors.append(
                "training_token_fraction curriculum requires stop_at_total_tokens"
            )
        if any(
            isinstance(stage, dict) and stage.get("group_weights")
            for stage in curriculum.get("stages", [])
        ):
            errors.append(
                "training_token_fraction curriculum cannot override sampler group weights"
            )
    if curriculum_unit == "training_compute_fraction":
        if not stop_at_student_flops:
            errors.append(
                "training_compute_fraction curriculum requires "
                "stop_at_student_flops"
            )
        if any(
            isinstance(stage, dict) and stage.get("group_weights")
            for stage in curriculum.get("stages", [])
        ):
            errors.append(
                "training_compute_fraction curriculum cannot override "
                "sampler group weights"
            )
    if adaptive_enabled and any(
        isinstance(stage, dict) and stage.get("group_weights")
        for stage in curriculum.get("stages", [])
    ):
        errors.append(
            "adaptive mixture cannot be combined with curriculum "
            "group-weight overrides"
        )
    gradient_probe = blueprint["training"]["pretraining"].get(
        "gradient_conflict_probe",
        {},
    ) or {}
    if not isinstance(gradient_probe, dict):
        errors.append(
            "training.pretraining.gradient_conflict_probe must be a mapping"
        )
        gradient_probe = {}
    if not isinstance(gradient_probe.get("enabled", False), bool):
        errors.append(
            "training.pretraining.gradient_conflict_probe.enabled "
            "must be boolean"
        )
    every_steps = gradient_probe.get("every_steps", 1000)
    if (
        not isinstance(every_steps, int)
        or isinstance(every_steps, bool)
        or every_steps <= 0
    ):
        errors.append(
            "training.pretraining.gradient_conflict_probe.every_steps "
            "must be a positive integer"
        )
    probe_components = gradient_probe.get(
        "components",
        ["vision", "connector", "language"],
    )
    supported_probe_components = {"vision", "connector", "language"}
    if (
        not isinstance(probe_components, list)
        or not probe_components
        or any(
            not isinstance(component, str)
            or component not in supported_probe_components
            for component in probe_components
        )
        or len(set(probe_components)) != len(probe_components)
    ):
        errors.append(
            "training.pretraining.gradient_conflict_probe.components must be "
            "a unique non-empty subset of vision, connector, language"
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
    checkpointing = training.get("activation_checkpointing")
    if not isinstance(checkpointing, dict):
        errors.append(
            "training.activation_checkpointing must be a mapping"
        )
    else:
        if not isinstance(checkpointing.get("enabled"), bool):
            errors.append(
                "training.activation_checkpointing.enabled must be a boolean"
            )
        if not isinstance(
            checkpointing.get("use_reentrant"), bool
        ):
            errors.append(
                "training.activation_checkpointing.use_reentrant "
                "must be a boolean"
            )
        components = checkpointing.get("components")
        supported_components = {"vision", "connector", "language"}
        if (
            not isinstance(components, list)
            or not components
            or any(
                not isinstance(component, str)
                or component not in supported_components
                for component in components
            )
            or len(set(components)) != len(components)
        ):
            errors.append(
                "training.activation_checkpointing.components must be "
                "a unique non-empty subset of vision, connector, language"
            )
    supported_pretraining_losses = {
        "autoregressive",
        "teacher_kl",
        "hidden_feature_distillation",
        "region_text_contrastive",
        "box_regression",
        "orientation",
    }
    connector_family = blueprint["student"]["connector"].get(
        "family",
        "gated_resampler",
    )
    if connector_family not in {
        "gated_resampler",
        "average_pool_projector",
    }:
        errors.append(
            "student.connector.family must be gated_resampler or average_pool_projector"
        )
    task_heads = blueprint["student"]["task_heads"]
    contrastive_objective = task_heads.get(
        "contrastive_objective",
        "softmax",
    )
    if contrastive_objective not in {"softmax", "siglip"}:
        errors.append(
            "student.task_heads.contrastive_objective must be softmax or siglip"
        )
    contrastive_temperature = task_heads.get(
        "contrastive_temperature",
        0.07,
    )
    if (
        not isinstance(contrastive_temperature, (int, float))
        or isinstance(contrastive_temperature, bool)
        or not math.isfinite(float(contrastive_temperature))
        or float(contrastive_temperature) <= 0
    ):
        errors.append(
            "student.task_heads.contrastive_temperature must be a positive finite number"
        )
    contrastive_bias_init = task_heads.get("contrastive_bias_init", -10.0)
    if (
        not isinstance(contrastive_bias_init, (int, float))
        or isinstance(contrastive_bias_init, bool)
        or not math.isfinite(float(contrastive_bias_init))
    ):
        errors.append(
            "student.task_heads.contrastive_bias_init must be a finite number"
        )
    box_iou_loss = training["pretraining"].get("box_iou_loss", "giou")
    if (
        not isinstance(box_iou_loss, str)
        or box_iou_loss not in {"giou", "diou", "ciou"}
    ):
        errors.append(
            "training.pretraining.box_iou_loss must be giou, diou, or ciou"
        )
    for name, weight in training["pretraining"]["losses"].items():
        if name not in supported_pretraining_losses:
            errors.append(f"training.pretraining.losses.{name} is not implemented")
        if float(weight) < 0:
            errors.append(f"training.pretraining.losses.{name} must be non-negative")
    curriculum = training["pretraining"].get("curriculum", {})
    if not isinstance(curriculum, dict):
        errors.append("training.pretraining.curriculum must be a mapping")
    else:
        if curriculum.get("unit") not in {
            "optimizer_step_fraction",
            "training_token_fraction",
            "training_compute_fraction",
        }:
            errors.append(
                "training.pretraining.curriculum.unit must be "
                "optimizer_step_fraction, training_token_fraction, or "
                "training_compute_fraction"
            )
        stages = curriculum.get("stages")
        if not isinstance(stages, list) or not stages:
            errors.append("training.pretraining.curriculum.stages must be a non-empty list")
        else:
            stage_ids: set[str] = set()
            previous_boundary = 0.0
            for index, stage in enumerate(stages):
                prefix = f"training.pretraining.curriculum.stages[{index}]"
                if not isinstance(stage, dict):
                    errors.append(f"{prefix} must be a mapping")
                    continue
                stage_id = str(stage.get("id", ""))
                if not stage_id or stage_id in stage_ids:
                    errors.append(f"{prefix}.id must be non-empty and unique")
                stage_ids.add(stage_id)
                boundary = float(stage.get("until_fraction", -1.0))
                if not previous_boundary < boundary <= 1.0:
                    errors.append(
                        f"{prefix}.until_fraction must increase within (0, 1]"
                    )
                previous_boundary = boundary
                for field in ("group_weights", "loss_weights"):
                    weights = stage.get(field, {})
                    if not isinstance(weights, dict):
                        errors.append(f"{prefix}.{field} must be a mapping")
                        continue
                    if any(float(weight) < 0 for weight in weights.values()):
                        errors.append(f"{prefix}.{field} must be non-negative")
                unknown_losses = set(stage.get("loss_weights", {})) - (
                    supported_pretraining_losses
                )
                if unknown_losses:
                    errors.append(
                        f"{prefix}.loss_weights has unsupported losses: "
                        f"{sorted(unknown_losses)}"
                    )
            if not math.isclose(previous_boundary, 1.0, abs_tol=1e-9):
                errors.append(
                    "training.pretraining.curriculum final stage must end at 1.0"
                )
    posttraining = training["posttraining"]
    sft = posttraining["sft"]
    if sft.get("response_format") != "structured_json_v1":
        errors.append("training.posttraining.sft.response_format must be structured_json_v1")
    if sft.get("target_mode") not in {
        "answer_only",
        "free_rationale",
        "evidence_linked",
    }:
        errors.append("training.posttraining.sft.target_mode is invalid")
    sft_optimizer = sft.get("optimizer", {})
    for field in (
        "micro_batch_size",
        "grad_accum_steps",
        "learning_rate",
        "max_grad_norm",
        "total_tokens",
        "log_every_steps",
    ):
        if float(sft_optimizer.get(field, 0)) <= 0:
            errors.append(f"training.posttraining.sft.optimizer.{field} must be positive")
    sft_epochs = sft_optimizer.get("epochs")
    sft_stop_tokens = bool(sft_optimizer.get("stop_at_total_tokens", False))
    sft_stop_flops = bool(
        sft_optimizer.get("stop_at_student_flops", False)
    )
    if sft_epochs is None:
        if not (sft_stop_tokens or sft_stop_flops):
            errors.append(
                "training.posttraining.sft.optimizer.epochs can be null only "
                "when a token or student-FLOP stop is active"
            )
    elif int(sft_epochs) <= 0:
        errors.append(
            "training.posttraining.sft.optimizer.epochs must be positive"
        )
    sft_total_flops = sft_optimizer.get("total_student_flops")
    if sft_total_flops is not None and int(sft_total_flops) <= 0:
        errors.append(
            "training.posttraining.sft.optimizer.total_student_flops "
            "must be positive"
        )
    if int(sft_optimizer.get("warmup_student_flops", 0)) < 0:
        errors.append(
            "training.posttraining.sft.optimizer.warmup_student_flops "
            "must be non-negative"
        )
    if sft_stop_flops and sft_total_flops is None:
        errors.append(
            "training.posttraining.sft.optimizer.stop_at_student_flops "
            "requires total_student_flops"
        )
    sft_schedule_unit = str(sft_optimizer.get("schedule_unit", "tokens"))
    if sft_schedule_unit not in {"tokens", "student_flops"}:
        errors.append(
            "training.posttraining.sft.optimizer.schedule_unit must be "
            "tokens or student_flops"
        )
    if sft_schedule_unit == "student_flops":
        if sft_total_flops is None:
            errors.append(
                "SFT student-FLOP scheduling requires total_student_flops"
            )
        elif not (
            0
            <= int(sft_optimizer.get("warmup_student_flops", -1))
            < int(sft_total_flops)
        ):
            errors.append(
                "SFT student-FLOP scheduling requires "
                "0 <= warmup_student_flops < total_student_flops"
            )
    sft_warmup = int(sft_optimizer.get("warmup_tokens", -1))
    sft_total = int(sft_optimizer.get("total_tokens", 0))
    if not 0 <= sft_warmup < sft_total:
        errors.append(
            "training.posttraining.sft.optimizer requires "
            "0 <= warmup_tokens < total_tokens"
        )
    sft_betas = sft_optimizer.get("betas", ())
    if len(sft_betas) != 2 or any(
        not 0 <= float(beta) < 1 for beta in sft_betas
    ):
        errors.append(
            "training.posttraining.sft.optimizer.betas must contain two values in [0, 1)"
        )
    preference = posttraining.get("preference")
    if not isinstance(preference, dict):
        errors.append("training.posttraining.preference must be a mapping")
        preference = {}
    if preference.get("objective") not in {"dpo", "ipo"}:
        errors.append(
            "training.posttraining.preference.objective must be dpo or ipo"
        )
    if preference.get("preference_source") != "reference_verifier_ranked":
        errors.append(
            "training.posttraining.preference.preference_source must be "
            "reference_verifier_ranked"
        )
    if preference.get("reference_policy") != "sft_checkpoint":
        errors.append(
            "training.posttraining.preference.reference_policy must be "
            "sft_checkpoint"
        )
    if int(preference.get("group_size", 0)) < 2:
        errors.append(
            "training.posttraining.preference.group_size must be at least two"
        )
    if float(preference.get("minimum_reward_margin", -1)) < 0:
        errors.append(
            "training.posttraining.preference.minimum_reward_margin must be "
            "non-negative"
        )
    if float(preference.get("dpo_beta", 0)) <= 0:
        errors.append(
            "training.posttraining.preference.dpo_beta must be positive"
        )
    if float(preference.get("ipo_tau", 0)) <= 0:
        errors.append(
            "training.posttraining.preference.ipo_tau must be positive"
        )
    if preference.get("sequence_reduction") not in {"sum", "mean"}:
        errors.append(
            "training.posttraining.preference.sequence_reduction must be "
            "sum or mean"
        )
    preference_rollout = preference.get("rollout", {})
    if int(preference_rollout.get("max_new_tokens", 0)) <= 0:
        errors.append(
            "training.posttraining.preference.rollout.max_new_tokens must be "
            "positive"
        )
    if float(preference_rollout.get("temperature", 0)) <= 0:
        errors.append(
            "training.posttraining.preference.rollout.temperature must be "
            "positive"
        )
    if not 0 < float(preference_rollout.get("top_p", 0)) <= 1:
        errors.append(
            "training.posttraining.preference.rollout.top_p must be within "
            "(0, 1]"
        )
    if not isinstance(preference_rollout.get("use_kv_cache"), bool):
        errors.append(
            "training.posttraining.preference.rollout.use_kv_cache must be "
            "a boolean"
        )
    preference_optimizer = preference.get("optimizer", {})
    for field in ("learning_rate", "max_grad_norm", "log_every_steps"):
        if float(preference_optimizer.get(field, 0)) <= 0:
            errors.append(
                "training.posttraining.preference.optimizer."
                f"{field} must be positive"
            )
    preference_max_steps = preference_optimizer.get("max_steps")
    preference_stop_flops = bool(
        preference_optimizer.get("stop_at_student_flops", False)
    )
    preference_total_flops = preference_optimizer.get("total_student_flops")
    if preference_max_steps is None:
        if not preference_stop_flops:
            errors.append(
                "training.posttraining.preference.optimizer.max_steps can "
                "be null only with a student-FLOP stop"
            )
    elif int(preference_max_steps) <= 0:
        errors.append(
            "training.posttraining.preference.optimizer.max_steps must be "
            "positive"
        )
    if (
        preference_total_flops is not None
        and int(preference_total_flops) <= 0
    ):
        errors.append(
            "training.posttraining.preference.optimizer.total_student_flops "
            "must be positive"
        )
    if preference_stop_flops and preference_total_flops is None:
        errors.append(
            "training.posttraining.preference.optimizer."
            "stop_at_student_flops requires total_student_flops"
        )
    preference_betas = preference_optimizer.get("betas", ())
    if len(preference_betas) != 2 or any(
        not 0 <= float(beta) < 1 for beta in preference_betas
    ):
        errors.append(
            "training.posttraining.preference.optimizer.betas must contain "
            "two values in [0, 1)"
        )

    rlvr = posttraining["rlvr"]
    if rlvr.get("algorithm") != "grpo":
        errors.append("training.posttraining.rlvr.algorithm must be grpo")
    if rlvr.get("update_policy") != "one_on_policy_update_per_group":
        errors.append(
            "training.posttraining.rlvr.update_policy must be "
            "one_on_policy_update_per_group"
        )
    if int(rlvr.get("group_size", 0)) < 2:
        errors.append("training.posttraining.rlvr.group_size must be at least two")
    advantage_estimator = rlvr.get(
        "advantage_estimator",
        "group_standardized",
    )
    if (
        not isinstance(advantage_estimator, str)
        or advantage_estimator
        not in {"group_standardized", "leave_one_out"}
    ):
        errors.append(
            "training.posttraining.rlvr.advantage_estimator must be "
            "group_standardized or leave_one_out"
        )
    if float(rlvr.get("kl_coefficient", -1)) < 0:
        errors.append("training.posttraining.rlvr.kl_coefficient must be non-negative")
    if float(rlvr.get("advantage_epsilon", 0)) <= 0:
        errors.append("training.posttraining.rlvr.advantage_epsilon must be positive")
    replay = rlvr.get("supervised_replay", {})
    replay_every = int(replay.get("every_steps", -1))
    replay_coefficient = float(replay.get("loss_coefficient", -1))
    if replay_every < 0:
        errors.append(
            "training.posttraining.rlvr.supervised_replay.every_steps "
            "must be non-negative"
        )
    if replay_coefficient < 0:
        errors.append(
            "training.posttraining.rlvr.supervised_replay.loss_coefficient "
            "must be non-negative"
        )
    if (replay_every == 0) != (replay_coefficient == 0):
        errors.append(
            "training.posttraining.rlvr.supervised_replay interval and "
            "coefficient must both be zero or both be positive"
        )
    malformed_reward = float(rlvr.get("malformed_reward", -1))
    if not 0 <= malformed_reward <= 1:
        errors.append("training.posttraining.rlvr.malformed_reward must be within [0, 1]")
    rollout = rlvr.get("rollout", {})
    if int(rollout.get("max_new_tokens", 0)) <= 0:
        errors.append(
            "training.posttraining.rlvr.rollout.max_new_tokens must be positive"
        )
    if float(rollout.get("temperature", 0)) <= 0:
        errors.append("training.posttraining.rlvr.rollout.temperature must be positive")
    if not 0 < float(rollout.get("top_p", 0)) <= 1:
        errors.append("training.posttraining.rlvr.rollout.top_p must be within (0, 1]")
    if not isinstance(rollout.get("use_kv_cache"), bool):
        errors.append(
            "training.posttraining.rlvr.rollout.use_kv_cache must be a boolean"
        )
    supported_rewards = {
        "answer_correctness",
        "normalized_text_similarity",
        "box_iou",
        "table_tree_similarity",
        "chart_numeric_tolerance",
        "formula_equivalence",
        "grounded_rationale_consistency",
        "calibrated_abstention",
    }
    unknown_rewards = set(rlvr["reward_mix"]) - supported_rewards
    if unknown_rewards:
        errors.append(
            "training.posttraining.rlvr.reward_mix has unsupported rewards: "
            f"{sorted(unknown_rewards)}"
        )
    rl_optimizer = rlvr.get("optimizer", {})
    for field in (
        "learning_rate",
        "max_grad_norm",
        "log_every_steps",
    ):
        if float(rl_optimizer.get(field, 0)) <= 0:
            errors.append(f"training.posttraining.rlvr.optimizer.{field} must be positive")
    rl_max_steps = rl_optimizer.get("max_steps")
    rl_stop_flops = bool(
        rl_optimizer.get("stop_at_student_flops", False)
    )
    rl_total_flops = rl_optimizer.get("total_student_flops")
    if rl_max_steps is None:
        if not rl_stop_flops:
            errors.append(
                "training.posttraining.rlvr.optimizer.max_steps can be null "
                "only with a student-FLOP stop"
            )
    elif int(rl_max_steps) <= 0:
        errors.append(
            "training.posttraining.rlvr.optimizer.max_steps must be positive"
        )
    if rl_total_flops is not None and int(rl_total_flops) <= 0:
        errors.append(
            "training.posttraining.rlvr.optimizer.total_student_flops "
            "must be positive"
        )
    if rl_stop_flops and rl_total_flops is None:
        errors.append(
            "training.posttraining.rlvr.optimizer.stop_at_student_flops "
            "requires total_student_flops"
        )
    rl_betas = rl_optimizer.get("betas", ())
    if len(rl_betas) != 2 or any(
        not 0 <= float(beta) < 1 for beta in rl_betas
    ):
        errors.append(
            "training.posttraining.rlvr.optimizer.betas must contain two values in [0, 1)"
        )
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
    gates = blueprint.get("evaluation_gates")
    expected_gate_ids = {
        "parameter_budget",
        "generalization",
        "grounding",
        "reasoning",
        "multilingual",
        "reliability",
        "visual_efficiency",
        "training_feasibility",
    }
    if not isinstance(gates, list):
        errors.append("evaluation_gates must be a list")
    else:
        gate_ids = [
            str(gate.get("id", "")) if isinstance(gate, dict) else ""
            for gate in gates
        ]
        if set(gate_ids) != expected_gate_ids or len(gate_ids) != len(
            expected_gate_ids
        ):
            errors.append(
                "evaluation_gates must declare each supported gate exactly once"
            )
        for index, gate in enumerate(gates):
            prefix = f"evaluation_gates[{index}]"
            if not isinstance(gate, dict):
                errors.append(f"{prefix} must be a mapping")
                continue
            if not str(gate.get("requirement", "")).strip():
                errors.append(f"{prefix}.requirement must be non-empty")
        by_id = {
            str(gate.get("id")): gate
            for gate in gates
            if isinstance(gate, dict)
        }
        if int(by_id.get("parameter_budget", {}).get("max_parameters", 0)) <= 0:
            errors.append(
                "evaluation_gates.parameter_budget.max_parameters must be positive"
            )
        nonnegative_fields = {
            "generalization": (
                "min_heldout_score_delta",
                "max_gap_increase",
            ),
            "grounding": (
                "min_box_iou_delta",
                "max_extraction_similarity_drop",
            ),
            "reasoning": ("min_score_delta",),
            "multilingual": ("max_language_drop",),
            "reliability": (
                "min_selective_risk_reduction",
                "max_hallucination_increase",
                "max_calibrated_ece",
                "max_ece_increase_vs_raw",
                "min_calibrated_ece_reduction",
            ),
            "visual_efficiency": ("max_abs_delta_vs_loop",),
        }
        for gate_id, fields in nonnegative_fields.items():
            gate = by_id.get(gate_id, {})
            for field in fields:
                if float(gate.get(field, -1)) < 0:
                    errors.append(
                        f"evaluation_gates.{gate_id}.{field} "
                        "must be non-negative"
                    )
        grounding_gate = by_id.get("grounding", {})
        patterns = grounding_gate.get("extraction_answer_type_patterns")
        if not isinstance(patterns, list) or not all(
            str(pattern).strip() for pattern in patterns
        ):
            errors.append(
                "evaluation_gates.grounding.extraction_answer_type_patterns "
                "must be a non-empty string list"
            )
        if int(
            by_id.get("reasoning", {}).get("min_counterfactual_pairs", 0)
        ) <= 0:
            errors.append(
                "evaluation_gates.reasoning.min_counterfactual_pairs "
                "must be positive"
            )
        if int(by_id.get("multilingual", {}).get("min_languages", 0)) <= 0:
            errors.append(
                "evaluation_gates.multilingual.min_languages must be positive"
            )
        coverage = float(by_id.get("reliability", {}).get("coverage", 0))
        if not 0 < coverage <= 1:
            errors.append(
                "evaluation_gates.reliability.coverage must be within (0, 1]"
            )
        efficiency = by_id.get("visual_efficiency", {})
        if efficiency.get("candidate_requested_backend") not in {
            "auto",
            "flex",
        }:
            errors.append(
                "evaluation_gates.visual_efficiency."
                "candidate_requested_backend must be auto or flex"
            )
        if (
            efficiency.get("dense_control_requested_backend")
            != "dense_adaptive"
        ):
            errors.append(
                "evaluation_gates.visual_efficiency."
                "dense_control_requested_backend must be dense_adaptive"
            )
        if efficiency.get("required_resolved_backend") != "flex":
            errors.append(
                "evaluation_gates.visual_efficiency."
                "required_resolved_backend must be flex"
            )
        if efficiency.get("required_device_type") != "cuda":
            errors.append(
                "evaluation_gates.visual_efficiency."
                "required_device_type must be cuda"
            )
        if efficiency.get("required_mode") != "training":
            errors.append(
                "evaluation_gates.visual_efficiency.required_mode must be training"
            )
        for field in (
            "min_benchmark_schema_version",
            "min_visual_tokens",
            "min_batch_size",
            "min_warmup_iterations",
            "min_measured_iterations",
            "min_rounds",
        ):
            if int(efficiency.get(field, -1)) <= 0:
                errors.append(
                    f"evaluation_gates.visual_efficiency.{field} "
                    "must be positive"
                )
        feasibility = by_id.get("training_feasibility", {})
        if feasibility.get("required_device_type") != "cuda":
            errors.append(
                "evaluation_gates.training_feasibility."
                "required_device_type must be cuda"
            )
        if (
            feasibility.get(
                "required_resolved_visual_attention_backend"
            )
            != "flex"
        ):
            errors.append(
                "evaluation_gates.training_feasibility."
                "required_resolved_visual_attention_backend must be flex"
            )
        if feasibility.get("required_precision") not in {
            "float16",
            "bfloat16",
        }:
            errors.append(
                "evaluation_gates.training_feasibility."
                "required_precision must be float16 or bfloat16"
            )
        if not isinstance(
            feasibility.get("require_gradient_checkpointing"), bool
        ):
            errors.append(
                "evaluation_gates.training_feasibility."
                "require_gradient_checkpointing must be a boolean"
            )
        required_checkpoint_components = feasibility.get(
            "required_gradient_checkpointing_components"
        )
        if (
            not isinstance(required_checkpoint_components, list)
            or not required_checkpoint_components
            or len(set(required_checkpoint_components))
            != len(required_checkpoint_components)
            or not set(required_checkpoint_components)
            <= {"vision", "connector", "language"}
        ):
            errors.append(
                "evaluation_gates.training_feasibility."
                "required_gradient_checkpointing_components are invalid"
            )
        if not isinstance(
            feasibility.get(
                "required_gradient_checkpointing_use_reentrant"
            ),
            bool,
        ):
            errors.append(
                "evaluation_gates.training_feasibility."
                "required_gradient_checkpointing_use_reentrant "
                "must be a boolean"
            )
        if isinstance(checkpointing, dict) and (
            feasibility.get("require_gradient_checkpointing")
            != checkpointing.get("enabled")
            or required_checkpoint_components
            != checkpointing.get("components")
            or feasibility.get(
                "required_gradient_checkpointing_use_reentrant"
            )
            != checkpointing.get("use_reentrant")
        ):
            errors.append(
                "training feasibility gate checkpointing requirements "
                "must match training.activation_checkpointing"
            )
        for field in (
            "min_benchmark_schema_version",
            "required_micro_batch_size",
            "min_text_tokens",
            "min_visual_tokens_per_sample",
            "min_warmup_steps",
            "min_measured_steps",
        ):
            if int(feasibility.get(field, -1)) <= 0:
                errors.append(
                    f"evaluation_gates.training_feasibility.{field} "
                    "must be positive"
                )
        peak_fraction = float(
            feasibility.get("max_peak_reserved_fraction", 0)
        )
        if not 0 < peak_fraction < 1:
            errors.append(
                "evaluation_gates.training_feasibility."
                "max_peak_reserved_fraction must be within (0, 1)"
            )
        for field in (
            "min_median_speedup_vs_loop",
            "min_round_speedup_vs_loop",
            "max_peak_memory_ratio_vs_loop",
            "min_median_speedup_vs_dense_adaptive",
            "min_round_speedup_vs_dense_adaptive",
            "max_peak_memory_ratio_vs_dense_adaptive",
        ):
            if float(efficiency.get(field, 0)) <= 0:
                errors.append(
                    f"evaluation_gates.visual_efficiency.{field} "
                    "must be positive"
                )
    return estimates, errors
