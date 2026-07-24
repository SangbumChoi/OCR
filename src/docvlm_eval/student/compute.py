"""Analytical student FLOP accounting for compute-matched architecture studies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .config import StudentConfig


@dataclass(frozen=True)
class ForwardFlops:
    vision: int
    connector: int
    language: int
    lm_head: int
    task_heads: int

    @property
    def total(self) -> int:
        return (
            self.vision
            + self.connector
            + self.language
            + self.lm_head
            + self.task_heads
        )

    def to_dict(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class TrainingFlops:
    """Separate scientific compute from checkpoint implementation overhead."""

    algorithmic: int
    checkpoint_recompute: int = 0

    @property
    def executed(self) -> int:
        return self.algorithmic + self.checkpoint_recompute

    def to_dict(self) -> dict[str, int]:
        return {
            "algorithmic": self.algorithmic,
            "checkpoint_recompute": self.checkpoint_recompute,
            "executed": self.executed,
        }


def visual_tokens_for_canvas(
    image_long_side: int,
    patch_size: int,
    max_position_tokens: int,
) -> int:
    if image_long_side <= 0 or patch_size <= 0 or max_position_tokens <= 0:
        raise ValueError("visual canvas dimensions must be positive")
    maximum_side = math.isqrt(max_position_tokens)
    if maximum_side * maximum_side != max_position_tokens:
        raise ValueError("max_position_tokens must form a square canvas")
    patch_side = min(
        math.ceil(image_long_side / patch_size),
        maximum_side,
    )
    return patch_side * patch_side


def estimate_language_kv_cache_bytes(
    config: StudentConfig,
    *,
    sequence_tokens: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    """Return compact GQA key/value storage for every decoder layer."""

    if min(sequence_tokens, batch_size, bytes_per_element) <= 0:
        raise ValueError("KV-cache dimensions must be positive")
    head_dim = (
        config.language.width // config.language.attention_heads
    )
    return (
        config.language.layers
        * 2
        * batch_size
        * config.language.kv_heads
        * sequence_tokens
        * head_dim
        * bytes_per_element
    )


def _vision_block_flops(
    config: StudentConfig,
    tokens: int,
    batch_size: int,
) -> int:
    if tokens == 0:
        return 0
    vision = config.vision
    width = vision.width
    hidden = int(width * vision.mlp_ratio)
    attention_projections = 8 * tokens * width * width
    attention_products = 4 * tokens * tokens * width
    mlp = 4 * tokens * width * hidden
    return batch_size * vision.layers * (
        attention_projections + attention_products + mlp
    )


def _vision_flops(config: StudentConfig, tokens: int, batch_size: int) -> int:
    if tokens == 0:
        return 0
    vision = config.vision
    patch_projection = (
        2
        * tokens
        * vision.width
        * 3
        * vision.patch_size
        * vision.patch_size
    )
    return (
        batch_size * patch_projection
        + _vision_block_flops(config, tokens, batch_size)
    )


def _connector_flops(
    config: StudentConfig,
    vision_tokens: int,
    batch_size: int,
) -> int:
    if vision_tokens == 0:
        return 0
    connector = config.connector
    latents = connector.latent_tokens
    source_width = connector.input_width
    width = connector.output_width
    projections = (
        4 * latents * width * width
        + 4 * vision_tokens * source_width * width
    )
    attention_products = 4 * latents * vision_tokens * width
    mlp = 6 * latents * width * connector.mlp_width
    return batch_size * connector.layers * (
        projections + attention_products + mlp
    )


def _language_flops(
    config: StudentConfig,
    sequence_tokens: int,
    batch_size: int,
) -> int:
    language = config.language
    width = language.width
    kv_width = (
        language.kv_heads
        * (language.width // language.attention_heads)
    )
    projections = (
        4 * sequence_tokens * width * width
        + 4 * sequence_tokens * width * kv_width
    )
    attention_products = (
        4 * sequence_tokens * sequence_tokens * width
    )
    mlp = 6 * sequence_tokens * width * language.mlp_width
    return batch_size * language.layers * (
        projections + attention_products + mlp
    )


def _language_decode_flops(
    config: StudentConfig,
    key_tokens: int,
    batch_size: int,
) -> int:
    """Estimate one cached-token decoder pass against existing keys."""

    language = config.language
    width = language.width
    kv_width = (
        language.kv_heads
        * (language.width // language.attention_heads)
    )
    projections = 4 * width * width + 4 * width * kv_width
    attention_products = 4 * key_tokens * width
    mlp = 6 * width * language.mlp_width
    return batch_size * language.layers * (
        projections + attention_products + mlp
    )


def _lm_head_flops(
    config: StudentConfig,
    sequence_tokens: int,
    batch_size: int,
) -> int:
    return (
        batch_size
        * 2
        * sequence_tokens
        * config.language.width
        * config.language.vocab_size
    )


def _task_head_flops(
    config: StudentConfig,
    *,
    has_image: bool,
    batch_size: int,
) -> int:
    heads = config.task_heads
    total = 0
    if heads.box_regression:
        total += 2 * config.language.width * 4
    if has_image and heads.orientation:
        total += 2 * config.vision.width * 4
    if has_image and heads.region_text_contrastive:
        total += (
            2 * config.vision.width * heads.contrastive_width
            + 2 * config.language.width * heads.contrastive_width
            + 2 * batch_size * heads.contrastive_width
        )
    return batch_size * total


def estimate_forward_flops(
    config: StudentConfig,
    *,
    text_tokens: int,
    vision_tokens: int,
    batch_size: int = 1,
    include_vision: bool = True,
    include_lm_head: bool = True,
    include_task_heads: bool = True,
) -> ForwardFlops:
    """Estimate dense multiply-add FLOPs for one native-student forward."""

    if text_tokens <= 0 or vision_tokens < 0 or batch_size <= 0:
        raise ValueError("text tokens and batch size must be positive")
    has_image = include_vision and vision_tokens > 0
    prefix_tokens = config.connector.latent_tokens if vision_tokens > 0 else 0
    sequence_tokens = text_tokens + prefix_tokens
    return ForwardFlops(
        vision=(
            _vision_flops(config, vision_tokens, batch_size)
            if has_image
            else 0
        ),
        connector=(
            _connector_flops(config, vision_tokens, batch_size)
            if has_image
            else 0
        ),
        language=_language_flops(
            config,
            sequence_tokens,
            batch_size,
        ),
        lm_head=(
            _lm_head_flops(config, sequence_tokens, batch_size)
            if include_lm_head
            else 0
        ),
        task_heads=(
            _task_head_flops(
                config,
                has_image=has_image,
                batch_size=batch_size,
            )
            if include_task_heads
            else 0
        ),
    )


def estimate_training_flops(
    config: StudentConfig,
    *,
    text_tokens: int,
    vision_tokens: int,
    batch_size: int = 1,
) -> int:
    """Estimate one forward plus parameter/activation backward as 3x forward."""

    return 3 * estimate_forward_flops(
        config,
        text_tokens=text_tokens,
        vision_tokens=vision_tokens,
        batch_size=batch_size,
    ).total


def estimate_training_flops_breakdown(
    config: StudentConfig,
    *,
    text_tokens: int,
    vision_tokens: int,
    batch_size: int = 1,
    checkpoint_components: tuple[str, ...] = (),
) -> TrainingFlops:
    """Estimate algorithmic training FLOPs and checkpoint recomputation."""

    supported = {"vision", "connector", "language"}
    requested = set(checkpoint_components)
    if len(requested) != len(checkpoint_components) or not requested <= supported:
        raise ValueError(
            "checkpoint components must be a unique subset of "
            "vision, connector, language"
        )
    forward = estimate_forward_flops(
        config,
        text_tokens=text_tokens,
        vision_tokens=vision_tokens,
        batch_size=batch_size,
    )
    has_image = vision_tokens > 0
    sequence_tokens = text_tokens + (
        config.connector.latent_tokens if has_image else 0
    )
    recompute = 0
    if "vision" in requested and has_image:
        recompute += _vision_block_flops(
            config,
            vision_tokens,
            batch_size,
        )
    if "connector" in requested and has_image:
        recompute += _connector_flops(
            config,
            vision_tokens,
            batch_size,
        )
    if "language" in requested:
        recompute += _language_flops(
            config,
            sequence_tokens,
            batch_size,
        )
    return TrainingFlops(
        algorithmic=3 * forward.total,
        checkpoint_recompute=recompute,
    )


def estimate_batch_training_flops(
    config: StudentConfig,
    batch: dict[str, Any],
) -> int:
    return estimate_batch_training_flops_breakdown(config, batch).algorithmic


def estimate_batch_training_flops_breakdown(
    config: StudentConfig,
    batch: dict[str, Any],
    *,
    checkpoint_components: tuple[str, ...] = (),
) -> TrainingFlops:
    """Account for actual batch shapes and optional block recomputation."""

    input_ids = batch.get("input_ids")
    if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
        raise ValueError("compute accounting requires rank-2 input_ids")
    pixel_values = batch.get("pixel_values")
    packed_pixels = batch.get("packed_pixel_values")
    packed_cu_seqlens = batch.get("packed_cu_seqlens")
    if pixel_values is not None and packed_pixels is not None:
        raise ValueError("compute accounting received dense and packed visual inputs")
    if packed_pixels is not None:
        if packed_cu_seqlens is None:
            raise ValueError("packed compute accounting requires cu_seqlens")
        if getattr(packed_pixels, "ndim", 0) != 4:
            raise ValueError("packed_pixel_values must have rank four")
        boundaries = [int(value) for value in packed_cu_seqlens.tolist()]
        if len(boundaries) != int(input_ids.shape[0]) + 1:
            raise ValueError("packed visual batch dimension must match input_ids")
        if boundaries[0] != 0 or boundaries[-1] != int(packed_pixels.shape[0]):
            raise ValueError("packed cu_seqlens do not cover every visual token")
        lengths = [
            end - start for start, end in zip(boundaries, boundaries[1:])
        ]
        if any(
            length <= 0 or length > config.vision.max_position_tokens
            for length in lengths
        ):
            raise ValueError("packed sample exceeds the visual position budget")
        per_sample = [
            estimate_training_flops_breakdown(
                config,
                text_tokens=int(input_ids.shape[1]),
                vision_tokens=length,
                batch_size=1,
                checkpoint_components=checkpoint_components,
            )
            for length in lengths
        ]
        return TrainingFlops(
            algorithmic=sum(item.algorithmic for item in per_sample),
            checkpoint_recompute=sum(
                item.checkpoint_recompute for item in per_sample
            ),
        )
    if pixel_values is None:
        vision_tokens = 0
    else:
        if getattr(pixel_values, "ndim", 0) != 4:
            raise ValueError("pixel_values must have rank four")
        height, width = pixel_values.shape[-2:]
        patch = config.vision.patch_size
        vision_tokens = math.ceil(height / patch) * math.ceil(width / patch)
        if vision_tokens > config.vision.max_position_tokens:
            raise ValueError("batch exceeds the visual position budget")
    return estimate_training_flops_breakdown(
        config,
        text_tokens=int(input_ids.shape[1]),
        vision_tokens=vision_tokens,
        batch_size=int(input_ids.shape[0]),
        checkpoint_components=checkpoint_components,
    )


def estimate_rlvr_step_flops(
    config: StudentConfig,
    *,
    vision_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    group_size: int,
    replay_text_tokens: int | None = None,
    use_kv_cache: bool = True,
    checkpoint_components: tuple[str, ...] = (),
) -> dict[str, int]:
    """Estimate one rollout, policy/reference scoring, and optional replay."""

    if min(prompt_tokens, completion_tokens, group_size) <= 0:
        raise ValueError("RLVR token counts and group size must be positive")
    image_only = estimate_forward_flops(
        config,
        text_tokens=1,
        vision_tokens=vision_tokens,
        include_lm_head=False,
        include_task_heads=False,
    )
    image_encoding = image_only.vision + image_only.connector
    rollout = image_encoding
    if use_kv_cache:
        prefill = estimate_forward_flops(
            config,
            text_tokens=prompt_tokens,
            vision_tokens=vision_tokens,
            batch_size=group_size,
            include_vision=False,
            include_lm_head=False,
            include_task_heads=False,
        ).total
        rollout += prefill + _lm_head_flops(config, 1, group_size)
        prefix_tokens = (
            config.connector.latent_tokens if vision_tokens > 0 else 0
        )
        for generated in range(1, completion_tokens):
            key_tokens = prefix_tokens + prompt_tokens + generated
            rollout += _language_decode_flops(
                config,
                key_tokens,
                group_size,
            )
            rollout += _lm_head_flops(config, 1, group_size)
    else:
        for generated in range(completion_tokens):
            rollout += estimate_forward_flops(
                config,
                text_tokens=prompt_tokens + generated,
                vision_tokens=vision_tokens,
                batch_size=group_size,
                include_vision=False,
                include_task_heads=False,
            ).total
    scored_tokens = prompt_tokens + completion_tokens
    full_group = estimate_forward_flops(
        config,
        text_tokens=scored_tokens,
        vision_tokens=vision_tokens,
        batch_size=group_size,
    ).total
    policy_update_breakdown = estimate_training_flops_breakdown(
        config,
        text_tokens=scored_tokens,
        vision_tokens=vision_tokens,
        batch_size=group_size,
        checkpoint_components=checkpoint_components,
    )
    policy_update = policy_update_breakdown.algorithmic
    reference_scoring = full_group
    replay_breakdown = (
        estimate_training_flops_breakdown(
            config,
            text_tokens=replay_text_tokens,
            vision_tokens=vision_tokens,
            checkpoint_components=checkpoint_components,
        )
        if replay_text_tokens is not None
        else TrainingFlops(algorithmic=0)
    )
    replay = replay_breakdown.algorithmic
    result = {
        "rollout": rollout,
        "policy_update": policy_update,
        "reference_scoring": reference_scoring,
        "supervised_replay": replay,
    }
    result["total"] = sum(result.values())
    result["checkpoint_recompute"] = (
        policy_update_breakdown.checkpoint_recompute
        + replay_breakdown.checkpoint_recompute
    )
    result["executed_total"] = (
        result["total"] + result["checkpoint_recompute"]
    )
    return result


def compute_profile(
    config: StudentConfig,
    *,
    image_long_side: int,
    text_tokens: int,
    rlvr_prompt_tokens: int = 256,
    rlvr_completion_tokens: int = 128,
    rlvr_group_size: int = 8,
    rlvr_replay_every_steps: int = 20,
    rlvr_use_kv_cache: bool = True,
    checkpoint_components: tuple[str, ...] = (),
) -> dict[str, Any]:
    vision_tokens = visual_tokens_for_canvas(
        image_long_side,
        config.vision.patch_size,
        config.vision.max_position_tokens,
    )
    forward = estimate_forward_flops(
        config,
        text_tokens=text_tokens,
        vision_tokens=vision_tokens,
    )
    rlvr_without_replay = estimate_rlvr_step_flops(
        config,
        vision_tokens=vision_tokens,
        prompt_tokens=rlvr_prompt_tokens,
        completion_tokens=rlvr_completion_tokens,
        group_size=rlvr_group_size,
        use_kv_cache=rlvr_use_kv_cache,
        checkpoint_components=checkpoint_components,
    )
    training = estimate_training_flops_breakdown(
        config,
        text_tokens=text_tokens,
        vision_tokens=vision_tokens,
        checkpoint_components=checkpoint_components,
    )
    replay = training.algorithmic
    replay_recompute = training.checkpoint_recompute
    expected_replay = (
        replay // rlvr_replay_every_steps
        if rlvr_replay_every_steps > 0
        else 0
    )
    expected_rlvr = rlvr_without_replay["total"] + expected_replay
    expected_rlvr_recompute = (
        rlvr_without_replay["checkpoint_recompute"]
        + (
            replay_recompute // rlvr_replay_every_steps
            if rlvr_replay_every_steps > 0
            else 0
        )
    )
    peak_cache_tokens = (
        config.connector.latent_tokens
        + rlvr_prompt_tokens
        + max(rlvr_completion_tokens - 1, 0)
    )
    peak_kv_cache_bytes = (
        estimate_language_kv_cache_bytes(
            config,
            sequence_tokens=peak_cache_tokens,
            batch_size=rlvr_group_size,
        )
        if rlvr_use_kv_cache
        else 0
    )
    return {
        "image_long_side": image_long_side,
        "vision_tokens": vision_tokens,
        "latent_tokens": config.connector.latent_tokens,
        "text_tokens": text_tokens,
        "forward_flops": forward.to_dict(),
        "training_flops_per_sample": training.algorithmic,
        "checkpoint_recompute_flops_per_sample": (
            training.checkpoint_recompute
        ),
        "executed_training_flops_per_sample": training.executed,
        "rlvr": {
            **rlvr_without_replay,
            "expected_replay_per_step": expected_replay,
            "expected_total_per_step": int(expected_rlvr),
            "expected_checkpoint_recompute_per_step": int(
                expected_rlvr_recompute
            ),
            "expected_executed_total_per_step": int(
                expected_rlvr + expected_rlvr_recompute
            ),
            "peak_kv_cache_tokens": peak_cache_tokens,
            "peak_kv_cache_bytes_bfloat16": peak_kv_cache_bytes,
        },
        "convention": {
            "multiply_add": "two_flops",
            "training_multiplier": 3,
            "attention": "dense_masked",
            "student_flops_seen": (
                "algorithmic forward plus backward; checkpoint "
                "recomputation is reported separately"
            ),
            "checkpoint_components": list(checkpoint_components),
            "rlvr_use_kv_cache": rlvr_use_kv_cache,
            "excluded": [
                "normalization",
                "activation",
                "softmax",
                "loss",
                "optimizer",
                "data_loading",
                "external_teacher",
            ],
        },
    }
