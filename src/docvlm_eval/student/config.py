"""Typed configuration for the native document VLM student."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VisionConfig:
    image_size: int
    patch_size: int
    width: int
    layers: int
    attention_heads: int
    mlp_ratio: float
    max_position_tokens: int
    dropout: float = 0.0


@dataclass(frozen=True)
class LanguageConfig:
    vocab_size: int
    width: int
    layers: int
    attention_heads: int
    kv_heads: int
    mlp_width: int
    tied_embeddings: bool = True
    rope_base: float = 10_000.0
    dropout: float = 0.0
    full_attention_layers: tuple[int, ...] | None = None
    conv_kernel_size: int = 3
    conv_bias: bool = False

    @property
    def layer_types(self) -> tuple[str, ...]:
        if self.full_attention_layers is None:
            return ("attention",) * self.layers
        attention = set(self.full_attention_layers)
        return tuple(
            "attention" if index in attention else "short_conv"
            for index in range(self.layers)
        )


@dataclass(frozen=True)
class ConnectorConfig:
    input_width: int
    output_width: int
    latent_tokens: int
    layers: int
    attention_heads: int
    mlp_width: int
    gate_init: float = 0.01


@dataclass(frozen=True)
class TaskHeadConfig:
    region_text_contrastive: bool = True
    orientation: bool = True
    box_regression: bool = True
    contrastive_width: int = 256
    contrastive_objective: str = "softmax"
    contrastive_temperature: float = 0.07
    contrastive_bias_init: float = -10.0


@dataclass(frozen=True)
class StudentConfig:
    vision: VisionConfig
    language: LanguageConfig
    connector: ConnectorConfig
    task_heads: TaskHeadConfig
    ignore_index: int = -100

    @classmethod
    def from_blueprint(cls, blueprint: dict[str, Any]) -> "StudentConfig":
        student = blueprint["student"]
        vision = {k: v for k, v in student["vision"].items() if k != "family"}
        language = {k: v for k, v in student["language"].items() if k != "family"}
        if language.get("full_attention_layers") is not None:
            language["full_attention_layers"] = tuple(
                language["full_attention_layers"]
            )
        connector = {k: v for k, v in student["connector"].items() if k != "family"}
        return cls(
            vision=VisionConfig(**vision),
            language=LanguageConfig(**language),
            connector=ConnectorConfig(**connector),
            task_heads=TaskHeadConfig(**student["task_heads"]),
        )

    @classmethod
    def tiny(cls, *, vocab_size: int = 256) -> "StudentConfig":
        return cls(
            vision=VisionConfig(
                image_size=32,
                patch_size=8,
                width=64,
                layers=2,
                attention_heads=4,
                mlp_ratio=2.0,
                max_position_tokens=64,
            ),
            language=LanguageConfig(
                vocab_size=vocab_size,
                width=128,
                layers=2,
                attention_heads=8,
                kv_heads=2,
                mlp_width=256,
            ),
            connector=ConnectorConfig(
                input_width=64,
                output_width=128,
                latent_tokens=8,
                layers=1,
                attention_heads=8,
                mlp_width=256,
            ),
            task_heads=TaskHeadConfig(contrastive_width=32),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudentConfig":
        language = dict(raw["language"])
        if language.get("full_attention_layers") is not None:
            language["full_attention_layers"] = tuple(
                language["full_attention_layers"]
            )
        return cls(
            vision=VisionConfig(**raw["vision"]),
            language=LanguageConfig(**language),
            connector=ConnectorConfig(**raw["connector"]),
            task_heads=TaskHeadConfig(**raw["task_heads"]),
            ignore_index=int(raw.get("ignore_index", -100)),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.vision.width % self.vision.attention_heads:
            errors.append("vision width must be divisible by vision attention heads")
        if self.language.width % self.language.attention_heads:
            errors.append("language width must be divisible by language attention heads")
        if self.language.attention_heads % self.language.kv_heads:
            errors.append("language attention heads must be divisible by KV heads")
        attention_layers = self.language.full_attention_layers
        if attention_layers is not None:
            valid_indices = all(
                type(index) is int for index in attention_layers
            )
            if not valid_indices:
                errors.append(
                    "language full attention layers must contain integers"
                )
            elif (
                len(set(attention_layers)) != len(attention_layers)
                or tuple(sorted(attention_layers)) != attention_layers
            ):
                errors.append(
                    "language full attention layers must be unique and sorted"
                )
            if not attention_layers:
                errors.append(
                    "language requires at least one full attention layer"
                )
            if valid_indices and any(
                index < 0 or index >= self.language.layers
                for index in attention_layers
            ):
                errors.append(
                    "language full attention layer index is out of range"
                )
        if self.language.conv_kernel_size < 2:
            errors.append("language convolution kernel size must be at least two")
        if not isinstance(self.language.conv_bias, bool):
            errors.append("language convolution bias must be boolean")
        if self.connector.output_width % self.connector.attention_heads:
            errors.append("connector output width must be divisible by connector attention heads")
        if self.connector.input_width != self.vision.width:
            errors.append("connector input width must equal vision width")
        if self.connector.output_width != self.language.width:
            errors.append("connector output width must equal language width")
        if self.vision.max_position_tokens <= 0:
            errors.append("vision max_position_tokens must be positive")
        position_side = int(self.vision.max_position_tokens**0.5)
        if position_side * position_side != self.vision.max_position_tokens:
            errors.append(
                "vision max_position_tokens must form a square two-dimensional grid"
            )
        if self.task_heads.contrastive_objective not in {"softmax", "siglip"}:
            errors.append(
                "task-head contrastive objective must be softmax or siglip"
            )
        if self.task_heads.contrastive_temperature <= 0:
            errors.append("task-head contrastive temperature must be positive")
        return errors


def student_config_fingerprint(config: StudentConfig) -> str:
    """Return the canonical architecture fingerprint used by runtime evidence."""

    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
