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
    contrastive_temperature: float = 0.07


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
        return cls(
            vision=VisionConfig(**raw["vision"]),
            language=LanguageConfig(**raw["language"]),
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
        return errors


def student_config_fingerprint(config: StudentConfig) -> str:
    """Return the canonical architecture fingerprint used by runtime evidence."""

    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
