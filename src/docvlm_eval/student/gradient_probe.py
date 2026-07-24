"""Configuration for deterministic multi-loss gradient-conflict diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_SUPPORTED_COMPONENTS = {"vision", "connector", "language"}


@dataclass(frozen=True)
class GradientConflictProbeConfig:
    """Controls periodic shared-trunk gradient cosine measurements."""

    enabled: bool = False
    every_steps: int = 1000
    components: tuple[str, ...] = ("vision", "connector", "language")

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("gradient conflict probe enabled must be boolean")
        if (
            not isinstance(self.every_steps, int)
            or isinstance(self.every_steps, bool)
            or self.every_steps <= 0
        ):
            raise ValueError(
                "gradient conflict probe every_steps must be a positive integer"
            )
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or any(
                not isinstance(component, str)
                or component not in _SUPPORTED_COMPONENTS
                for component in self.components
            )
            or len(set(self.components)) != len(self.components)
        ):
            raise ValueError(
                "gradient conflict probe components must be a unique non-empty "
                "subset of vision, connector, language"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: Mapping[str, Any],
    ) -> "GradientConflictProbeConfig":
        raw = (
            blueprint["training"]["pretraining"].get(
                "gradient_conflict_probe",
                {},
            )
            or {}
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                "gradient conflict probe configuration must be a mapping"
            )
        components = raw.get(
            "components",
            ["vision", "connector", "language"],
        )
        if not isinstance(components, (list, tuple)):
            raise ValueError(
                "gradient conflict probe components must be a list"
            )
        config = cls(
            enabled=raw.get("enabled", False),
            every_steps=raw.get("every_steps", 1000),
            components=tuple(components),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = list(self.components)
        return payload
