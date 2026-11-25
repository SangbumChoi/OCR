"""Deterministic heldout-loss reweighting for balanced pretraining groups."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AdaptiveMixtureConfig:
    """Controls epoch-boundary multiplicative group reweighting."""

    enabled: bool = False
    step_size: float = 0.5
    ema_decay: float = 0.8
    min_probability: float = 0.02
    warmup_evaluations: int = 1

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("adaptive mixture enabled must be boolean")
        if (
            not isinstance(self.step_size, (int, float))
            or isinstance(self.step_size, bool)
            or not math.isfinite(self.step_size)
            or self.step_size <= 0
        ):
            raise ValueError("adaptive mixture step_size must be positive")
        if (
            not isinstance(self.ema_decay, (int, float))
            or isinstance(self.ema_decay, bool)
            or not math.isfinite(self.ema_decay)
            or not 0 <= self.ema_decay < 1
        ):
            raise ValueError("adaptive mixture ema_decay must be within [0, 1)")
        if (
            not isinstance(self.min_probability, (int, float))
            or isinstance(self.min_probability, bool)
            or not math.isfinite(self.min_probability)
            or not 0 <= self.min_probability < 1
        ):
            raise ValueError(
                "adaptive mixture min_probability must be within [0, 1)"
            )
        if (
            not isinstance(self.warmup_evaluations, int)
            or isinstance(self.warmup_evaluations, bool)
            or self.warmup_evaluations < 0
        ):
            raise ValueError(
                "adaptive mixture warmup_evaluations must be a non-negative integer"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: Mapping[str, Any],
    ) -> "AdaptiveMixtureConfig":
        raw = (
            blueprint["training"]["pretraining"]["input_pipeline"].get(
                "adaptive_mixture",
                {},
            )
            or {}
        )
        if not isinstance(raw, Mapping):
            raise ValueError("adaptive mixture configuration must be a mapping")
        config = cls(
            enabled=raw.get("enabled", False),
            step_size=float(raw.get("step_size", 0.5)),
            ema_decay=float(raw.get("ema_decay", 0.8)),
            min_probability=float(raw.get("min_probability", 0.02)),
            warmup_evaluations=int(raw.get("warmup_evaluations", 1)),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("adaptive mixture requires at least one group")
    normalized = {str(name): float(value) for name, value in weights.items()}
    if any(
        not math.isfinite(value) or value < 0
        for value in normalized.values()
    ):
        raise ValueError(
            "adaptive mixture group weights must be finite and non-negative"
        )
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError(
            "adaptive mixture requires at least one positive group weight"
        )
    return {name: value / total for name, value in normalized.items()}


class AdaptiveMixtureController:
    """Update group probabilities from heldout losses at epoch boundaries."""

    def __init__(
        self,
        config: AdaptiveMixtureConfig,
        base_weights: Mapping[str, float],
    ):
        config.validate()
        if not config.enabled:
            raise ValueError("adaptive mixture controller requires enabled config")
        self.config = config
        self.weights = _normalize(base_weights)
        if config.min_probability * len(self.weights) >= 1:
            raise ValueError(
                "adaptive mixture min_probability leaves no probability mass"
            )
        self.ema_losses: dict[str, float] = {}
        self.pending = False
        self.evaluations = 0
        self.updates = 0

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.weights))

    def observe(self, losses: Mapping[str, float]) -> None:
        values = {str(name): float(value) for name, value in losses.items()}
        if set(values) != set(self.weights):
            missing = sorted(set(self.weights) - set(values))
            extra = sorted(set(values) - set(self.weights))
            raise ValueError(
                "adaptive mixture heldout groups do not match sampler groups: "
                f"missing={missing}, extra={extra}"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in values.values()
        ):
            raise ValueError(
                "adaptive mixture heldout losses must be finite and non-negative"
            )
        decay = self.config.ema_decay
        self.ema_losses = {
            group: (
                values[group]
                if group not in self.ema_losses
                else decay * self.ema_losses[group]
                + (1 - decay) * values[group]
            )
            for group in self.groups
        }
        self.evaluations += 1
        self.pending = True

    def apply_pending(self) -> bool:
        """Apply one pending observation and return whether weights changed."""
        if not self.pending:
            return False
        self.pending = False
        if self.evaluations <= self.config.warmup_evaluations:
            return False
        mean_loss = sum(self.ema_losses.values()) / len(self.ema_losses)
        scale = max(mean_loss, 1e-12)
        logits = {
            group: math.log(max(self.weights[group], 1e-300))
            + self.config.step_size
            * (self.ema_losses[group] / scale - 1.0)
            for group in self.groups
        }
        maximum = max(logits.values())
        exponentials = {
            group: math.exp(value - maximum)
            for group, value in logits.items()
        }
        raw = _normalize(exponentials)
        floor = self.config.min_probability
        free_mass = 1.0 - floor * len(raw)
        self.weights = {
            group: floor + free_mass * raw[group]
            for group in self.groups
        }
        self.updates += 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "weights": dict(self.weights),
            "ema_losses": dict(self.ema_losses),
            "pending": self.pending,
            "evaluations": self.evaluations,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("config") != self.config.to_dict():
            raise ValueError("adaptive mixture checkpoint config does not match")
        weights = _normalize(state.get("weights") or {})
        if set(weights) != set(self.weights):
            raise ValueError("adaptive mixture checkpoint groups do not match")
        ema_losses = {
            str(name): float(value)
            for name, value in (state.get("ema_losses") or {}).items()
        }
        if ema_losses and set(ema_losses) != set(weights):
            raise ValueError("adaptive mixture checkpoint EMA groups do not match")
        if any(
            not math.isfinite(value) or value < 0
            for value in ema_losses.values()
        ):
            raise ValueError(
                "adaptive mixture checkpoint EMA losses must be finite "
                "and non-negative"
            )
        pending = state.get("pending", False)
        if not isinstance(pending, bool):
            raise ValueError(
                "adaptive mixture checkpoint pending flag must be boolean"
            )
        evaluations = state.get("evaluations", 0)
        updates = state.get("updates", 0)
        if (
            not isinstance(evaluations, int)
            or isinstance(evaluations, bool)
            or not isinstance(updates, int)
            or isinstance(updates, bool)
            or min(evaluations, updates) < 0
            or updates > evaluations
        ):
            raise ValueError(
                "adaptive mixture checkpoint counters are invalid"
            )
        if evaluations == 0 and (ema_losses or pending or updates):
            raise ValueError(
                "adaptive mixture checkpoint has state before any evaluation"
            )
        if evaluations > 0 and not ema_losses:
            raise ValueError(
                "adaptive mixture checkpoint is missing EMA losses"
            )
        self.weights = weights
        self.ema_losses = ema_losses
        self.pending = pending
        self.evaluations = evaluations
        self.updates = updates
