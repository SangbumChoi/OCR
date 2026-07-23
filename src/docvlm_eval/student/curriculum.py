"""Deterministic runtime curriculum controls for native-student pretraining."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


_SUPPORTED_LOSSES = frozenset(
    {
        "autoregressive",
        "teacher_kl",
        "hidden_feature_distillation",
        "region_text_contrastive",
        "box_regression",
        "orientation",
    }
)


@dataclass(frozen=True)
class CurriculumStage:
    """Partial sampler and loss overrides active through one progress boundary."""

    id: str
    until_fraction: float
    group_weights: dict[str, float] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumSchedule:
    """Progress-fraction schedule shared by the sampler and loss composition."""

    stages: tuple[CurriculumStage, ...] = ()
    unit: str = "optimizer_step_fraction"

    @classmethod
    def from_blueprint(cls, blueprint: dict[str, Any]) -> "CurriculumSchedule":
        raw = blueprint["training"]["pretraining"].get("curriculum")
        if not raw:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("pretraining curriculum must be a mapping")
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("pretraining curriculum stages must be a non-empty list")
        if any(not isinstance(stage, dict) for stage in raw_stages):
            raise ValueError("every pretraining curriculum stage must be a mapping")
        stages = tuple(
            CurriculumStage(
                id=str(stage.get("id", "")),
                until_fraction=float(stage.get("until_fraction", -1.0)),
                group_weights={
                    str(name): float(weight)
                    for name, weight in (stage.get("group_weights") or {}).items()
                },
                loss_weights={
                    str(name): float(weight)
                    for name, weight in (stage.get("loss_weights") or {}).items()
                },
            )
            for stage in raw_stages
        )
        schedule = cls(
            stages=stages,
            unit=str(raw.get("unit", "optimizer_step_fraction")),
        )
        schedule.validate()
        return schedule

    def validate(self) -> None:
        if self.unit not in {
            "optimizer_step_fraction",
            "training_token_fraction",
            "training_compute_fraction",
        }:
            raise ValueError(
                "curriculum unit must be optimizer_step_fraction, "
                "training_token_fraction, or training_compute_fraction"
            )
        if not self.stages:
            return
        previous = 0.0
        ids: set[str] = set()
        for stage in self.stages:
            if not stage.id or stage.id in ids:
                raise ValueError("curriculum stage ids must be non-empty and unique")
            ids.add(stage.id)
            if not previous < stage.until_fraction <= 1.0:
                raise ValueError("curriculum until_fraction values must increase within (0, 1]")
            previous = stage.until_fraction
            weights = (*stage.group_weights.values(), *stage.loss_weights.values())
            if any(weight < 0 for weight in weights):
                raise ValueError("curriculum weights must be non-negative")
            unknown_losses = set(stage.loss_weights) - _SUPPORTED_LOSSES
            if unknown_losses:
                raise ValueError(
                    f"curriculum stage {stage.id!r} has unsupported losses: "
                    f"{sorted(unknown_losses)}"
                )
        if not math.isclose(previous, 1.0, abs_tol=1e-9):
            raise ValueError("the final curriculum stage must end at until_fraction=1.0")

    @property
    def fingerprint(self) -> str | None:
        if not self.stages:
            return None
        payload = {
            "unit": self.unit,
            "stages": [
                {
                    "id": stage.id,
                    "until_fraction": stage.until_fraction,
                    "group_weights": stage.group_weights,
                    "loss_weights": stage.loss_weights,
                }
                for stage in self.stages
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def stage_for_step(
        self,
        step: int,
        total_steps: int,
    ) -> CurriculumStage | None:
        if self.unit != "optimizer_step_fraction":
            raise ValueError(
                "token/compute curricula require stage_for_fraction"
            )
        if total_steps <= 0:
            raise ValueError("total curriculum steps must be positive")
        progress = min(max(int(step), 0), total_steps - 1) / total_steps
        return self.stage_for_fraction(progress)

    def stage_for_fraction(self, progress: float) -> CurriculumStage | None:
        if not self.stages:
            return None
        progress = min(max(float(progress), 0.0), 1.0)
        for stage in self.stages:
            if progress < stage.until_fraction:
                return stage
        return self.stages[-1]

    def loss_weights_for_step(
        self,
        base_weights: dict[str, float],
        step: int,
        total_steps: int,
    ) -> dict[str, float]:
        weights = dict(base_weights)
        stage = self.stage_for_step(step, total_steps)
        if stage is not None:
            weights.update(stage.loss_weights)
        return weights

    def loss_weights_for_fraction(
        self,
        base_weights: dict[str, float],
        progress: float,
    ) -> dict[str, float]:
        weights = dict(base_weights)
        stage = self.stage_for_fraction(progress)
        if stage is not None:
            weights.update(stage.loss_weights)
        return weights


def planned_optimizer_steps(
    *,
    num_batches: int,
    grad_accum_steps: int,
    epochs: int,
    max_steps: int | None,
) -> int:
    """Return the exact maximum number of optimizer updates in the training plan."""

    if num_batches <= 0 or grad_accum_steps <= 0 or epochs <= 0:
        raise ValueError("batches, accumulation steps, and epochs must be positive")
    total = math.ceil(num_batches / grad_accum_steps) * epochs
    return min(total, max_steps) if max_steps is not None else total
