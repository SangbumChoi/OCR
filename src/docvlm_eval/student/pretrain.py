"""Token-scheduled, mixed-precision, resumable pretraining for the native student."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .adaptive_mixture import (
    AdaptiveMixtureConfig,
    AdaptiveMixtureController,
)
from .data import student_model_inputs
from .curriculum import CurriculumSchedule, planned_optimizer_steps
from .compute import estimate_batch_training_flops_breakdown
from .gradient_probe import GradientConflictProbeConfig
from .distillation import DistillationLoss, NativeStudentTeacher, TeacherSignals
from .losses import BOX_IOU_LOSSES
from .model import DocumentVLMStudent


_ONLINE_TEACHER_LOSSES = frozenset(
    {"teacher_kl", "hidden_feature_distillation"}
)


@dataclass(frozen=True)
class PretrainConfig:
    output_dir: str
    epochs: int | None = 1
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_tokens: int = 100_000_000
    total_tokens: int = 20_000_000_000
    stop_at_total_tokens: bool = False
    warmup_student_flops: int = 0
    total_student_flops: int | None = None
    stop_at_student_flops: bool = False
    schedule_unit: str = "tokens"
    token_unit: str = "supervised"
    visual_tokens_per_image: int = 0
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    precision: str = "auto"
    gradient_checkpointing: bool = False
    gradient_checkpointing_components: tuple[str, ...] = (
        "vision",
        "connector",
        "language",
    )
    gradient_checkpointing_use_reentrant: bool = False
    checkpoint_every_steps: int = 1000
    eval_every_steps: int = 1000
    log_every_steps: int = 10
    seed: int = 7
    device: str = "auto"
    resume_from: str | None = None
    tokenizer_fingerprint: str | None = None
    run_stage: str = "pretraining"
    loss_weights: dict[str, float] = field(default_factory=dict)
    box_iou_loss: str = "giou"
    target_source_counts: dict[str, int] = field(default_factory=dict)
    curriculum: CurriculumSchedule = field(default_factory=CurriculumSchedule)
    adaptive_mixture: AdaptiveMixtureConfig = field(
        default_factory=AdaptiveMixtureConfig
    )
    gradient_conflict_probe: GradientConflictProbeConfig = field(
        default_factory=GradientConflictProbeConfig
    )

    def __post_init__(self) -> None:
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be positive when set")
        if self.epochs is None and not (
            self.stop_at_total_tokens or self.stop_at_student_flops
        ):
            raise ValueError(
                "epochs can be null only when a token or student-FLOP stop is active"
            )
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.learning_rate <= 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("learning rate must be positive and min_lr_ratio within [0, 1]")
        if not 0 <= self.weight_decay or self.max_grad_norm <= 0:
            raise ValueError("weight decay and max_grad_norm are invalid")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must be within [0, 1)")
        if self.total_tokens <= 0 or not 0 <= self.warmup_tokens < self.total_tokens:
            raise ValueError("token schedule requires 0 <= warmup_tokens < total_tokens")
        if self.total_student_flops is not None and self.total_student_flops <= 0:
            raise ValueError("total_student_flops must be positive when set")
        if self.warmup_student_flops < 0:
            raise ValueError("warmup_student_flops must be non-negative")
        if self.stop_at_student_flops and self.total_student_flops is None:
            raise ValueError(
                "stop_at_student_flops requires total_student_flops"
            )
        if self.schedule_unit not in {"tokens", "student_flops"}:
            raise ValueError("schedule_unit must be tokens or student_flops")
        if self.schedule_unit == "student_flops":
            if self.total_student_flops is None:
                raise ValueError(
                    "student-FLOP scheduling requires total_student_flops"
                )
            if not (
                0
                <= self.warmup_student_flops
                < self.total_student_flops
            ):
                raise ValueError(
                    "student-FLOP schedule requires "
                    "0 <= warmup_student_flops < total_student_flops"
                )
        if self.token_unit not in {"supervised", "text", "effective"}:
            raise ValueError("token_unit must be supervised, text, or effective")
        if self.visual_tokens_per_image < 0:
            raise ValueError("visual_tokens_per_image must be non-negative")
        if self.token_unit == "effective" and self.visual_tokens_per_image <= 0:
            raise ValueError(
                "effective token accounting requires visual_tokens_per_image"
            )
        if self.precision not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("precision must be auto, float32, bfloat16, or float16")
        supported_checkpointing = {"vision", "connector", "language"}
        if (
            not self.gradient_checkpointing_components
            or len(set(self.gradient_checkpointing_components))
            != len(self.gradient_checkpointing_components)
            or not set(self.gradient_checkpointing_components)
            <= supported_checkpointing
        ):
            raise ValueError(
                "gradient checkpointing components must be a unique "
                "non-empty subset of vision, connector, language"
            )
        if self.checkpoint_every_steps < 0 or self.eval_every_steps < 0:
            raise ValueError("checkpoint and evaluation intervals must be non-negative")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        if any(weight < 0 for weight in self.loss_weights.values()):
            raise ValueError("pretraining loss weights must be non-negative")
        if self.box_iou_loss not in BOX_IOU_LOSSES:
            raise ValueError(
                f"box_iou_loss must be one of {sorted(BOX_IOU_LOSSES)}"
            )
        if any(count < 0 for count in self.target_source_counts.values()):
            raise ValueError("target source counts must be non-negative")
        if not self.run_stage.strip():
            raise ValueError("run_stage cannot be empty")
        self.curriculum.validate()
        self.adaptive_mixture.validate()
        self.gradient_conflict_probe.validate()
        if self.adaptive_mixture.enabled:
            if self.eval_every_steps <= 0:
                raise ValueError(
                    "adaptive mixture requires periodic heldout evaluation"
                )
            if any(
                stage.group_weights
                for stage in self.curriculum.stages
            ):
                raise ValueError(
                    "adaptive mixture cannot be combined with curriculum "
                    "group-weight overrides"
                )
        if (
            (self.stop_at_total_tokens or self.stop_at_student_flops)
            and self.curriculum.stages
            and self.curriculum.unit
            not in {"training_token_fraction", "training_compute_fraction"}
            and self.epochs is None
        ):
            raise ValueError(
                "an unbounded budget run requires a token- or "
                "compute-fraction curriculum"
            )
        if (
            self.curriculum.unit == "training_compute_fraction"
            and self.total_student_flops is None
        ):
            raise ValueError(
                "training_compute_fraction requires total_student_flops"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        output_dir: str | Path,
        **overrides: Any,
    ) -> "PretrainConfig":
        raw = blueprint["training"]["pretraining"]["optimizer"]
        checkpointing = blueprint["training"]["activation_checkpointing"]
        values = {
            "output_dir": str(output_dir),
            "epochs": (
                None if raw.get("epochs") is None else int(raw["epochs"])
            ),
            "max_steps": (
                None if raw.get("max_steps") is None else int(raw["max_steps"])
            ),
            "learning_rate": float(raw["learning_rate"]),
            "min_lr_ratio": float(raw["min_lr_ratio"]),
            "weight_decay": float(raw["weight_decay"]),
            "beta1": float(raw["betas"][0]),
            "beta2": float(raw["betas"][1]),
            "warmup_tokens": int(raw["warmup_tokens"]),
            "total_tokens": int(raw["total_tokens"]),
            "stop_at_total_tokens": bool(raw.get("stop_at_total_tokens", False)),
            "warmup_student_flops": int(
                raw.get("warmup_student_flops", 0)
            ),
            "total_student_flops": (
                None
                if raw.get("total_student_flops") is None
                else int(raw["total_student_flops"])
            ),
            "stop_at_student_flops": bool(
                raw.get("stop_at_student_flops", False)
            ),
            "schedule_unit": str(raw.get("schedule_unit", "tokens")),
            "token_unit": str(raw.get("token_unit", "supervised")),
            "visual_tokens_per_image": int(
                blueprint["student"]["connector"]["latent_tokens"]
            ),
            "grad_accum_steps": int(raw["grad_accum_steps"]),
            "max_grad_norm": float(raw["max_grad_norm"]),
            "precision": str(raw["precision"]),
            "gradient_checkpointing": bool(
                checkpointing["enabled"]
            ),
            "gradient_checkpointing_components": tuple(
                str(value) for value in checkpointing["components"]
            ),
            "gradient_checkpointing_use_reentrant": bool(
                checkpointing["use_reentrant"]
            ),
            "checkpoint_every_steps": int(raw["checkpoint_every_steps"]),
            "eval_every_steps": int(raw["eval_every_steps"]),
            "log_every_steps": int(raw["log_every_steps"]),
            "seed": int(raw["seed"]),
            "loss_weights": {
                str(name): float(weight)
                for name, weight in blueprint["training"]["pretraining"]["losses"].items()
            },
            "box_iou_loss": str(
                blueprint["training"]["pretraining"].get(
                    "box_iou_loss",
                    "giou",
                )
            ),
            "curriculum": CurriculumSchedule.from_blueprint(blueprint),
            "adaptive_mixture": (
                AdaptiveMixtureConfig.from_blueprint(blueprint)
            ),
            "gradient_conflict_probe": (
                GradientConflictProbeConfig.from_blueprint(blueprint)
            ),
        }
        values.update(overrides)
        return cls(**values)


def pretraining_supervision_contract(
    config: PretrainConfig,
    *,
    has_online_teacher: bool,
) -> dict[str, Any]:
    """Resolve stage-level active losses and reject silent supervision gaps."""

    if config.curriculum.stages:
        profiles: list[dict[str, Any]] = [
            {
                "id": stage.id,
                "weights": {
                    **config.loss_weights,
                    **stage.loss_weights,
                },
            }
            for stage in config.curriculum.stages
        ]
    else:
        profiles = [{"id": "base", "weights": dict(config.loss_weights)}]
    active_online: set[str] = set()
    for profile in profiles:
        active = sorted(
            name
            for name, weight in profile["weights"].items()
            if float(weight) > 0
        )
        if not active:
            raise ValueError(
                f"pretraining supervision stage {profile['id']!r} "
                "has no active loss"
            )
        profile["active_losses"] = active
        del profile["weights"]
        active_online.update(_ONLINE_TEACHER_LOSSES.intersection(active))
    if active_online and not has_online_teacher:
        raise ValueError(
            "active online-teacher losses require a native teacher checkpoint: "
            f"{sorted(active_online)}"
        )
    if has_online_teacher and not active_online:
        raise ValueError(
            "native teacher checkpoint provided but teacher_kl and "
            "hidden_feature_distillation are inactive"
        )
    return {
        "has_online_teacher": has_online_teacher,
        "online_teacher_losses": sorted(active_online),
        "target_source_counts": {
            name: int(count)
            for name, count in sorted(config.target_source_counts.items())
        },
        "adaptive_mixture": config.adaptive_mixture.to_dict(),
        "gradient_conflict_probe": config.gradient_conflict_probe.to_dict(),
        "box_iou_loss": config.box_iou_loss,
        "stages": profiles,
    }


@dataclass
class TrainerState:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    text_tokens_seen: int = 0
    effective_tokens_seen: int = 0
    student_flops_seen: int = 0
    checkpoint_recompute_flops_seen: int = 0
    dense_visual_tokens_seen: int = 0
    executed_visual_tokens_seen: int = 0
    valid_visual_tokens_seen: int = 0
    visual_samples_seen: int = 0
    visual_attention_backend: str = "none"


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    global_step: int
    tokens_seen: int
    text_tokens_seen: int
    effective_tokens_seen: int
    student_flops_seen: int
    checkpoint_recompute_flops_seen: int
    executed_student_flops_seen: int
    dense_visual_tokens_seen: int
    executed_visual_tokens_seen: int
    valid_visual_tokens_seen: int
    visual_samples_seen: int
    visual_attention_backend: str
    budget_tokens_seen: int
    token_unit: str
    schedule_unit: str
    last_checkpoint: str
    final_metrics: dict[str, float]


class TokenCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_tokens: int,
        total_tokens: int,
        min_lr_ratio: float,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.min_lr_ratio = min_lr_ratio
        self.tokens_seen = 0

    def _scale(self, tokens: int) -> float:
        if self.warmup_tokens and tokens < self.warmup_tokens:
            return max(tokens, 1) / self.warmup_tokens
        progress = (
            (tokens - self.warmup_tokens)
            / max(1, self.total_tokens - self.warmup_tokens)
        )
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, tokens_seen: int) -> float:
        self.tokens_seen = int(tokens_seen)
        lr = self.base_lr * self._scale(self.tokens_seen)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class PretrainingModule(nn.Module):
    def __init__(
        self,
        student: DocumentVLMStudent,
        distillation_loss: DistillationLoss | None = None,
        box_iou_loss_kind: str = "giou",
    ):
        super().__init__()
        self.student = student
        self.distillation_loss = distillation_loss
        if box_iou_loss_kind not in BOX_IOU_LOSSES:
            raise ValueError(
                f"box_iou_loss_kind must be one of {sorted(BOX_IOU_LOSSES)}"
            )
        self.box_iou_loss_kind = box_iou_loss_kind

    def forward(
        self,
        batch: dict[str, Any],
        teacher_signals: TeacherSignals | None,
        loss_weights: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = student_model_inputs(batch)
        if teacher_signals is not None:
            if self.distillation_loss is None:
                raise ValueError("teacher signals require a distillation loss module")
            inputs["feature_layers"] = (
                self.distillation_loss.config.student_feature_layers
            )
        inputs["box_iou_loss_kind"] = self.box_iou_loss_kind
        output = self.student(**inputs)
        losses = dict(output.losses)
        if teacher_signals is not None:
            losses.update(
                self.distillation_loss(
                    output,
                    teacher_signals,
                    batch["attention_mask"],
                )
            )
        total = None
        for name, loss in losses.items():
            weight = float(loss_weights.get(name, 1.0))
            if weight == 0:
                continue
            weighted = loss * weight
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("batch produced no active pretraining loss")
        return total, losses


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _distributed_context(device_name: str) -> _DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "auto":
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return _DistributedContext(rank, world_size, local_rank, device)


def _batch_token_counts(
    batch: dict[str, Any],
    visual_tokens_per_image: int,
) -> dict[str, int]:
    supervised = int((batch["labels"] != -100).sum().item())
    text = int(batch["attention_mask"].sum().item())
    images = (
        int(batch["pixel_values"].shape[0])
        if batch.get("pixel_values") is not None
        else (
            int(batch["packed_cu_seqlens"].numel() - 1)
            if batch.get("packed_cu_seqlens") is not None
            else 0
        )
    )
    return {
        "supervised": supervised,
        "text": text,
        "effective": text + images * visual_tokens_per_image,
    }


def _all_reduce_token_counts(
    counts: dict[str, int],
    context: _DistributedContext,
) -> dict[str, int]:
    names = ("supervised", "text", "effective")
    if context.world_size == 1:
        return {name: int(counts[name]) for name in names}
    import torch.distributed as dist

    values = torch.tensor(
        [counts[name] for name in names],
        dtype=torch.long,
        device=context.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        name: int(values[index].item())
        for index, name in enumerate(names)
    }


def _state_token_count(state: TrainerState, unit: str) -> int:
    return {
        "supervised": state.tokens_seen,
        "text": state.text_tokens_seen,
        "effective": state.effective_tokens_seen,
    }[unit]


def _state_schedule_count(state: TrainerState, config: PretrainConfig) -> int:
    if config.schedule_unit == "student_flops":
        return state.student_flops_seen
    return _state_token_count(state, config.token_unit)


def _schedule_budget(config: PretrainConfig) -> tuple[int, int]:
    if config.schedule_unit == "student_flops":
        if config.total_student_flops is None:
            raise ValueError("student-FLOP schedule has no total budget")
        return config.warmup_student_flops, config.total_student_flops
    return config.warmup_tokens, config.total_tokens


def _budget_contract(config: PretrainConfig) -> dict[str, Any]:
    return {
        "stop_at_total_tokens": config.stop_at_total_tokens,
        "total_tokens": config.total_tokens,
        "token_unit": config.token_unit,
        "visual_tokens_per_image": config.visual_tokens_per_image,
        "stop_at_student_flops": config.stop_at_student_flops,
        "warmup_student_flops": config.warmup_student_flops,
        "total_student_flops": config.total_student_flops,
        "schedule_unit": config.schedule_unit,
    }


def _all_reduce_int(value: int, context: _DistributedContext) -> int:
    if context.world_size == 1:
        return int(value)
    import torch.distributed as dist

    tensor = torch.tensor(value, dtype=torch.long, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_sums(
    sums: dict[str, float],
    count: int,
    context: _DistributedContext,
) -> tuple[dict[str, float], int]:
    if context.world_size == 1:
        return sums, count
    import torch.distributed as dist

    rank_keys: list[list[str] | None] = [None] * context.world_size
    dist.all_gather_object(rank_keys, sorted(sums))
    keys = sorted({key for names in rank_keys for key in (names or [])})
    values = torch.tensor(
        [*(sums.get(key, 0.0) for key in keys), float(count)],
        dtype=torch.float64,
        device=context.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (
        {key: float(values[index].item()) for index, key in enumerate(keys)},
        int(values[-1].item()),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _set_loader_epoch(loader: Any, epoch: int, seed: int, rank: int) -> None:
    generator = getattr(loader, "generator", None)
    if generator is None:
        generator = torch.Generator()
        loader.generator = generator
    generator.manual_seed(seed + rank + epoch * 1_000_003)
    for candidate in (
        getattr(loader, "batch_sampler", None),
        getattr(loader, "sampler", None),
        getattr(loader, "collate_fn", None),
    ):
        if hasattr(candidate, "set_epoch"):
            candidate.set_epoch(epoch)


def _resolve_adaptive_sampler(loader: Any) -> Any:
    for candidate in (
        getattr(loader, "batch_sampler", None),
        getattr(loader, "sampler", None),
    ):
        if (
            hasattr(candidate, "group_names")
            and hasattr(candidate, "base_weights")
            and hasattr(candidate, "set_group_weights")
        ):
            return candidate
    raise ValueError(
        "adaptive mixture requires BalancedGroupBatchSampler"
    )


def _gradient_probe_anchors(
    student: DocumentVLMStudent,
    components: tuple[str, ...],
) -> dict[str, nn.Parameter]:
    anchors: dict[str, nn.Parameter] = {}
    if "vision" in components:
        anchors["vision"] = student.vision.norm.weight
    if "connector" in components:
        anchors["connector"] = student.connector.gradient_probe_anchor
    if "language" in components:
        anchors["language"] = student.language.norm.weight
    if any(not parameter.requires_grad for parameter in anchors.values()):
        raise ValueError(
            "gradient conflict probe anchors must remain trainable"
        )
    return anchors


def _global_probe_loss_names(
    local_names: list[str],
    context: _DistributedContext,
) -> list[str]:
    if context.world_size == 1:
        return local_names
    import torch.distributed as dist

    rank_names: list[list[str] | None] = [None] * context.world_size
    dist.all_gather_object(rank_names, local_names)
    return sorted(
        {name for names in rank_names for name in (names or [])}
    )


def _gradient_conflict_statistics(
    losses: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
    anchors: dict[str, nn.Parameter],
    context: _DistributedContext,
) -> dict[str, float]:
    local_names = sorted(
        name
        for name, loss in losses.items()
        if float(loss_weights.get(name, 1.0)) > 0 and loss.requires_grad
    )
    global_names = _global_probe_loss_names(local_names, context)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    anchor_values = tuple(anchors.values())
    for index, name in enumerate(local_names):
        raw_gradients = torch.autograd.grad(
            losses[name] * float(loss_weights.get(name, 1.0)),
            anchor_values,
            retain_graph=index + 1 < len(local_names),
            allow_unused=True,
        )
        gradients[name] = tuple(
            None if gradient is None else gradient.detach().float()
            for gradient in raw_gradients
        )

    pairs = list(combinations(global_names, 2))
    values: list[float] = []
    for name in global_names:
        values.append(
            sum(
                float(gradient.square().sum())
                for gradient in gradients.get(name, ())
                if gradient is not None
            )
        )
    for left, right in pairs:
        left_gradients = gradients.get(left, ())
        right_gradients = gradients.get(right, ())
        shared = [
            (left_gradient, right_gradient)
            for left_gradient, right_gradient in zip(
                left_gradients,
                right_gradients,
            )
            if left_gradient is not None and right_gradient is not None
        ]
        values.extend(
            [
                sum(float(left_gradient.mul(right_gradient).sum()) for left_gradient, right_gradient in shared),
                sum(float(left_gradient.square().sum()) for left_gradient, _ in shared),
                sum(float(right_gradient.square().sum()) for _, right_gradient in shared),
                float(sum(left_gradient.numel() for left_gradient, _ in shared)),
            ]
        )
    reduced = torch.tensor(
        values,
        dtype=torch.float64,
        device=context.device,
    )
    if context.world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    offset = 0
    metrics: dict[str, float] = {
        "gradient_probe/active_losses": float(len(global_names)),
        "gradient_probe/extra_forward_passes": 1.0,
        "gradient_probe/extra_backward_passes": float(len(global_names)),
        **{
            f"gradient_probe/anchor_elements/{name}": float(parameter.numel())
            for name, parameter in sorted(anchors.items())
        },
    }
    for name in global_names:
        metrics[f"gradient_probe/norm/{name}"] = math.sqrt(
            max(float(reduced[offset]), 0.0)
        )
        offset += 1
    measured_cosines: list[float] = []
    for left, right in pairs:
        dot, left_squared, right_squared, overlap = (
            float(value) for value in reduced[offset : offset + 4]
        )
        offset += 4
        pair = f"{left}__{right}"
        metrics[f"gradient_probe/overlap_elements/{pair}"] = overlap
        denominator = math.sqrt(max(left_squared * right_squared, 0.0))
        if overlap > 0 and denominator > 0:
            cosine = max(-1.0, min(1.0, dot / denominator))
            metrics[f"gradient_probe/cosine/{pair}"] = cosine
            metrics[f"gradient_probe/conflict/{pair}"] = float(cosine < 0)
            measured_cosines.append(cosine)
    metrics["gradient_probe/measured_pairs"] = float(len(measured_cosines))
    metrics["gradient_probe/negative_pair_fraction"] = (
        sum(cosine < 0 for cosine in measured_cosines)
        / len(measured_cosines)
        if measured_cosines
        else 0.0
    )
    metrics["gradient_probe/minimum_cosine"] = (
        min(measured_cosines) if measured_cosines else 0.0
    )
    return metrics


def _run_gradient_conflict_probe(
    module: PretrainingModule,
    batch: dict[str, Any],
    teacher_signals: TeacherSignals | None,
    loss_weights: dict[str, float],
    config: GradientConflictProbeConfig,
    context: _DistributedContext,
    precision: str,
) -> dict[str, float]:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        with _autocast_context(context.device, precision):
            _, losses = module(batch, teacher_signals, loss_weights)
        return _gradient_conflict_statistics(
            losses,
            loss_weights,
            _gradient_probe_anchors(module.student, config.components),
            context,
        )
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _parameter_groups(module: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    if precision == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _uses_fp16(device: torch.device, precision: str) -> bool:
    if device.type != "cuda":
        return False
    return precision == "float16" or (
        precision == "auto" and not torch.cuda.is_bf16_supported()
    )


def _append_metric(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _batch_visual_counts(
    batch: dict[str, Any],
    patch_size: int,
) -> tuple[int, int, int]:
    pixel_values = batch.get("pixel_values")
    batch_size = int(batch["input_ids"].shape[0])
    packed_pixels = batch.get("packed_pixel_values")
    packed_cu_seqlens = batch.get("packed_cu_seqlens")
    if packed_pixels is not None:
        if pixel_values is not None:
            raise ValueError(
                "visual efficiency accounting received dense and packed inputs"
            )
        if packed_cu_seqlens is None:
            raise ValueError("packed visual accounting requires cu_seqlens")
        samples = int(packed_cu_seqlens.numel() - 1)
        if samples != batch_size:
            raise ValueError("packed visual batch dimension must match input_ids")
        executed = int(packed_pixels.shape[0])
        return executed, executed, samples
    if pixel_values is None:
        return 0, 0, 0
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is None:
        raise ValueError("visual efficiency accounting requires pixel_mask")
    _, _, height, width = pixel_values.shape
    patch_height = math.ceil(height / patch_size)
    patch_width = math.ceil(width / patch_size)
    dense = int(batch_size * patch_height * patch_width)
    valid = int(
        torch.nn.functional.max_pool2d(
            pixel_mask[:, None].to(dtype=torch.float32),
            kernel_size=patch_size,
            stride=patch_size,
            ceil_mode=True,
        ).sum().item()
    )
    return dense, valid, batch_size


def _save_checkpoint(
    module: PretrainingModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: TrainerState,
    config: PretrainConfig,
    context: _DistributedContext,
    curriculum_total_steps: int,
    supervision_contract: dict[str, Any],
    adaptive_controller: AdaptiveMixtureController | None,
) -> Path:
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    target = checkpoints / f"step-{state.global_step:08d}"
    local_rng_state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    rng_states: list[dict[str, Any] | None] | None
    if context.world_size > 1:
        import torch.distributed as dist

        rng_states = [None] * context.world_size if context.is_main else None
        dist.gather_object(local_rng_state, rng_states, dst=0)
    else:
        rng_states = [local_rng_state]
    if not context.is_main:
        return target

    checkpoints.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints))
    module.student.save_pretrained(
        temporary / "student",
        metadata={
            "trainer_state": asdict(state),
            "tokenizer_fingerprint": config.tokenizer_fingerprint,
            "world_size": context.world_size,
            "run_stage": config.run_stage,
            "gradient_checkpointing": (
                module.student.gradient_checkpointing_state
            ),
            "curriculum_fingerprint": config.curriculum.fingerprint,
            "curriculum_total_steps": (
                curriculum_total_steps if config.curriculum.stages else None
            ),
            "supervision_contract": supervision_contract,
            "token_budget": {
                **_budget_contract(config),
            },
        },
    )
    payload = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "distillation_loss": (
            module.distillation_loss.state_dict()
            if module.distillation_loss is not None
            else None
        ),
        "rng_states": rng_states,
        "adaptive_mixture": (
            adaptive_controller.state_dict()
            if adaptive_controller is not None
            else None
        ),
    }
    torch.save(payload, temporary / "training_state.pt")
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    latest_temp = output / ".latest_checkpoint.tmp"
    latest_temp.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    os.replace(latest_temp, output / "latest_checkpoint.txt")
    return target


def _resolve_resume(config: PretrainConfig) -> Path | None:
    if config.resume_from is None:
        return None
    if config.resume_from != "latest":
        return Path(config.resume_from)
    latest = Path(config.output_dir) / "latest_checkpoint.txt"
    if not latest.exists():
        raise FileNotFoundError(f"no latest checkpoint pointer at {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _load_checkpoint(
    path: Path,
    module: PretrainingModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context: _DistributedContext,
    expected_tokenizer_fingerprint: str | None,
    expected_run_stage: str,
    expected_gradient_checkpointing: dict[str, Any],
    expected_curriculum_fingerprint: str | None,
    expected_curriculum_total_steps: int,
    expected_token_budget: dict[str, Any],
    expected_supervision_contract: dict[str, Any],
    adaptive_controller: AdaptiveMixtureController | None,
) -> TrainerState:
    metadata_path = path / "student" / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    saved_fingerprint = metadata.get("tokenizer_fingerprint")
    if (
        expected_tokenizer_fingerprint is not None
        and saved_fingerprint != expected_tokenizer_fingerprint
    ):
        raise ValueError(
            "resume checkpoint tokenizer fingerprint does not match the active tokenizer"
        )
    saved_run_stage = str(metadata.get("run_stage", "pretraining"))
    if saved_run_stage != expected_run_stage:
        raise ValueError(
            f"resume checkpoint run stage {saved_run_stage!r} does not match "
            f"{expected_run_stage!r}"
        )
    saved_checkpointing = metadata.get("gradient_checkpointing")
    if saved_checkpointing != expected_gradient_checkpointing:
        raise ValueError(
            "resume checkpoint gradient-checkpointing contract does not "
            "match the active training configuration"
        )
    saved_curriculum = metadata.get("curriculum_fingerprint")
    if saved_curriculum != expected_curriculum_fingerprint:
        raise ValueError(
            "resume checkpoint curriculum fingerprint does not match the active schedule"
        )
    saved_curriculum_steps = metadata.get("curriculum_total_steps")
    expected_steps = (
        expected_curriculum_total_steps
        if expected_curriculum_fingerprint is not None
        else None
    )
    if saved_curriculum_steps != expected_steps:
        raise ValueError(
            "resume checkpoint curriculum horizon does not match the active training plan"
        )
    saved_token_budget = metadata.get("token_budget")
    if saved_token_budget != expected_token_budget:
        raise ValueError(
            "resume checkpoint token-budget contract does not match the active training plan"
        )
    if metadata.get("supervision_contract") != expected_supervision_contract:
        raise ValueError(
            "resume checkpoint supervision contract does not match "
            "the active training plan"
        )
    saved_world_size = int(metadata.get("world_size", 1))
    if saved_world_size != context.world_size:
        raise ValueError(
            f"exact resume requires world_size={saved_world_size}, "
            f"received {context.world_size}"
        )
    student_state = torch.load(
        path / "student" / "model.pt",
        map_location=context.device,
        weights_only=True,
    )
    module.student.load_state_dict(student_state)
    payload = torch.load(
        path / "training_state.pt",
        map_location=context.device,
        weights_only=False,
    )
    saved_adaptive = payload.get("adaptive_mixture")
    if adaptive_controller is None:
        if saved_adaptive is not None:
            raise ValueError(
                "resume checkpoint contains unexpected adaptive mixture state"
            )
    else:
        if saved_adaptive is None:
            raise ValueError(
                "resume checkpoint has no adaptive mixture state"
            )
        adaptive_controller.load_state_dict(saved_adaptive)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    if module.distillation_loss is not None:
        state = payload.get("distillation_loss")
        if state is None:
            raise ValueError("checkpoint has no distillation projector state")
        module.distillation_loss.load_state_dict(state)
    rng_states = payload.get("rng_states")
    if rng_states is None or len(rng_states) != context.world_size:
        raise ValueError("checkpoint does not contain RNG state for every active rank")
    rank_rng = rng_states[context.rank]
    if rank_rng is None:
        raise ValueError(f"checkpoint has no RNG state for rank {context.rank}")
    torch.set_rng_state(rank_rng["torch"].cpu())
    random.setstate(rank_rng["python"])
    if torch.cuda.is_available() and rank_rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rank_rng["cuda"])
    trainer_state = json.loads(
        (path / "trainer_state.json").read_text(encoding="utf-8")
    )
    trainer_state.setdefault(
        "executed_visual_tokens_seen",
        trainer_state.get("dense_visual_tokens_seen", 0),
    )
    return TrainerState(**trainer_state)


@torch.no_grad()
def _evaluate(
    student: DocumentVLMStudent,
    loaders: dict[str, Iterable[dict[str, Any]]],
    context: _DistributedContext,
    loss_weights: dict[str, float],
    precision: str,
) -> dict[str, float]:
    was_training = student.training
    student.eval()
    metrics: dict[str, float] = {}
    for name, loader in loaders.items():
        sums: dict[str, float] = {}
        samples = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, context.device)
            inputs = student_model_inputs(batch)
            inputs["contrastive"] = False
            with _autocast_context(context.device, precision):
                output = student(**inputs)
            batch_samples = int(batch["input_ids"].shape[0])
            for loss_name, value in output.losses.items():
                sums[loss_name] = (
                    sums.get(loss_name, 0.0) + float(value) * batch_samples
                )
            samples += batch_samples
        sums, samples = _all_reduce_sums(sums, samples, context)
        if samples:
            for loss_name, total in sums.items():
                metrics[f"eval/{name}/{loss_name}"] = total / samples
            metrics[f"eval/{name}/weighted_loss"] = sum(
                metrics[f"eval/{name}/{loss_name}"]
                * float(loss_weights.get(loss_name, 1.0))
                for loss_name in sums
            )
    student.train(was_training)
    return metrics


def train_student(
    student: DocumentVLMStudent,
    train_loader: Any,
    config: PretrainConfig,
    *,
    teacher: NativeStudentTeacher | None = None,
    distillation_loss: DistillationLoss | None = None,
    eval_loaders: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> TrainingResult:
    """Train or resume a student. Use ``torchrun`` plus a distributed balanced sampler for DDP."""

    if (teacher is None) != (distillation_loss is None):
        raise ValueError("teacher and distillation_loss must be provided together")
    supervision_contract = pretraining_supervision_contract(
        config,
        has_online_teacher=teacher is not None,
    )
    if getattr(train_loader, "persistent_workers", False):
        raise ValueError("exact-resume augmentation requires persistent_workers=False")
    adaptive_sampler = None
    adaptive_controller = None
    if config.adaptive_mixture.enabled:
        adaptive_sampler = _resolve_adaptive_sampler(train_loader)
        adaptive_controller = AdaptiveMixtureController(
            config.adaptive_mixture,
            adaptive_sampler.base_weights,
        )
        eval_groups = set((eval_loaders or {}).keys())
        sampler_groups = set(adaptive_sampler.group_names)
        if eval_groups != sampler_groups:
            raise ValueError(
                "adaptive mixture eval groups must match sampler groups: "
                f"missing={sorted(sampler_groups - eval_groups)}, "
                f"extra={sorted(eval_groups - sampler_groups)}"
            )
        adaptive_sampler.set_group_weights(
            adaptive_controller.weights
        )
    context = _distributed_context(config.device)
    random.seed(config.seed + context.rank)
    torch.manual_seed(config.seed + context.rank)
    output_dir = Path(config.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    student.configure_gradient_checkpointing(
        enabled=config.gradient_checkpointing,
        components=config.gradient_checkpointing_components,
        use_reentrant=(
            config.gradient_checkpointing_use_reentrant
        ),
    )
    module = PretrainingModule(
        student,
        distillation_loss,
        config.box_iou_loss,
    ).to(context.device)
    if teacher is not None:
        teacher.model.to(context.device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(module, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=_uses_fp16(context.device, config.precision),
    )
    schedule_warmup, schedule_total = _schedule_budget(config)
    scheduler = TokenCosineScheduler(
        optimizer,
        config.learning_rate,
        schedule_warmup,
        schedule_total,
        config.min_lr_ratio,
    )
    if config.curriculum.unit == "training_token_fraction":
        curriculum_horizon = config.total_tokens
    elif config.curriculum.unit == "training_compute_fraction":
        if config.total_student_flops is None:
            raise ValueError("compute curriculum has no total FLOP budget")
        curriculum_horizon = config.total_student_flops
    else:
        if config.epochs is None:
            curriculum_horizon = 1
        else:
            curriculum_horizon = planned_optimizer_steps(
                num_batches=len(train_loader),
                grad_accum_steps=config.grad_accum_steps,
                epochs=config.epochs,
                max_steps=config.max_steps,
            )
    state = TrainerState()
    resume_path = _resolve_resume(config)
    if resume_path is not None:
        state = _load_checkpoint(
            resume_path,
            module,
            optimizer,
            scaler,
            context,
            config.tokenizer_fingerprint,
            config.run_stage,
            module.student.gradient_checkpointing_state,
            config.curriculum.fingerprint,
            curriculum_horizon,
            _budget_contract(config),
            supervision_contract,
            adaptive_controller,
        )
        if adaptive_controller is not None:
            adaptive_sampler.set_group_weights(
                adaptive_controller.weights
            )
        scheduler.step(_state_schedule_count(state, config))

    wrapped: nn.Module = module
    if context.world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(
            module,
            device_ids=(
                [context.local_rank] if context.device.type == "cuda" else None
            ),
            find_unused_parameters=True,
        )

    optimizer.zero_grad(set_to_none=True)
    last_checkpoint = resume_path
    final_metrics: dict[str, float] = {}
    stop = (
        config.stop_at_total_tokens
        and _state_token_count(state, config.token_unit) >= config.total_tokens
    ) or (
        config.stop_at_student_flops
        and config.total_student_flops is not None
        and state.student_flops_seen >= config.total_student_flops
    ) or (
        config.max_steps is not None
        and state.global_step >= config.max_steps
    )
    accumulated_token_counts = {
        "supervised": 0,
        "text": 0,
        "effective": 0,
    }
    accumulated_losses: dict[str, float] = {}
    accumulated_microbatches = 0
    accumulated_student_flops = 0
    accumulated_checkpoint_recompute_flops = 0
    accumulated_dense_visual_tokens = 0
    accumulated_valid_visual_tokens = 0
    accumulated_samples = 0
    epoch = state.epoch
    while not stop and (config.epochs is None or epoch < config.epochs):
        if (
            adaptive_controller is not None
            and state.batch_in_epoch == 0
        ):
            had_pending = adaptive_controller.pending
            weights_changed = adaptive_controller.apply_pending()
            adaptive_sampler.set_group_weights(
                adaptive_controller.weights
            )
            if context.is_main and (
                had_pending
                or (state.global_step == 0 and epoch == 0)
            ):
                _append_metric(
                    output_dir,
                    {
                        "kind": "adaptive_mixture",
                        "train/global_step": state.global_step,
                        "train/epoch": epoch,
                        "adaptive/weights_changed": weights_changed,
                        "adaptive/evaluations": (
                            adaptive_controller.evaluations
                        ),
                        "adaptive/updates": adaptive_controller.updates,
                        **{
                            f"adaptive/group_weight/{group}": weight
                            for group, weight in sorted(
                                adaptive_controller.weights.items()
                            )
                        },
                        **{
                            f"adaptive/heldout_loss_ema/{group}": loss
                            for group, loss in sorted(
                                adaptive_controller.ema_losses.items()
                            )
                        },
                    },
                )
        _set_loader_epoch(train_loader, epoch, config.seed, context.rank)
        module.train()
        loader_length = len(train_loader)
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == state.epoch and batch_index < state.batch_in_epoch:
                continue
            batch = _move_batch(raw_batch, context.device)
            if config.curriculum.unit in {
                "training_token_fraction",
                "training_compute_fraction",
            }:
                progress_count = (
                    state.student_flops_seen
                    if config.curriculum.unit == "training_compute_fraction"
                    else _state_token_count(state, config.token_unit)
                )
                curriculum_progress = min(
                    progress_count / curriculum_horizon,
                    1.0,
                )
                curriculum_stage = config.curriculum.stage_for_fraction(
                    curriculum_progress
                )
                active_loss_weights = (
                    config.curriculum.loss_weights_for_fraction(
                        config.loss_weights,
                        curriculum_progress,
                    )
                )
            else:
                curriculum_stage = config.curriculum.stage_for_step(
                    state.global_step,
                    curriculum_horizon,
                )
                active_loss_weights = config.curriculum.loss_weights_for_step(
                    config.loss_weights,
                    state.global_step,
                    curriculum_horizon,
                )
                curriculum_progress = (
                    min(state.global_step, curriculum_horizon - 1)
                    / curriculum_horizon
                )
            is_last_batch = batch_index + 1 == loader_length
            microbatch_number = accumulated_microbatches + 1
            should_step = (
                microbatch_number >= config.grad_accum_steps or is_last_batch
            )
            sync_context = (
                wrapped.no_sync()
                if context.world_size > 1 and not should_step
                else nullcontext()
            )
            gradient_probe_metrics = None
            with sync_context:
                with _autocast_context(context.device, config.precision):
                    online_teacher_active = any(
                        active_loss_weights.get(name, 0.0) > 0
                        for name in _ONLINE_TEACHER_LOSSES
                    )
                    teacher_signals = (
                        teacher(batch)
                        if teacher is not None and online_teacher_active
                        else None
                    )
                if (
                    config.gradient_conflict_probe.enabled
                    and accumulated_microbatches == 0
                    and state.global_step
                    % config.gradient_conflict_probe.every_steps
                    == 0
                ):
                    gradient_probe_metrics = (
                        _run_gradient_conflict_probe(
                            module,
                            batch,
                            teacher_signals,
                            active_loss_weights,
                            config.gradient_conflict_probe,
                            context,
                            config.precision,
                        )
                    )
                with _autocast_context(context.device, config.precision):
                    total, losses = wrapped(
                        batch,
                        teacher_signals,
                        active_loss_weights,
                    )
                    scaled_loss = total / config.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            if context.is_main and gradient_probe_metrics is not None:
                _append_metric(
                    output_dir,
                    {
                        "kind": "gradient_conflict",
                        "train/global_step": state.global_step + 1,
                        "train/epoch": epoch,
                        "gradient_probe/batch_index": batch_index,
                        **gradient_probe_metrics,
                    },
                )
            accumulated_microbatches = microbatch_number
            state.visual_attention_backend = (
                module.student.last_visual_attention_backend
            )
            batch_token_counts = _batch_token_counts(
                batch,
                config.visual_tokens_per_image,
            )
            batch_flops = estimate_batch_training_flops_breakdown(
                module.student.config,
                batch,
                checkpoint_components=(
                    config.gradient_checkpointing_components
                    if config.gradient_checkpointing
                    else ()
                ),
            )
            accumulated_student_flops += batch_flops.algorithmic
            accumulated_checkpoint_recompute_flops += (
                batch_flops.checkpoint_recompute
            )
            dense_visual, valid_visual, visual_samples = _batch_visual_counts(
                batch,
                module.student.config.vision.patch_size,
            )
            accumulated_dense_visual_tokens += dense_visual
            accumulated_valid_visual_tokens += valid_visual
            accumulated_samples += visual_samples
            for name, count in batch_token_counts.items():
                accumulated_token_counts[name] += count
            for name, value in losses.items():
                accumulated_losses[name] = (
                    accumulated_losses.get(name, 0.0) + float(value.detach())
                )
            state.batch_in_epoch = batch_index + 1
            if not should_step:
                continue

            global_token_counts = _all_reduce_token_counts(
                accumulated_token_counts,
                context,
            )
            state.tokens_seen += global_token_counts["supervised"]
            state.text_tokens_seen += global_token_counts["text"]
            state.effective_tokens_seen += global_token_counts["effective"]
            budget_tokens_seen = _state_token_count(state, config.token_unit)
            global_student_flops = _all_reduce_int(
                accumulated_student_flops,
                context,
            )
            state.student_flops_seen += global_student_flops
            global_checkpoint_recompute_flops = _all_reduce_int(
                accumulated_checkpoint_recompute_flops,
                context,
            )
            state.checkpoint_recompute_flops_seen += (
                global_checkpoint_recompute_flops
            )
            global_dense_visual_tokens = _all_reduce_int(
                accumulated_dense_visual_tokens,
                context,
            )
            global_valid_visual_tokens = _all_reduce_int(
                accumulated_valid_visual_tokens,
                context,
            )
            global_samples = _all_reduce_int(accumulated_samples, context)
            state.dense_visual_tokens_seen += global_dense_visual_tokens
            state.executed_visual_tokens_seen += global_dense_visual_tokens
            state.valid_visual_tokens_seen += global_valid_visual_tokens
            state.visual_samples_seen += global_samples
            schedule_count = _state_schedule_count(state, config)
            learning_rate = scheduler.step(schedule_count)
            scaler.unscale_(optimizer)
            if accumulated_microbatches < config.grad_accum_steps:
                correction = config.grad_accum_steps / accumulated_microbatches
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            global_loss_sums, global_microbatches = _all_reduce_sums(
                accumulated_losses,
                accumulated_microbatches,
                context,
            )
            means = {
                name: total_value / global_microbatches
                for name, total_value in global_loss_sums.items()
            }
            means.update(
                {
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": float(gradient_norm),
                    "train/tokens_seen": float(state.tokens_seen),
                    "train/text_tokens_seen": float(state.text_tokens_seen),
                    "train/effective_tokens_seen": float(
                        state.effective_tokens_seen
                    ),
                    "train/budget_tokens_seen": float(budget_tokens_seen),
                    "train/student_flops_seen": float(
                        state.student_flops_seen
                    ),
                    "train/checkpoint_recompute_flops_seen": float(
                        state.checkpoint_recompute_flops_seen
                    ),
                    "train/executed_student_flops_seen": float(
                        state.student_flops_seen
                        + state.checkpoint_recompute_flops_seen
                    ),
                    "train/dense_visual_tokens_per_sample": (
                        state.dense_visual_tokens_seen
                        / state.visual_samples_seen
                        if state.visual_samples_seen
                        else 0.0
                    ),
                    "train/executed_visual_tokens_per_sample": (
                        state.executed_visual_tokens_seen
                        / state.visual_samples_seen
                        if state.visual_samples_seen
                        else 0.0
                    ),
                    "train/valid_visual_token_fraction": (
                        state.valid_visual_tokens_seen
                        / state.dense_visual_tokens_seen
                        if state.dense_visual_tokens_seen
                        else 0.0
                    ),
                    "train/schedule_count": float(schedule_count),
                    "train/global_step": float(state.global_step),
                    "train/curriculum_stage": (
                        curriculum_stage.id if curriculum_stage is not None else "base"
                    ),
                    "train/curriculum_progress": curriculum_progress,
                    "train/box_iou_loss": config.box_iou_loss,
                    "train/visual_attention_backend": (
                        module.student.last_visual_attention_backend
                    ),
                    **{
                        f"train/loss_weight/{name}": weight
                        for name, weight in sorted(active_loss_weights.items())
                    },
                }
            )
            if adaptive_controller is not None:
                means.update(
                    {
                        f"train/group_weight/{group}": weight
                        for group, weight in sorted(
                            adaptive_controller.weights.items()
                        )
                    }
                )
            accumulated_token_counts = {
                "supervised": 0,
                "text": 0,
                "effective": 0,
            }
            accumulated_losses = {}
            accumulated_microbatches = 0
            accumulated_student_flops = 0
            accumulated_checkpoint_recompute_flops = 0
            accumulated_dense_visual_tokens = 0
            accumulated_valid_visual_tokens = 0
            accumulated_samples = 0

            if context.is_main and (
                state.global_step == 1
                or state.global_step % config.log_every_steps == 0
            ):
                _append_metric(output_dir, {"kind": "train", **means})
                print(
                    f"[student] step={state.global_step} "
                    f"{config.token_unit}_tokens={budget_tokens_seen:,} "
                    f"student_flops={state.student_flops_seen:,} "
                    f"loss={sum(means.get(name, 0.0) * active_loss_weights.get(name, 1.0) for name in losses):.4f} "
                    f"curriculum={means['train/curriculum_stage']} "
                    f"lr={learning_rate:.3e}",
                    flush=True,
                )
            if (
                eval_loaders
                and config.eval_every_steps > 0
                and state.global_step % config.eval_every_steps == 0
            ):
                final_metrics = _evaluate(
                    module.student,
                    eval_loaders,
                    context,
                    active_loss_weights,
                    config.precision,
                )
                if adaptive_controller is not None:
                    adaptive_controller.observe(
                        {
                            group: final_metrics[
                                f"eval/{group}/weighted_loss"
                            ]
                            for group in adaptive_controller.groups
                        }
                    )
                if context.is_main:
                    _append_metric(
                        output_dir,
                        {
                            "kind": "eval",
                            "train/global_step": state.global_step,
                            **(
                                {
                                    "adaptive/pending_update": True,
                                    **{
                                        (
                                            "adaptive/heldout_loss/"
                                            f"{group}"
                                        ): final_metrics[
                                            f"eval/{group}/weighted_loss"
                                        ]
                                        for group in adaptive_controller.groups
                                    },
                                }
                                if adaptive_controller is not None
                                else {}
                            ),
                            **final_metrics,
                        },
                    )
                module.train()
            if is_last_batch:
                state.epoch = epoch + 1
                state.batch_in_epoch = 0
            if (
                config.checkpoint_every_steps > 0
                and state.global_step % config.checkpoint_every_steps == 0
            ):
                last_checkpoint = _save_checkpoint(
                    module,
                    optimizer,
                    scaler,
                    state,
                    config,
                    context,
                    curriculum_horizon,
                    supervision_contract,
                    adaptive_controller,
                )
            reached_step_limit = (
                config.max_steps is not None
                and state.global_step >= config.max_steps
            )
            reached_token_limit = (
                config.stop_at_total_tokens
                and budget_tokens_seen >= config.total_tokens
            )
            reached_compute_limit = (
                config.stop_at_student_flops
                and config.total_student_flops is not None
                and state.student_flops_seen >= config.total_student_flops
            )
            if (
                reached_step_limit
                or reached_token_limit
                or reached_compute_limit
            ):
                stop = True
                break
        if stop:
            break
        epoch += 1
        state.epoch = epoch
        state.batch_in_epoch = 0

    if (
        last_checkpoint is None
        or last_checkpoint.name != f"step-{state.global_step:08d}"
    ):
        last_checkpoint = _save_checkpoint(
            module,
            optimizer,
            scaler,
            state,
            config,
            context,
            curriculum_horizon,
            supervision_contract,
            adaptive_controller,
        )
    budget_tokens_seen = _state_token_count(state, config.token_unit)
    if (
        config.stop_at_total_tokens
        and config.max_steps is None
        and budget_tokens_seen < config.total_tokens
    ):
        raise RuntimeError(
            f"training exhausted epochs at {budget_tokens_seen:,} "
            f"{config.token_unit} tokens before total_tokens="
            f"{config.total_tokens:,}"
        )
    if (
        config.stop_at_student_flops
        and config.max_steps is None
        and config.total_student_flops is not None
        and state.student_flops_seen < config.total_student_flops
    ):
        raise RuntimeError(
            f"training exhausted epochs at {state.student_flops_seen:,} "
            "student FLOPs before total_student_flops="
            f"{config.total_student_flops:,}"
        )
    if context.world_size > 1:
        import torch.distributed as dist

        dist.barrier()
    if last_checkpoint is None:
        last_checkpoint = Path(config.output_dir) / "checkpoints" / (
            f"step-{state.global_step:08d}"
        )
    return TrainingResult(
        output_dir=str(output_dir),
        global_step=state.global_step,
        tokens_seen=state.tokens_seen,
        text_tokens_seen=state.text_tokens_seen,
        effective_tokens_seen=state.effective_tokens_seen,
        student_flops_seen=state.student_flops_seen,
        checkpoint_recompute_flops_seen=(
            state.checkpoint_recompute_flops_seen
        ),
        executed_student_flops_seen=(
            state.student_flops_seen
            + state.checkpoint_recompute_flops_seen
        ),
        dense_visual_tokens_seen=state.dense_visual_tokens_seen,
        executed_visual_tokens_seen=state.executed_visual_tokens_seen,
        valid_visual_tokens_seen=state.valid_visual_tokens_seen,
        visual_samples_seen=state.visual_samples_seen,
        visual_attention_backend=state.visual_attention_backend,
        budget_tokens_seen=budget_tokens_seen,
        token_unit=config.token_unit,
        schedule_unit=config.schedule_unit,
        last_checkpoint=str(last_checkpoint),
        final_metrics=final_metrics,
    )
