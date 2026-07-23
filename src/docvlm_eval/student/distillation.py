"""Efficient online distillation for teachers that share the student tokenizer contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentConfig
from .data import visual_model_inputs
from .model import DocumentVLMStudent, StudentOutput


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    logit_top_k: int = 128
    vision_layer_pairs: tuple[tuple[int, int], ...] = ()
    language_layer_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("distillation temperature must be positive")
        if self.logit_top_k < 0:
            raise ValueError("logit_top_k must be non-negative")

    @classmethod
    def from_blueprint(cls, blueprint: dict[str, Any]) -> "DistillationConfig":
        raw = blueprint["training"]["pretraining"]["distillation"]
        return cls(
            temperature=float(raw["temperature"]),
            logit_top_k=int(raw["logit_top_k"]),
            vision_layer_pairs=tuple(
                (int(pair[0]), int(pair[1]))
                for pair in raw.get("vision_layer_pairs", [])
            ),
            language_layer_pairs=tuple(
                (int(pair[0]), int(pair[1]))
                for pair in raw.get("language_layer_pairs", [])
            ),
        )

    @property
    def student_feature_layers(self) -> dict[str, list[int]]:
        return {
            "vision": sorted({pair[0] for pair in self.vision_layer_pairs}),
            "language": sorted({pair[0] for pair in self.language_layer_pairs}),
        }

    @property
    def teacher_feature_layers(self) -> dict[str, list[int]]:
        return {
            "vision": sorted({pair[1] for pair in self.vision_layer_pairs}),
            "language": sorted({pair[1] for pair in self.language_layer_pairs}),
        }


@dataclass
class TeacherSignals:
    """Compressed teacher targets retained only for the current optimization step."""

    token_mask: torch.Tensor
    topk_indices: torch.Tensor | None = None
    bucket_logits: torch.Tensor | None = None
    full_logits: torch.Tensor | None = None
    vision_features: dict[int, torch.Tensor] = field(default_factory=dict)
    language_features: dict[int, torch.Tensor] = field(default_factory=dict)
    vision_mask: torch.Tensor | None = None


def _other_logsumexp(
    logits: torch.Tensor,
    selected_logits: torch.Tensor,
) -> torch.Tensor:
    all_lse = torch.logsumexp(logits, dim=-1)
    selected_lse = torch.logsumexp(selected_logits, dim=-1)
    ratio = torch.exp(selected_lse - all_lse).clamp(max=1.0 - 1e-7)
    return all_lse + torch.log1p(-ratio)


def _compress_teacher_logits(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    selected = logits[token_mask].float() / temperature
    if top_k == 0:
        return None, None, selected
    if top_k >= selected.shape[-1]:
        raise ValueError("logit_top_k must be smaller than the teacher vocabulary")
    top_values, top_indices = torch.topk(selected, top_k, dim=-1)
    other = _other_logsumexp(selected, top_values)
    return top_indices, torch.cat((top_values, other[:, None]), dim=-1), None


class NativeStudentTeacher:
    """Frozen native teacher with immediate top-k logit compression."""

    def __init__(
        self,
        model: DocumentVLMStudent,
        config: DistillationConfig,
    ):
        self.model = model.eval()
        self.config = config
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> TeacherSignals:
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            **visual_model_inputs(batch),
        }
        output = self.model(
            **inputs,
            feature_layers=self.config.teacher_feature_layers,
        )
        text_length = batch["input_ids"].shape[1]
        token_mask = batch["labels"] != self.model.config.ignore_index
        topk_indices, bucket_logits, full_logits = _compress_teacher_logits(
            output.logits[:, -text_length:],
            token_mask,
            top_k=self.config.logit_top_k,
            temperature=self.config.temperature,
        )
        return TeacherSignals(
            token_mask=token_mask,
            topk_indices=topk_indices,
            bucket_logits=bucket_logits,
            full_logits=full_logits,
            vision_features={
                layer: value.detach()
                for layer, value in output.vision_features.items()
            },
            language_features={
                layer: value[:, -text_length:].detach()
                for layer, value in output.language_features.items()
            },
            vision_mask=(
                output.vision_mask.detach()
                if output.vision_mask is not None
                else None
            ),
        )


class DistillationLoss(nn.Module):
    """Project incompatible widths and compute top-k KL plus selected feature losses."""

    def __init__(
        self,
        student_config: StudentConfig,
        teacher_config: StudentConfig,
        config: DistillationConfig,
    ):
        super().__init__()
        if student_config.language.vocab_size != teacher_config.language.vocab_size:
            raise ValueError(
                "online logit distillation requires an identical tokenizer vocabulary"
            )
        self.config = config
        self.language_projections = nn.ModuleDict()
        self.vision_projections = nn.ModuleDict()
        for student_layer, teacher_layer in config.language_layer_pairs:
            self.language_projections[self._pair_key(student_layer, teacher_layer)] = (
                nn.Identity()
                if student_config.language.width == teacher_config.language.width
                else nn.Linear(
                    student_config.language.width,
                    teacher_config.language.width,
                    bias=False,
                )
            )
        for student_layer, teacher_layer in config.vision_layer_pairs:
            self.vision_projections[self._pair_key(student_layer, teacher_layer)] = (
                nn.Identity()
                if student_config.vision.width == teacher_config.vision.width
                else nn.Linear(
                    student_config.vision.width,
                    teacher_config.vision.width,
                    bias=False,
                )
            )

    @staticmethod
    def _pair_key(student_layer: int, teacher_layer: int) -> str:
        return f"s{student_layer}_t{teacher_layer}".replace("-", "final")

    def _logit_loss(
        self,
        student_logits: torch.Tensor,
        signals: TeacherSignals,
    ) -> torch.Tensor:
        selected = student_logits[signals.token_mask].float() / self.config.temperature
        if signals.full_logits is not None:
            return (
                F.kl_div(
                    F.log_softmax(selected, dim=-1),
                    F.softmax(signals.full_logits, dim=-1),
                    reduction="batchmean",
                )
                * self.config.temperature**2
            )
        if signals.topk_indices is None or signals.bucket_logits is None:
            raise ValueError("teacher signals contain no logit target")
        top_values = torch.gather(selected, -1, signals.topk_indices)
        other = _other_logsumexp(selected, top_values)
        student_buckets = torch.cat((top_values, other[:, None]), dim=-1)
        return (
            F.kl_div(
                F.log_softmax(student_buckets, dim=-1),
                F.softmax(signals.bucket_logits, dim=-1),
                reduction="batchmean",
            )
            * self.config.temperature**2
        )

    @staticmethod
    def _cosine_feature_loss(
        student: torch.Tensor,
        teacher: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if student.shape[:-1] != teacher.shape[:-1]:
            raise ValueError(
                "student and teacher feature sequences must align before width projection"
            )
        loss = 1.0 - (
            F.normalize(student.float(), dim=-1)
            * F.normalize(teacher.float(), dim=-1)
        ).sum(dim=-1)
        if mask is None:
            return loss.mean()
        valid = mask.to(device=loss.device, dtype=torch.bool)
        if valid.shape != loss.shape:
            raise ValueError("feature mask does not match feature sequence")
        return loss[valid].mean()

    def forward(
        self,
        student_output: StudentOutput,
        signals: TeacherSignals,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_length = attention_mask.shape[1]
        losses = {
            "teacher_kl": self._logit_loss(
                student_output.logits[:, -text_length:],
                signals,
            )
        }
        feature_losses: list[torch.Tensor] = []
        for student_layer, teacher_layer in self.config.language_layer_pairs:
            projected = self.language_projections[
                self._pair_key(student_layer, teacher_layer)
            ](student_output.language_features[student_layer][:, -text_length:])
            feature_losses.append(
                self._cosine_feature_loss(
                    projected,
                    signals.language_features[teacher_layer],
                    attention_mask,
                )
            )
        for student_layer, teacher_layer in self.config.vision_layer_pairs:
            student_has = student_layer in student_output.vision_features
            teacher_has = teacher_layer in signals.vision_features
            if not student_has and not teacher_has:
                continue
            if not student_has or not teacher_has:
                raise ValueError(
                    "student and teacher must expose the same requested vision features"
                )
            projected = self.vision_projections[
                self._pair_key(student_layer, teacher_layer)
            ](student_output.vision_features[student_layer])
            feature_losses.append(
                self._cosine_feature_loss(
                    projected,
                    signals.vision_features[teacher_layer],
                    signals.vision_mask,
                )
            )
        if feature_losses:
            losses["hidden_feature_distillation"] = torch.stack(
                feature_losses
            ).mean()
        return losses
