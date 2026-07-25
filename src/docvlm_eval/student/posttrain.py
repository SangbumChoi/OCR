"""Grounded SFT and single-update GRPO for the native document VLM."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from ..generation_policy import (
    resolve_generation_token_budget,
    validate_generation_token_budget_policy,
)
from ..schema import Sample
from .data import (
    DeterministicDistributedBatchSampler,
    StudentCollator,
    StudentExample,
    student_model_inputs,
    visual_model_inputs,
)
from .compute import estimate_preference_step_flops, estimate_rlvr_step_flops
from .generation import repeated_suffix_cycle_mask
from .model import (
    DocumentVLMStudent,
    validate_checkpoint_initialization_lineage,
)
from .optim import (
    OptimizerSpec,
    build_optimizer,
    optimizer_runtime_contract,
)
from .pretrain import (
    PretrainConfig,
    TrainingResult,
    _autocast_context,
    _parameter_groups,
    _record_metric,
    _uses_fp16,
    train_student,
)
from .rewards import (
    RewardConfig,
    RewardContext,
    RewardResult,
    build_structured_target,
    score_structured_response,
)


STRUCTURED_RESPONSE_INSTRUCTION = (
    "Return exactly one JSON object with keys answer, evidence, and rationale. "
    "answer must be a string. evidence must be a list of normalized [x1,y1,x2,y2] "
    "boxes, or an empty list. rationale must be a concise evidence-linked string, "
    "or an empty string. Do not add text outside the JSON object."
)


@dataclass(frozen=True)
class SFTConfig:
    output_dir: str
    target_mode: str = "evidence_linked"
    epochs: int | None = 2
    max_steps: int | None = None
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 2e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    warmup_tokens: int = 10_000_000
    total_tokens: int = 1_000_000_000
    stop_at_total_tokens: bool = False
    warmup_student_flops: int = 0
    total_student_flops: int | None = None
    stop_at_student_flops: bool = False
    schedule_unit: str = "tokens"
    max_grad_norm: float = 1.0
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False
    gradient_checkpointing_components: tuple[str, ...] = (
        "vision",
        "connector",
        "language",
    )
    gradient_checkpointing_use_reentrant: bool = False
    checkpoint_every_steps: int = 500
    eval_every_steps: int = 0
    log_every_steps: int = 10
    num_workers: int = 4
    seed: int = 17
    device: str = "auto"
    resume_from: str | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.target_mode not in {
            "answer_only",
            "free_rationale",
            "evidence_linked",
        }:
            raise ValueError("unsupported SFT target_mode")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("SFT batch_size must be positive and num_workers non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("SFT optimizer betas must be within [0, 1)")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("SFT epochs must be positive when set")
        if self.epochs is None and not (
            self.stop_at_total_tokens or self.stop_at_student_flops
        ):
            raise ValueError(
                "SFT epochs can be null only when a budget stop is active"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("SFT max_steps must be positive when set")
        if self.total_student_flops is not None and self.total_student_flops <= 0:
            raise ValueError("SFT total_student_flops must be positive")
        if self.warmup_student_flops < 0:
            raise ValueError("SFT warmup_student_flops must be non-negative")
        if self.stop_at_student_flops and self.total_student_flops is None:
            raise ValueError(
                "SFT student-FLOP stop requires total_student_flops"
            )
        if self.schedule_unit not in {"tokens", "student_flops"}:
            raise ValueError("SFT schedule_unit must be tokens or student_flops")
        if self.schedule_unit == "student_flops" and (
            self.total_student_flops is None
            or not (
                0
                <= self.warmup_student_flops
                < self.total_student_flops
            )
        ):
            raise ValueError("SFT student-FLOP schedule is invalid")
        if (
            not self.gradient_checkpointing_components
            or len(set(self.gradient_checkpointing_components))
            != len(self.gradient_checkpointing_components)
            or not set(self.gradient_checkpointing_components)
            <= {"vision", "connector", "language"}
        ):
            raise ValueError(
                "SFT gradient checkpointing components are invalid"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        output_dir: str | Path,
        **overrides: Any,
    ) -> "SFTConfig":
        raw = blueprint["training"]["posttraining"]["sft"]
        optimizer = raw["optimizer"]
        checkpointing = blueprint["training"]["activation_checkpointing"]
        values = {
            "output_dir": str(output_dir),
            "target_mode": str(raw["target_mode"]),
            "epochs": (
                None
                if optimizer.get("epochs") is None
                else int(optimizer["epochs"])
            ),
            "max_steps": (
                None
                if optimizer.get("max_steps") is None
                else int(optimizer["max_steps"])
            ),
            "batch_size": int(optimizer["micro_batch_size"]),
            "grad_accum_steps": int(optimizer["grad_accum_steps"]),
            "learning_rate": float(optimizer["learning_rate"]),
            "min_lr_ratio": float(optimizer["min_lr_ratio"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "beta1": float(optimizer["betas"][0]),
            "beta2": float(optimizer["betas"][1]),
            "optimizer": OptimizerSpec.from_mapping(optimizer),
            "warmup_tokens": int(optimizer["warmup_tokens"]),
            "total_tokens": int(optimizer["total_tokens"]),
            "stop_at_total_tokens": bool(
                optimizer.get("stop_at_total_tokens", False)
            ),
            "warmup_student_flops": int(
                optimizer.get("warmup_student_flops", 0)
            ),
            "total_student_flops": (
                None
                if optimizer.get("total_student_flops") is None
                else int(optimizer["total_student_flops"])
            ),
            "stop_at_student_flops": bool(
                optimizer.get("stop_at_student_flops", False)
            ),
            "schedule_unit": str(
                optimizer.get("schedule_unit", "tokens")
            ),
            "max_grad_norm": float(optimizer["max_grad_norm"]),
            "precision": str(optimizer["precision"]),
            "gradient_checkpointing": bool(
                checkpointing["enabled"]
            ),
            "gradient_checkpointing_components": tuple(
                str(value) for value in checkpointing["components"]
            ),
            "gradient_checkpointing_use_reentrant": bool(
                checkpointing["use_reentrant"]
            ),
            "checkpoint_every_steps": int(optimizer["checkpoint_every_steps"]),
            "eval_every_steps": int(optimizer["eval_every_steps"]),
            "log_every_steps": int(optimizer["log_every_steps"]),
            "num_workers": int(optimizer["num_workers"]),
            "seed": int(optimizer["seed"]),
        }
        values.update(overrides)
        return cls(**values)

    def as_pretrain_config(self) -> PretrainConfig:
        return PretrainConfig(
            output_dir=self.output_dir,
            epochs=self.epochs,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            min_lr_ratio=self.min_lr_ratio,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            optimizer=self.optimizer,
            warmup_tokens=self.warmup_tokens,
            total_tokens=self.total_tokens,
            stop_at_total_tokens=self.stop_at_total_tokens,
            warmup_student_flops=self.warmup_student_flops,
            total_student_flops=self.total_student_flops,
            stop_at_student_flops=self.stop_at_student_flops,
            schedule_unit=self.schedule_unit,
            grad_accum_steps=self.grad_accum_steps,
            max_grad_norm=self.max_grad_norm,
            precision=self.precision,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_components=(
                self.gradient_checkpointing_components
            ),
            gradient_checkpointing_use_reentrant=(
                self.gradient_checkpointing_use_reentrant
            ),
            checkpoint_every_steps=self.checkpoint_every_steps,
            eval_every_steps=self.eval_every_steps,
            log_every_steps=self.log_every_steps,
            seed=self.seed,
            device=self.device,
            resume_from=self.resume_from,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            run_stage=f"sft:{self.target_mode}",
            loss_weights={
                "autoregressive": 1.0,
                "teacher_kl": 0.0,
                "hidden_feature_distillation": 0.0,
                "region_text_contrastive": 0.0,
                "box_regression": 0.0,
                "orientation": 0.0,
            },
        )


class StructuredPostTrainingDataset:
    """Convert unified benchmark samples into strict structured SFT/RL examples."""

    def __init__(self, samples: Sequence[Sample], target_mode: str = "evidence_linked"):
        if target_mode not in {"answer_only", "free_rationale", "evidence_linked"}:
            raise ValueError("unsupported post-training target mode")
        if not samples:
            raise ValueError("post-training dataset cannot be empty")
        self.samples = list(samples)
        self.target_mode = target_mode
        self.contexts = [RewardContext.from_sample(sample) for sample in self.samples]
        if any(not sample.answers for sample in self.samples):
            raise ValueError("every post-training sample requires at least one answer")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def tasks(self) -> list[str]:
        return [sample.answer_type for sample in self.samples]

    @property
    def sources(self) -> list[str]:
        return [
            str(sample.meta.get("source") or sample.meta.get("doc_type") or "unknown")
            for sample in self.samples
        ]

    @property
    def languages(self) -> list[str]:
        return [str(sample.meta.get("language") or "und") for sample in self.samples]

    def groups(self, key: str) -> list[str]:
        if key == "task":
            return self.tasks
        if key == "source":
            return self.sources
        if key == "language":
            return self.languages
        raise ValueError("group key must be task, source, or language")

    def prompt(self, index: int) -> str:
        return f"{self.samples[index].question.strip()}\n{STRUCTURED_RESPONSE_INSTRUCTION}"

    def target(self, index: int) -> str:
        sample = self.samples[index]
        context = self.contexts[index]
        evidence = (
            context.gold_boxes
            if self.target_mode == "evidence_linked"
            else ()
        )
        rationale = (
            context.gold_rationale
            if self.target_mode in {"free_rationale", "evidence_linked"}
            else ""
        )
        return build_structured_target(
            str(sample.answers[0]),
            evidence=evidence,
            rationale=rationale,
        )

    def __getitem__(self, index: int) -> StudentExample:
        sample = self.samples[index]
        return StudentExample(
            sample_id=sample.sample_id,
            source=self.sources[index],
            task=sample.answer_type,
            prompt=self.prompt(index),
            answer=self.target(index),
            image=sample.image_path or None,
            image_key=str(sample.meta.get("image_key") or sample.image_path),
            language=self.languages[index],
        )


def train_sft(
    student: DocumentVLMStudent,
    dataset: StructuredPostTrainingDataset,
    collator: StudentCollator,
    config: SFTConfig,
    *,
    metric_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Run answer-only, free-rationale, or evidence-linked structured SFT."""

    from torch.utils.data import DataLoader

    sampler = DeterministicDistributedBatchSampler(
        len(dataset),
        config.batch_size,
        seed=config.seed,
        num_replicas=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return train_student(
        student,
        loader,
        config.as_pretrain_config(),
        metric_callback=metric_callback,
    )


@dataclass(frozen=True)
class RLVRConfig:
    output_dir: str
    max_steps: int | None = 1000
    total_student_flops: int | None = None
    stop_at_student_flops: bool = False
    group_size: int = 8
    advantage_estimator: str = "group_standardized"
    max_new_tokens: int = 128
    max_new_tokens_hard_cap: int = 128
    max_new_tokens_by_answer_type: tuple[tuple[str, int], ...] = ()
    temperature: float = 0.8
    top_p: float = 0.95
    use_kv_cache: bool = True
    repetition_guard_min_tokens: int = 24
    repetition_guard_max_period: int = 16
    repetition_guard_repetitions: int = 3
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    kl_coefficient: float = 0.04
    advantage_epsilon: float = 1e-4
    supervised_replay_every_steps: int = 0
    supervised_replay_loss_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False
    gradient_checkpointing_components: tuple[str, ...] = (
        "vision",
        "connector",
        "language",
    )
    gradient_checkpointing_use_reentrant: bool = False
    checkpoint_every_steps: int = 100
    log_every_steps: int = 1
    seed: int = 23
    device: str = "auto"
    resume_from: str | None = None
    tokenizer_fingerprint: str | None = None
    reference_id: str = ""
    policy_start_id: str | None = None
    policy_start_stage: str = "sft"

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("RLVR max_steps must be positive when set")
        if self.max_steps is None and not self.stop_at_student_flops:
            raise ValueError(
                "RLVR max_steps can be null only with a student-FLOP stop"
            )
        if self.total_student_flops is not None and self.total_student_flops <= 0:
            raise ValueError("RLVR total_student_flops must be positive")
        if self.stop_at_student_flops and self.total_student_flops is None:
            raise ValueError(
                "RLVR student-FLOP stop requires total_student_flops"
            )
        if self.group_size < 2 or self.max_new_tokens <= 0:
            raise ValueError("RLVR steps/tokens must be positive and group_size at least two")
        validate_generation_token_budget_policy(
            base_tokens=self.max_new_tokens,
            hard_cap=self.max_new_tokens_hard_cap,
            by_answer_type=dict(self.max_new_tokens_by_answer_type),
        )
        if (
            not isinstance(self.advantage_estimator, str)
            or self.advantage_estimator
            not in {"group_standardized", "leave_one_out"}
        ):
            raise ValueError("unsupported RLVR advantage estimator")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("RLVR sampling controls are invalid")
        if not isinstance(self.use_kv_cache, bool):
            raise ValueError("RLVR use_kv_cache must be a boolean")
        if (
            self.repetition_guard_min_tokens < 1
            or self.repetition_guard_max_period < 1
            or self.repetition_guard_repetitions < 2
        ):
            raise ValueError("RLVR repetition guard controls are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("RLVR optimizer controls are invalid")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("RLVR optimizer betas must be within [0, 1)")
        if self.kl_coefficient < 0 or self.advantage_epsilon <= 0:
            raise ValueError("RLVR KL and advantage controls are invalid")
        if self.supervised_replay_every_steps < 0:
            raise ValueError("RLVR supervised replay interval must be non-negative")
        if self.supervised_replay_loss_coefficient < 0:
            raise ValueError("RLVR supervised replay coefficient must be non-negative")
        if (
            self.supervised_replay_every_steps == 0
        ) != (
            self.supervised_replay_loss_coefficient == 0
        ):
            raise ValueError(
                "RLVR supervised replay interval and coefficient must both be zero "
                "or both be positive"
            )
        if self.checkpoint_every_steps < 0 or self.log_every_steps <= 0:
            raise ValueError("RLVR checkpoint/log intervals are invalid")
        if self.precision not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("invalid RLVR precision")
        if (
            not self.gradient_checkpointing_components
            or len(set(self.gradient_checkpointing_components))
            != len(self.gradient_checkpointing_components)
            or not set(self.gradient_checkpointing_components)
            <= {"vision", "connector", "language"}
        ):
            raise ValueError(
                "RLVR gradient checkpointing components are invalid"
            )
        if not self.reference_id:
            raise ValueError("RLVR reference_id cannot be empty")
        if self.policy_start_id is not None and not self.policy_start_id:
            raise ValueError("RLVR policy_start_id cannot be empty")
        if not (
            self.policy_start_stage.startswith("sft:")
            or self.policy_start_stage
            in {"sft", "preference:dpo", "preference:ipo"}
        ):
            raise ValueError(
                "RLVR policy start must be an SFT or preference checkpoint"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        output_dir: str | Path,
        *,
        reference_id: str,
        policy_start_id: str | None = None,
        policy_start_stage: str = "sft",
        **overrides: Any,
    ) -> "RLVRConfig":
        raw = blueprint["training"]["posttraining"]["rlvr"]
        optimizer = raw["optimizer"]
        rollout = raw["rollout"]
        checkpointing = blueprint["training"]["activation_checkpointing"]
        supervised_replay = raw.get("supervised_replay") or {}
        values = {
            "output_dir": str(output_dir),
            "max_steps": (
                None
                if optimizer.get("max_steps") is None
                else int(optimizer["max_steps"])
            ),
            "total_student_flops": (
                None
                if optimizer.get("total_student_flops") is None
                else int(optimizer["total_student_flops"])
            ),
            "stop_at_student_flops": bool(
                optimizer.get("stop_at_student_flops", False)
            ),
            "group_size": int(raw["group_size"]),
            "advantage_estimator": str(
                raw.get("advantage_estimator", "group_standardized")
            ),
            "max_new_tokens": int(rollout["max_new_tokens"]),
            "max_new_tokens_hard_cap": int(
                rollout.get(
                    "max_new_tokens_hard_cap",
                    rollout["max_new_tokens"],
                )
            ),
            "max_new_tokens_by_answer_type": tuple(
                (str(pattern), int(budget))
                for pattern, budget in (
                    rollout.get("max_new_tokens_by_answer_type") or {}
                ).items()
            ),
            "temperature": float(rollout["temperature"]),
            "top_p": float(rollout["top_p"]),
            "use_kv_cache": bool(rollout["use_kv_cache"]),
            "repetition_guard_min_tokens": int(
                rollout.get("repetition_guard_min_tokens", 24)
            ),
            "repetition_guard_max_period": int(
                rollout.get("repetition_guard_max_period", 16)
            ),
            "repetition_guard_repetitions": int(
                rollout.get("repetition_guard_repetitions", 3)
            ),
            "learning_rate": float(optimizer["learning_rate"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "beta1": float(optimizer["betas"][0]),
            "beta2": float(optimizer["betas"][1]),
            "optimizer": OptimizerSpec.from_mapping(optimizer),
            "kl_coefficient": float(raw["kl_coefficient"]),
            "advantage_epsilon": float(raw["advantage_epsilon"]),
            "supervised_replay_every_steps": int(
                supervised_replay.get("every_steps", 0)
            ),
            "supervised_replay_loss_coefficient": float(
                supervised_replay.get("loss_coefficient", 0.0)
            ),
            "max_grad_norm": float(optimizer["max_grad_norm"]),
            "precision": str(optimizer["precision"]),
            "gradient_checkpointing": bool(
                checkpointing["enabled"]
            ),
            "gradient_checkpointing_components": tuple(
                str(value) for value in checkpointing["components"]
            ),
            "gradient_checkpointing_use_reentrant": bool(
                checkpointing["use_reentrant"]
            ),
            "checkpoint_every_steps": int(optimizer["checkpoint_every_steps"]),
            "log_every_steps": int(optimizer["log_every_steps"]),
            "seed": int(optimizer["seed"]),
            "reference_id": reference_id,
            "policy_start_id": policy_start_id,
            "policy_start_stage": policy_start_stage,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class PreferenceConfig:
    output_dir: str
    max_steps: int | None = 1000
    total_student_flops: int | None = None
    stop_at_student_flops: bool = False
    objective: str = "dpo"
    preference_source: str = "gold_anchored_verifier_ranked"
    group_size: int = 8
    minimum_reward_margin: float = 0.05
    dpo_beta: float = 0.1
    ipo_tau: float = 0.1
    sequence_reduction: str = "sum"
    max_new_tokens: int = 128
    max_new_tokens_hard_cap: int = 128
    max_new_tokens_by_answer_type: tuple[tuple[str, int], ...] = ()
    temperature: float = 0.8
    top_p: float = 0.95
    use_kv_cache: bool = True
    repetition_guard_min_tokens: int = 24
    repetition_guard_max_period: int = 16
    repetition_guard_repetitions: int = 3
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    max_grad_norm: float = 1.0
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False
    gradient_checkpointing_components: tuple[str, ...] = (
        "vision",
        "connector",
        "language",
    )
    gradient_checkpointing_use_reentrant: bool = False
    checkpoint_every_steps: int = 100
    log_every_steps: int = 1
    seed: int = 29
    device: str = "auto"
    resume_from: str | None = None
    tokenizer_fingerprint: str | None = None
    reference_id: str = ""

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("preference max_steps must be positive when set")
        if self.max_steps is None and not self.stop_at_student_flops:
            raise ValueError(
                "preference max_steps can be null only with a student-FLOP stop"
            )
        if self.total_student_flops is not None and self.total_student_flops <= 0:
            raise ValueError("preference total_student_flops must be positive")
        if self.stop_at_student_flops and self.total_student_flops is None:
            raise ValueError(
                "preference student-FLOP stop requires total_student_flops"
            )
        if self.objective not in {"dpo", "ipo"}:
            raise ValueError("preference objective must be dpo or ipo")
        if self.preference_source not in {
            "reference_verifier_ranked",
            "gold_anchored_verifier_ranked",
        }:
            raise ValueError("unsupported preference source")
        if self.group_size < 2 or self.max_new_tokens <= 0:
            raise ValueError(
                "preference candidate group must contain at least two completions"
            )
        validate_generation_token_budget_policy(
            base_tokens=self.max_new_tokens,
            hard_cap=self.max_new_tokens_hard_cap,
            by_answer_type=dict(self.max_new_tokens_by_answer_type),
        )
        if (
            self.minimum_reward_margin < 0
            or self.dpo_beta <= 0
            or self.ipo_tau <= 0
        ):
            raise ValueError("preference margin and regularization are invalid")
        if self.sequence_reduction not in {"sum", "mean"}:
            raise ValueError("preference sequence reduction must be sum or mean")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("preference sampling controls are invalid")
        if not isinstance(self.use_kv_cache, bool):
            raise ValueError("preference use_kv_cache must be a boolean")
        if (
            self.repetition_guard_min_tokens < 1
            or self.repetition_guard_max_period < 1
            or self.repetition_guard_repetitions < 2
        ):
            raise ValueError("preference repetition guard controls are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("preference optimizer controls are invalid")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("preference optimizer betas must be within [0, 1)")
        if self.checkpoint_every_steps < 0 or self.log_every_steps <= 0:
            raise ValueError("preference checkpoint/log intervals are invalid")
        if self.precision not in {
            "auto",
            "float32",
            "bfloat16",
            "float16",
        }:
            raise ValueError("invalid preference precision")
        if (
            not self.gradient_checkpointing_components
            or len(set(self.gradient_checkpointing_components))
            != len(self.gradient_checkpointing_components)
            or not set(self.gradient_checkpointing_components)
            <= {"vision", "connector", "language"}
        ):
            raise ValueError(
                "preference gradient checkpointing components are invalid"
            )
        if not self.reference_id:
            raise ValueError("preference reference_id cannot be empty")

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        output_dir: str | Path,
        *,
        reference_id: str,
        **overrides: Any,
    ) -> "PreferenceConfig":
        raw = blueprint["training"]["posttraining"]["preference"]
        optimizer = raw["optimizer"]
        rollout = raw["rollout"]
        checkpointing = blueprint["training"]["activation_checkpointing"]
        values = {
            "output_dir": str(output_dir),
            "max_steps": (
                None
                if optimizer.get("max_steps") is None
                else int(optimizer["max_steps"])
            ),
            "total_student_flops": (
                None
                if optimizer.get("total_student_flops") is None
                else int(optimizer["total_student_flops"])
            ),
            "stop_at_student_flops": bool(
                optimizer.get("stop_at_student_flops", False)
            ),
            "objective": str(raw["objective"]),
            "preference_source": str(raw["preference_source"]),
            "group_size": int(raw["group_size"]),
            "minimum_reward_margin": float(
                raw["minimum_reward_margin"]
            ),
            "dpo_beta": float(raw["dpo_beta"]),
            "ipo_tau": float(raw["ipo_tau"]),
            "sequence_reduction": str(raw["sequence_reduction"]),
            "max_new_tokens": int(rollout["max_new_tokens"]),
            "max_new_tokens_hard_cap": int(
                rollout.get(
                    "max_new_tokens_hard_cap",
                    rollout["max_new_tokens"],
                )
            ),
            "max_new_tokens_by_answer_type": tuple(
                (str(pattern), int(budget))
                for pattern, budget in (
                    rollout.get("max_new_tokens_by_answer_type") or {}
                ).items()
            ),
            "temperature": float(rollout["temperature"]),
            "top_p": float(rollout["top_p"]),
            "use_kv_cache": bool(rollout["use_kv_cache"]),
            "repetition_guard_min_tokens": int(
                rollout.get("repetition_guard_min_tokens", 24)
            ),
            "repetition_guard_max_period": int(
                rollout.get("repetition_guard_max_period", 16)
            ),
            "repetition_guard_repetitions": int(
                rollout.get("repetition_guard_repetitions", 3)
            ),
            "learning_rate": float(optimizer["learning_rate"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "beta1": float(optimizer["betas"][0]),
            "beta2": float(optimizer["betas"][1]),
            "optimizer": OptimizerSpec.from_mapping(optimizer),
            "max_grad_norm": float(optimizer["max_grad_norm"]),
            "precision": str(optimizer["precision"]),
            "gradient_checkpointing": bool(checkpointing["enabled"]),
            "gradient_checkpointing_components": tuple(
                str(value) for value in checkpointing["components"]
            ),
            "gradient_checkpointing_use_reentrant": bool(
                checkpointing["use_reentrant"]
            ),
            "checkpoint_every_steps": int(
                optimizer["checkpoint_every_steps"]
            ),
            "log_every_steps": int(optimizer["log_every_steps"]),
            "seed": int(optimizer["seed"]),
            "reference_id": reference_id,
        }
        values.update(overrides)
        return cls(**values)


@dataclass
class RLVRState:
    rollout_step: int = 0
    optimizer_step: int = 0
    policy_signal_steps: int = 0
    replay_only_steps: int = 0
    student_flops_seen: int = 0
    checkpoint_recompute_flops_seen: int = 0


@dataclass(frozen=True)
class RLVRResult:
    output_dir: str
    rollout_step: int
    optimizer_step: int
    policy_signal_steps: int
    replay_only_steps: int
    student_flops_seen: int
    checkpoint_recompute_flops_seen: int
    executed_student_flops_seen: int
    last_checkpoint: str
    final_metrics: dict[str, float]


@dataclass
class PreferenceState:
    preference_step: int = 0
    optimizer_step: int = 0
    accepted_pairs: int = 0
    skipped_pairs: int = 0
    student_flops_seen: int = 0
    checkpoint_recompute_flops_seen: int = 0


@dataclass(frozen=True)
class PreferenceResult:
    output_dir: str
    preference_step: int
    optimizer_step: int
    accepted_pairs: int
    skipped_pairs: int
    student_flops_seen: int
    checkpoint_recompute_flops_seen: int
    executed_student_flops_seen: int
    last_checkpoint: str
    final_metrics: dict[str, float]


def _device(name: str) -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "native preference/RL training currently requires one process; "
            "shard experiments by seed"
        )
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def posttraining_prompt_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move one collated example and retain only its prompt-side model inputs."""

    moved = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
    labels = moved["labels"][0]
    supervised = torch.nonzero(labels != -100, as_tuple=False)
    if supervised.numel() == 0:
        raise ValueError("post-training collator produced no target tokens")
    prompt_length = int(supervised[0].item())
    out = {
        "input_ids": moved["input_ids"][:, :prompt_length],
        "attention_mask": moved["attention_mask"][:, :prompt_length],
    }
    for key in (
        "pixel_values",
        "pixel_mask",
        "packed_pixel_values",
        "packed_position_ids",
        "packed_cu_seqlens",
    ):
        if key in moved:
            out[key] = moved[key]
    return out


def _repeat_batch(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
    return None if tensor is None else tensor.repeat_interleave(count, dim=0)


def _top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(sorted_probabilities, 1)
        return torch.gather(sorted_indices, -1, sampled_sorted)
    return torch.multinomial(probabilities, 1)


@torch.no_grad()
def sample_completion_group(
    model: DocumentVLMStudent,
    prompt_batch: dict[str, Any],
    tokenizer: Any,
    config: RLVRConfig | PreferenceConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Sample one group of completions for one document prompt."""

    group_size = config.group_size
    token_budget = config.max_new_tokens
    prompt_ids = _repeat_batch(prompt_batch["input_ids"], group_size)
    generated = prompt_ids
    completion_tokens: list[torch.Tensor] = []
    completion_masks: list[torch.Tensor] = []
    active = torch.ones(group_size, dtype=torch.bool, device=prompt_ids.device)
    eos = int(tokenizer.eos_token_id)
    was_training = model.training
    model.eval()
    visual_inputs = visual_model_inputs(prompt_batch)
    visual_prefix = (
        _repeat_batch(model.encode_images(**visual_inputs), group_size)
        if visual_inputs
        else None
    )
    next_logits = None
    generation_state = None
    try:
        if config.use_kv_cache:
            next_logits, generation_state = model.prefill_generation(
                generated,
                visual_prefix=visual_prefix,
                attention_mask=torch.ones_like(generated),
                max_new_tokens=token_budget,
            )
        for step in range(token_budget):
            if not config.use_kv_cache:
                output = model(
                    generated,
                    attention_mask=torch.ones_like(generated),
                    visual_prefix=visual_prefix,
                )
                next_logits = output.logits[:, -1].float()
            next_token = _top_p_sample(
                next_logits / config.temperature,
                config.top_p,
            ).squeeze(-1)
            history = (
                torch.stack(completion_tokens, dim=1)
                if completion_tokens
                else prompt_ids[:, :0]
            )
            repeated = repeated_suffix_cycle_mask(
                history,
                next_token,
                min_tokens=config.repetition_guard_min_tokens,
                max_period=config.repetition_guard_max_period,
                repetitions=config.repetition_guard_repetitions,
            )
            next_token = torch.where(
                active & repeated,
                torch.full_like(next_token, eos),
                next_token,
            )
            next_token = torch.where(
                active,
                next_token,
                torch.full_like(next_token, eos),
            )
            token_mask = active.clone()
            completion_tokens.append(next_token)
            completion_masks.append(token_mask)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            active = active & (next_token != eos)
            if not torch.any(active):
                break
            if (
                config.use_kv_cache
                and step + 1 < token_budget
            ):
                next_logits, generation_state = model.decode_generation(
                    next_token[:, None],
                    generation_state,
                )
    finally:
        model.train(was_training)
    token_tensor = torch.stack(completion_tokens, dim=1)
    mask_tensor = torch.stack(completion_masks, dim=1)
    texts = [
        tokenizer.decode(
            token_tensor[index][mask_tensor[index]].tolist(),
            skip_special_tokens=True,
        ).strip()
        for index in range(group_size)
    ]
    return token_tensor, mask_tensor, texts


def inject_gold_preference_candidate(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    texts: Sequence[str],
    batch: dict[str, Any],
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Replace one sampled candidate with the exact collated SFT target."""

    if (
        completion_ids.ndim != 2
        or completion_mask.shape != completion_ids.shape
        or len(texts) != completion_ids.shape[0]
    ):
        raise ValueError("preference candidates have inconsistent shapes")
    labels = batch.get("labels")
    input_ids = batch.get("input_ids")
    if (
        not isinstance(labels, torch.Tensor)
        or not isinstance(input_ids, torch.Tensor)
        or labels.shape != input_ids.shape
        or labels.shape[0] != 1
    ):
        raise ValueError("gold-anchored preference requires one collated target")
    supervised = labels[0] != -100
    gold_ids = input_ids[0][supervised].to(completion_ids.device)
    if gold_ids.numel() == 0:
        raise ValueError("gold-anchored preference target is empty")

    width = max(int(completion_ids.shape[1]), int(gold_ids.numel()))
    anchored_ids = torch.full(
        (completion_ids.shape[0], width),
        int(tokenizer.pad_token_id),
        dtype=completion_ids.dtype,
        device=completion_ids.device,
    )
    anchored_mask = torch.zeros(
        (completion_ids.shape[0], width),
        dtype=torch.bool,
        device=completion_ids.device,
    )
    sampled_width = completion_ids.shape[1]
    anchored_ids[:, :sampled_width] = completion_ids
    anchored_mask[:, :sampled_width] = completion_mask.bool()
    anchored_ids[0].fill_(int(tokenizer.pad_token_id))
    anchored_mask[0].fill_(False)
    anchored_ids[0, : gold_ids.numel()] = gold_ids
    anchored_mask[0, : gold_ids.numel()] = True
    anchored_texts = list(texts)
    anchored_texts[0] = tokenizer.decode(
        gold_ids.tolist(),
        skip_special_tokens=True,
    ).strip()
    return anchored_ids, anchored_mask, anchored_texts


def completion_log_probs(
    model: DocumentVLMStudent,
    prompt_batch: dict[str, Any],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Return aligned next-token log probabilities for sampled completion tokens."""

    group_size, completion_length = completion_ids.shape
    prompt_ids = _repeat_batch(prompt_batch["input_ids"], group_size)
    prompt_mask = _repeat_batch(prompt_batch["attention_mask"], group_size)
    visual_inputs = visual_model_inputs(prompt_batch)
    visual_prefix = (
        _repeat_batch(model.encode_images(**visual_inputs), group_size)
        if visual_inputs
        else None
    )
    sequence = torch.cat((prompt_ids, completion_ids), dim=1)
    attention_mask = torch.cat(
        (prompt_mask, completion_mask.to(prompt_mask.dtype)),
        dim=1,
    )
    output = model(
        sequence,
        attention_mask=attention_mask,
        visual_prefix=visual_prefix,
    )
    text_logits = output.logits[:, -sequence.shape[1] :]
    start = prompt_ids.shape[1] - 1
    token_logits = text_logits[:, start : start + completion_length]
    return torch.gather(
        torch.log_softmax(token_logits.float(), dim=-1),
        -1,
        completion_ids.unsqueeze(-1),
    ).squeeze(-1)


def group_relative_policy_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    rewards: torch.Tensor,
    *,
    kl_coefficient: float,
    advantage_epsilon: float,
    advantage_estimator: str = "group_standardized",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single-update group policy objective with a frozen-reference KL."""

    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-probability shapes must match")
    if completion_mask.shape != policy_log_probs.shape:
        raise ValueError("completion mask must align with token log probabilities")
    if rewards.shape != (policy_log_probs.shape[0],):
        raise ValueError("one reward is required per completion")
    mask = completion_mask.to(policy_log_probs.dtype)
    token_counts = mask.sum(dim=1).clamp_min(1)
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if advantage_estimator == "group_standardized":
        advantages = (
            (rewards - reward_mean)
            / reward_std.clamp_min(advantage_epsilon)
            if reward_std >= advantage_epsilon
            else torch.zeros_like(rewards)
        )
    elif advantage_estimator == "leave_one_out":
        group_size = rewards.numel()
        if group_size < 2:
            raise ValueError(
                "leave-one-out advantages require at least two rewards"
            )
        leave_one_out_baseline = (
            rewards.sum() - rewards
        ) / (group_size - 1)
        advantages = rewards - leave_one_out_baseline
    else:
        raise ValueError(
            f"unsupported advantage estimator {advantage_estimator!r}"
        )
    sequence_log_probs = (policy_log_probs * mask).sum(dim=1) / token_counts
    policy_loss = -(advantages.detach() * sequence_log_probs).mean()
    log_ratio = reference_log_probs - policy_log_probs
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    reference_kl = (per_token_kl * mask).sum() / mask.sum().clamp_min(1)
    total = policy_loss + kl_coefficient * reference_kl
    return total, {
        "policy_loss": policy_loss,
        "reference_kl": reference_kl,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "advantage_abs_mean": advantages.abs().mean(),
        "advantage_std": advantages.std(unbiased=False),
    }


def select_preference_pair(
    rewards: torch.Tensor,
    *,
    minimum_reward_margin: float,
) -> tuple[int, int, float] | None:
    """Select deterministic best/worst candidates when verifier margin is sufficient."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("preference selection requires at least two rewards")
    chosen = int(torch.argmax(rewards).item())
    rejected = int(torch.argmin(rewards).item())
    margin = float((rewards[chosen] - rewards[rejected]).item())
    if chosen == rejected or margin < minimum_reward_margin:
        return None
    return chosen, rejected, margin


def preference_optimization_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    objective: str,
    dpo_beta: float,
    ipo_tau: float,
    sequence_reduction: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DPO or IPO loss for one chosen/rejected pair."""

    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log-probability shapes must match")
    if completion_mask.shape != policy_log_probs.shape:
        raise ValueError("completion mask must align with token log probabilities")
    if policy_log_probs.shape[0] != 2:
        raise ValueError(
            "preference optimization requires chosen and rejected sequences"
        )
    if objective not in {"dpo", "ipo"}:
        raise ValueError("preference objective must be dpo or ipo")
    if dpo_beta <= 0 or ipo_tau <= 0:
        raise ValueError("preference regularization must be positive")
    mask = completion_mask.to(policy_log_probs.dtype)
    token_counts = mask.sum(dim=1).clamp_min(1)
    policy_sequences = (policy_log_probs * mask).sum(dim=1)
    reference_sequences = (reference_log_probs * mask).sum(dim=1)
    if sequence_reduction == "mean":
        policy_sequences = policy_sequences / token_counts
        reference_sequences = reference_sequences / token_counts
    elif sequence_reduction != "sum":
        raise ValueError("preference sequence reduction must be sum or mean")
    policy_log_ratio = policy_sequences[0] - policy_sequences[1]
    reference_log_ratio = reference_sequences[0] - reference_sequences[1]
    log_ratio_margin = policy_log_ratio - reference_log_ratio
    if objective == "dpo":
        preference_logit = dpo_beta * log_ratio_margin
        target_log_ratio_margin = torch.zeros_like(log_ratio_margin)
        loss = -F.logsigmoid(preference_logit)
    else:
        target_log_ratio_margin = torch.full_like(
            log_ratio_margin,
            1.0 / (2.0 * ipo_tau),
        )
        preference_logit = log_ratio_margin
        loss = (log_ratio_margin - target_log_ratio_margin).square()
    return loss, {
        "loss": loss,
        "preference_logit": preference_logit,
        "log_ratio_margin": log_ratio_margin,
        "target_log_ratio_margin": target_log_ratio_margin,
        "policy_log_ratio": policy_log_ratio,
        "reference_log_ratio": reference_log_ratio,
        "scaled_log_ratio_margin": (
            dpo_beta if objective == "dpo" else ipo_tau
        )
        * log_ratio_margin,
        "preference_accuracy": (log_ratio_margin > 0).to(
            policy_log_probs.dtype
        ),
        "chosen_sequence_log_prob": policy_sequences[0],
        "rejected_sequence_log_prob": policy_sequences[1],
    }


def direct_preference_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    beta: float,
    sequence_reduction: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backward-compatible DPO objective wrapper."""

    return preference_optimization_loss(
        policy_log_probs,
        reference_log_probs,
        completion_mask,
        objective="dpo",
        dpo_beta=beta,
        ipo_tau=0.1,
        sequence_reduction=sequence_reduction,
    )


def supervised_replay_loss(
    policy: DocumentVLMStudent,
    dataset: StructuredPostTrainingDataset,
    collator: StudentCollator,
    sample_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Compute one answer-target cross-entropy anchor without auxiliary heads."""

    raw_batch = collator([dataset[sample_index]])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
    }
    inputs = student_model_inputs(batch)
    inputs = {
        key: value
        for key, value in inputs.items()
        if key
        in {
            "input_ids",
            "pixel_values",
            "pixel_mask",
            "packed_pixel_values",
            "packed_position_ids",
            "packed_cu_seqlens",
            "attention_mask",
            "labels",
        }
    }
    output = policy(**inputs)
    loss = output.losses.get("autoregressive")
    if loss is None:
        raise RuntimeError("supervised replay batch produced no autoregressive loss")
    supervised_tokens = int((batch["labels"] != policy.config.ignore_index).sum().item())
    if supervised_tokens <= 0:
        raise RuntimeError("supervised replay batch contains no answer tokens")
    return loss, supervised_tokens, int(batch["input_ids"].shape[1])


def _supervised_replay_contract(config: RLVRConfig) -> dict[str, float | int]:
    return {
        "every_steps": config.supervised_replay_every_steps,
        "loss_coefficient": config.supervised_replay_loss_coefficient,
    }


def _rlvr_budget_contract(config: RLVRConfig) -> dict[str, int | bool | None]:
    return {
        "total_student_flops": config.total_student_flops,
        "stop_at_student_flops": config.stop_at_student_flops,
    }


def _rlvr_rollout_contract(
    config: RLVRConfig,
) -> dict[str, Any]:
    return {
        "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens,
        "max_new_tokens_hard_cap": config.max_new_tokens_hard_cap,
        "max_new_tokens_by_answer_type": dict(
            config.max_new_tokens_by_answer_type
        ),
        "temperature": config.temperature,
        "top_p": config.top_p,
        "use_kv_cache": config.use_kv_cache,
        "repetition_guard_min_tokens": config.repetition_guard_min_tokens,
        "repetition_guard_max_period": config.repetition_guard_max_period,
        "repetition_guard_repetitions": config.repetition_guard_repetitions,
    }


def _rlvr_objective_contract(
    config: RLVRConfig,
    reward_config: RewardConfig,
) -> dict[str, Any]:
    return {
        "advantage_estimator": config.advantage_estimator,
        "advantage_epsilon": config.advantage_epsilon,
        "kl_coefficient": config.kl_coefficient,
        "reward_weights": {
            name: float(weight)
            for name, weight in sorted(reward_config.weights.items())
        },
        "malformed_reward": reward_config.malformed_reward,
        "malformed_recovery_max": reward_config.malformed_recovery_max,
        "rationale_verifier": reward_config.rationale_verifier,
    }


def _rlvr_policy_start_contract(config: RLVRConfig) -> dict[str, str]:
    return {
        "content_id": config.policy_start_id or config.reference_id,
        "run_stage": config.policy_start_stage,
    }


def _save_rlvr_checkpoint(
    model: DocumentVLMStudent,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: RLVRState,
    config: RLVRConfig,
    reward_config: RewardConfig,
) -> Path:
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f"step-{state.rollout_step:08d}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite RLVR checkpoint {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints))
    model.save_pretrained(
        temporary / "student",
        metadata={
            "run_stage": "rlvr",
            "trainer_state": asdict(state),
            "tokenizer_fingerprint": config.tokenizer_fingerprint,
            "reference_id": config.reference_id,
            "policy_start": _rlvr_policy_start_contract(config),
            "gradient_checkpointing": (
                model.gradient_checkpointing_state
            ),
            "supervised_replay": _supervised_replay_contract(config),
            "rollout": _rlvr_rollout_contract(config),
            "objective": _rlvr_objective_contract(config, reward_config),
            "compute_budget": _rlvr_budget_contract(config),
            "optimizer": optimizer_runtime_contract(
                optimizer,
                config.optimizer,
            ),
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        temporary / "training_state.pt",
    )
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    pointer = output / ".latest_checkpoint.tmp"
    pointer.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    os.replace(pointer, output / "latest_checkpoint.txt")
    return target


def _resolve_rlvr_resume(config: RLVRConfig) -> Path | None:
    if config.resume_from is None:
        return None
    if config.resume_from != "latest":
        return Path(config.resume_from)
    pointer = Path(config.output_dir) / "latest_checkpoint.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"no RLVR checkpoint pointer at {pointer}")
    return Path(pointer.read_text(encoding="utf-8").strip())


def _load_rlvr_checkpoint(
    path: Path,
    model: DocumentVLMStudent,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: RLVRConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> RLVRState:
    metadata = json.loads(
        (path / "student" / "metadata.json").read_text(encoding="utf-8")
    )
    validate_checkpoint_initialization_lineage(model, metadata)
    if metadata.get("run_stage") != "rlvr":
        raise ValueError("resume checkpoint is not an RLVR checkpoint")
    if metadata.get("tokenizer_fingerprint") != config.tokenizer_fingerprint:
        raise ValueError("RLVR tokenizer fingerprint mismatch")
    if metadata.get("reference_id") != config.reference_id:
        raise ValueError("RLVR frozen reference mismatch")
    if metadata.get("policy_start") != _rlvr_policy_start_contract(config):
        raise ValueError("RLVR policy-start checkpoint mismatch")
    if (
        metadata.get("gradient_checkpointing")
        != model.gradient_checkpointing_state
    ):
        raise ValueError(
            "RLVR gradient-checkpointing contract mismatch"
        )
    if metadata.get("supervised_replay") != _supervised_replay_contract(config):
        raise ValueError("RLVR supervised replay contract mismatch")
    if metadata.get("rollout") != _rlvr_rollout_contract(config):
        raise ValueError("RLVR rollout contract mismatch")
    if metadata.get("objective") != _rlvr_objective_contract(
        config,
        reward_config,
    ):
        raise ValueError("RLVR objective contract mismatch")
    if metadata.get("compute_budget") != _rlvr_budget_contract(config):
        raise ValueError("RLVR compute-budget contract mismatch")
    if metadata.get("optimizer") != optimizer_runtime_contract(
        optimizer,
        config.optimizer,
    ):
        raise ValueError("RLVR optimizer contract mismatch")
    model.load_state_dict(
        torch.load(
            path / "student" / "model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    payload = torch.load(
        path / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return RLVRState(
        **json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    )


def _preference_rollout_contract(
    config: PreferenceConfig,
) -> dict[str, Any]:
    return {
        "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens,
        "max_new_tokens_hard_cap": config.max_new_tokens_hard_cap,
        "max_new_tokens_by_answer_type": dict(
            config.max_new_tokens_by_answer_type
        ),
        "temperature": config.temperature,
        "top_p": config.top_p,
        "use_kv_cache": config.use_kv_cache,
        "repetition_guard_min_tokens": config.repetition_guard_min_tokens,
        "repetition_guard_max_period": config.repetition_guard_max_period,
        "repetition_guard_repetitions": config.repetition_guard_repetitions,
    }


def _preference_objective_contract(
    config: PreferenceConfig,
    reward_config: RewardConfig,
) -> dict[str, Any]:
    return {
        "objective": config.objective,
        "preference_source": config.preference_source,
        "minimum_reward_margin": config.minimum_reward_margin,
        "dpo_beta": config.dpo_beta,
        "ipo_tau": config.ipo_tau,
        "sequence_reduction": config.sequence_reduction,
        "reward_weights": {
            name: float(weight)
            for name, weight in sorted(reward_config.weights.items())
        },
        "malformed_reward": reward_config.malformed_reward,
        "malformed_recovery_max": reward_config.malformed_recovery_max,
        "rationale_verifier": reward_config.rationale_verifier,
    }


def _preference_budget_contract(
    config: PreferenceConfig,
) -> dict[str, int | bool | None]:
    return {
        "total_student_flops": config.total_student_flops,
        "stop_at_student_flops": config.stop_at_student_flops,
    }


def _save_preference_checkpoint(
    model: DocumentVLMStudent,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: PreferenceState,
    config: PreferenceConfig,
    reward_config: RewardConfig,
) -> Path:
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f"step-{state.preference_step:08d}"
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite preference checkpoint {target}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints))
    model.save_pretrained(
        temporary / "student",
        metadata={
            "run_stage": f"preference:{config.objective}",
            "trainer_state": asdict(state),
            "tokenizer_fingerprint": config.tokenizer_fingerprint,
            "reference_id": config.reference_id,
            "gradient_checkpointing": (
                model.gradient_checkpointing_state
            ),
            "rollout": _preference_rollout_contract(config),
            "objective": _preference_objective_contract(config, reward_config),
            "compute_budget": _preference_budget_contract(config),
            "optimizer": optimizer_runtime_contract(
                optimizer,
                config.optimizer,
            ),
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        temporary / "training_state.pt",
    )
    (temporary / "trainer_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    pointer = output / ".latest_checkpoint.tmp"
    pointer.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    os.replace(pointer, output / "latest_checkpoint.txt")
    return target


def _resolve_preference_resume(config: PreferenceConfig) -> Path | None:
    if config.resume_from is None:
        return None
    if config.resume_from != "latest":
        return Path(config.resume_from)
    pointer = Path(config.output_dir) / "latest_checkpoint.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"no preference checkpoint pointer at {pointer}"
        )
    return Path(pointer.read_text(encoding="utf-8").strip())


def _load_preference_checkpoint(
    path: Path,
    model: DocumentVLMStudent,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: PreferenceConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> PreferenceState:
    metadata = json.loads(
        (path / "student" / "metadata.json").read_text(encoding="utf-8")
    )
    validate_checkpoint_initialization_lineage(model, metadata)
    if metadata.get("run_stage") != f"preference:{config.objective}":
        raise ValueError("resume checkpoint is not the requested preference objective")
    if metadata.get("tokenizer_fingerprint") != config.tokenizer_fingerprint:
        raise ValueError("preference tokenizer fingerprint mismatch")
    if metadata.get("reference_id") != config.reference_id:
        raise ValueError("preference frozen reference mismatch")
    if (
        metadata.get("gradient_checkpointing")
        != model.gradient_checkpointing_state
    ):
        raise ValueError("preference gradient-checkpointing contract mismatch")
    if metadata.get("rollout") != _preference_rollout_contract(config):
        raise ValueError("preference rollout contract mismatch")
    if metadata.get("objective") != _preference_objective_contract(
        config,
        reward_config,
    ):
        raise ValueError("preference objective contract mismatch")
    if metadata.get("compute_budget") != _preference_budget_contract(config):
        raise ValueError("preference compute-budget contract mismatch")
    if metadata.get("optimizer") != optimizer_runtime_contract(
        optimizer,
        config.optimizer,
    ):
        raise ValueError("preference optimizer contract mismatch")
    model.load_state_dict(
        torch.load(
            path / "student" / "model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    payload = torch.load(
        path / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return PreferenceState(
        **json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    )


def _prompt_vision_tokens(
    prompt_batch: dict[str, Any],
    policy: DocumentVLMStudent,
) -> int:
    packed_cu_seqlens = prompt_batch.get("packed_cu_seqlens")
    pixel_values = prompt_batch.get("pixel_values")
    if packed_cu_seqlens is not None:
        return int(
            packed_cu_seqlens[-1].item()
            - packed_cu_seqlens[-2].item()
        )
    if pixel_values is None:
        return 0
    patch_size = policy.config.vision.patch_size
    height, width = pixel_values.shape[-2:]
    return (
        (int(height) + patch_size - 1)
        // patch_size
        * ((int(width) + patch_size - 1) // patch_size)
    )


def train_preference(
    policy: DocumentVLMStudent,
    reference: DocumentVLMStudent,
    dataset: StructuredPostTrainingDataset,
    collator: StudentCollator,
    tokenizer: Any,
    config: PreferenceConfig,
    reward_config: RewardConfig,
    *,
    metric_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PreferenceResult:
    """Run resumable verifier-ranked preference optimization."""

    device = _device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.configure_gradient_checkpointing(
        enabled=config.gradient_checkpointing,
        components=config.gradient_checkpointing_components,
        use_reentrant=(
            config.gradient_checkpointing_use_reentrant
        ),
    )
    reference.configure_gradient_checkpointing(
        enabled=False,
        components=config.gradient_checkpointing_components,
        use_reentrant=(
            config.gradient_checkpointing_use_reentrant
        ),
    )
    policy.to(device).train()
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = build_optimizer(
        _parameter_groups(policy, config.weight_decay),
        config.optimizer,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=_uses_fp16(device, config.precision),
    )
    state = PreferenceState()
    resume_path = _resolve_preference_resume(config)
    if resume_path is not None:
        state = _load_preference_checkpoint(
            resume_path,
            policy,
            optimizer,
            scaler,
            config,
            reward_config,
            device,
        )
    last_checkpoint = resume_path
    final_metrics: dict[str, float] = {}

    def budget_remaining() -> bool:
        within_steps = (
            config.max_steps is None
            or state.preference_step < config.max_steps
        )
        within_compute = (
            not config.stop_at_student_flops
            or config.total_student_flops is None
            or state.student_flops_seen < config.total_student_flops
        )
        return within_steps and within_compute

    while budget_remaining():
        sample_index = random.randrange(len(dataset))
        generation_token_budget, generation_budget_source = (
            resolve_generation_token_budget(
                dataset.samples[sample_index].answer_type,
                base_tokens=config.max_new_tokens,
                hard_cap=config.max_new_tokens_hard_cap,
                by_answer_type=config.max_new_tokens_by_answer_type,
            )
        )
        raw_batch = collator([dataset[sample_index]])
        prompt_batch = posttraining_prompt_batch(raw_batch, device)
        with _autocast_context(device, config.precision):
            completion_ids, completion_mask, texts = sample_completion_group(
                reference,
                prompt_batch,
                tokenizer,
                replace(
                    config,
                    max_new_tokens=generation_token_budget,
                    max_new_tokens_by_answer_type=(),
                ),
            )
        gold_anchor_applied = (
            config.preference_source == "gold_anchored_verifier_ranked"
        )
        if gold_anchor_applied:
            completion_ids, completion_mask, texts = (
                inject_gold_preference_candidate(
                    completion_ids,
                    completion_mask,
                    texts,
                    raw_batch,
                    tokenizer,
                )
            )
        reward_results = [
            score_structured_response(
                text,
                dataset.contexts[sample_index],
                reward_config,
            )
            for text in texts
        ]
        if (
            gold_anchor_applied
            and not reward_results[0].structurally_valid
        ):
            raise ValueError(
                "collated gold preference anchor is not a valid structured response"
            )
        rewards = torch.tensor(
            [result.total for result in reward_results],
            dtype=torch.float32,
            device=device,
        )
        pair = select_preference_pair(
            rewards,
            minimum_reward_margin=config.minimum_reward_margin,
        )
        accepted = pair is not None
        verifier_margin = pair[2] if pair is not None else 0.0
        tensors: dict[str, torch.Tensor] = {}
        gradient_norm = torch.zeros((), device=device)
        if pair is not None:
            chosen, rejected, _ = pair
            indices = torch.tensor(
                [chosen, rejected],
                dtype=torch.long,
                device=device,
            )
            pair_ids = completion_ids.index_select(0, indices)
            pair_mask = completion_mask.index_select(0, indices)
            with _autocast_context(device, config.precision):
                policy_log_probs = completion_log_probs(
                    policy,
                    prompt_batch,
                    pair_ids,
                    pair_mask,
                )
                with torch.no_grad():
                    reference_log_probs = completion_log_probs(
                        reference,
                        prompt_batch,
                        pair_ids,
                        pair_mask,
                    )
                loss, tensors = preference_optimization_loss(
                    policy_log_probs,
                    reference_log_probs,
                    pair_mask,
                    objective=config.objective,
                    dpo_beta=config.dpo_beta,
                    ipo_tau=config.ipo_tau,
                    sequence_reduction=config.sequence_reduction,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            state.optimizer_step += 1
            state.accepted_pairs += 1
        else:
            state.skipped_pairs += 1
        vision_tokens = _prompt_vision_tokens(prompt_batch, policy)
        step_flops = estimate_preference_step_flops(
            policy.config,
            vision_tokens=vision_tokens,
            prompt_tokens=int(prompt_batch["input_ids"].shape[1]),
            completion_tokens=int(completion_ids.shape[1]),
            candidate_group_size=config.group_size,
            use_kv_cache=config.use_kv_cache,
            checkpoint_components=(
                config.gradient_checkpointing_components
                if config.gradient_checkpointing
                else ()
            ),
            accepted_pair=accepted,
        )
        state.student_flops_seen += step_flops["total"]
        state.checkpoint_recompute_flops_seen += step_flops[
            "checkpoint_recompute"
        ]
        state.preference_step += 1
        valid_fraction = sum(
            result.structurally_valid for result in reward_results
        ) / len(reward_results)
        sampled_results = (
            reward_results[1:] if gold_anchor_applied else reward_results
        )
        sampled_rewards = torch.tensor(
            [result.total for result in sampled_results],
            dtype=torch.float32,
            device=device,
        )
        sampled_valid_fraction = sum(
            result.structurally_valid for result in sampled_results
        ) / len(sampled_results)
        final_metrics = {
            f"preference/{name}": float(value.detach())
            for name, value in tensors.items()
        }
        for name in (
            "loss",
            "preference_logit",
            "log_ratio_margin",
            "target_log_ratio_margin",
            "policy_log_ratio",
            "reference_log_ratio",
            "scaled_log_ratio_margin",
            "preference_accuracy",
            "chosen_sequence_log_prob",
            "rejected_sequence_log_prob",
        ):
            final_metrics.setdefault(f"preference/{name}", 0.0)
        final_metrics.update(
            {
                "preference/accepted_pair": float(accepted),
                "preference/gold_anchor_applied": float(
                    gold_anchor_applied
                ),
                "preference/gold_anchor_reward": (
                    float(rewards[0]) if gold_anchor_applied else 0.0
                ),
                "preference/verifier_reward_margin": verifier_margin,
                "preference/reward_mean": float(rewards.mean()),
                "preference/reward_std": float(
                    rewards.std(unbiased=False)
                ),
                "preference/sampled_reward_mean": float(
                    sampled_rewards.mean()
                ),
                "preference/sampled_reward_std": float(
                    sampled_rewards.std(unbiased=False)
                ),
                "preference/valid_structure_fraction": valid_fraction,
                "preference/sampled_valid_structure_fraction": (
                    sampled_valid_fraction
                ),
                "preference/gradient_norm": float(gradient_norm),
                "preference/generation_token_budget": float(
                    generation_token_budget
                ),
                "preference/generation_budget_escalated": float(
                    generation_budget_source != "default"
                ),
                "preference/preference_step": float(state.preference_step),
                "preference/optimizer_step": float(state.optimizer_step),
                "preference/accepted_pairs": float(state.accepted_pairs),
                "preference/skipped_pairs": float(state.skipped_pairs),
                "preference/student_flops_seen": float(
                    state.student_flops_seen
                ),
                "preference/step_student_flops": float(step_flops["total"]),
                "preference/checkpoint_recompute_flops_seen": float(
                    state.checkpoint_recompute_flops_seen
                ),
                "preference/executed_student_flops_seen": float(
                    state.student_flops_seen
                    + state.checkpoint_recompute_flops_seen
                ),
                "preference/step_checkpoint_recompute_flops": float(
                    step_flops["checkpoint_recompute"]
                ),
                "preference/step_executed_student_flops": float(
                    step_flops["executed_total"]
                ),
            }
        )
        for name in reward_config.weights:
            values = [
                result.components[name]
                for result in reward_results
                if name in result.applicable
            ]
            if values:
                final_metrics[f"reward/{name}"] = sum(values) / len(values)
        for diagnostic in (
            "rationale_text_similarity",
            "rationale_program_fact_score",
            "program_trace_consistency",
        ):
            values = [
                result.components[diagnostic]
                for result in reward_results
                if diagnostic in result.applicable
            ]
            if values:
                final_metrics[f"reward_diagnostic/{diagnostic}"] = (
                    sum(values) / len(values)
                )
        if (
            state.preference_step == 1
            or state.preference_step % config.log_every_steps == 0
        ):
            _record_metric(
                output_dir,
                {
                    "kind": "preference",
                    "sample_id": dataset.samples[sample_index].sample_id,
                    "objective": config.objective,
                    "preference_source": config.preference_source,
                    "sequence_reduction": config.sequence_reduction,
                    "generation_token_budget_source": (
                        generation_budget_source
                    ),
                    **final_metrics,
                },
                metric_callback,
            )
            print(
                f"[preference] step={state.preference_step} "
                f"objective={config.objective} "
                f"optimizer_step={state.optimizer_step} "
                f"student_flops={state.student_flops_seen:,} "
                f"reward_margin={verifier_margin:.4f} "
                f"accepted={int(accepted)}",
                flush=True,
            )
        if (
            config.checkpoint_every_steps > 0
            and state.preference_step % config.checkpoint_every_steps == 0
        ):
            last_checkpoint = _save_preference_checkpoint(
                policy,
                optimizer,
                scaler,
                state,
                config,
                reward_config,
            )
    if (
        last_checkpoint is None
        or last_checkpoint.name != f"step-{state.preference_step:08d}"
    ):
        last_checkpoint = _save_preference_checkpoint(
            policy,
            optimizer,
            scaler,
            state,
            config,
            reward_config,
        )
    return PreferenceResult(
        output_dir=str(output_dir),
        preference_step=state.preference_step,
        optimizer_step=state.optimizer_step,
        accepted_pairs=state.accepted_pairs,
        skipped_pairs=state.skipped_pairs,
        student_flops_seen=state.student_flops_seen,
        checkpoint_recompute_flops_seen=(
            state.checkpoint_recompute_flops_seen
        ),
        executed_student_flops_seen=(
            state.student_flops_seen
            + state.checkpoint_recompute_flops_seen
        ),
        last_checkpoint=str(last_checkpoint),
        final_metrics=final_metrics,
    )


def train_grpo(
    policy: DocumentVLMStudent,
    reference: DocumentVLMStudent,
    dataset: StructuredPostTrainingDataset,
    collator: StudentCollator,
    tokenizer: Any,
    config: RLVRConfig,
    reward_config: RewardConfig,
    *,
    replay_dataset: StructuredPostTrainingDataset | None = None,
    metric_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RLVRResult:
    """Run resumable group-relative RL with an optional supervised replay anchor."""

    device = _device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.configure_gradient_checkpointing(
        enabled=config.gradient_checkpointing,
        components=config.gradient_checkpointing_components,
        use_reentrant=(
            config.gradient_checkpointing_use_reentrant
        ),
    )
    reference.configure_gradient_checkpointing(
        enabled=False,
        components=config.gradient_checkpointing_components,
        use_reentrant=(
            config.gradient_checkpointing_use_reentrant
        ),
    )
    policy.to(device).train()
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = build_optimizer(
        _parameter_groups(policy, config.weight_decay),
        config.optimizer,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=_uses_fp16(device, config.precision),
    )
    state = RLVRState()
    resume_path = _resolve_rlvr_resume(config)
    if resume_path is not None:
        state = _load_rlvr_checkpoint(
            resume_path,
            policy,
            optimizer,
            scaler,
            config,
            reward_config,
            device,
        )
    last_checkpoint = resume_path
    final_metrics: dict[str, float] = {}
    active_replay_dataset = replay_dataset or dataset

    def budget_remaining() -> bool:
        within_steps = (
            config.max_steps is None
            or state.rollout_step < config.max_steps
        )
        within_compute = (
            not config.stop_at_student_flops
            or config.total_student_flops is None
            or state.student_flops_seen < config.total_student_flops
        )
        return within_steps and within_compute

    while budget_remaining():
        sample_index = random.randrange(len(dataset))
        generation_token_budget, generation_budget_source = (
            resolve_generation_token_budget(
                dataset.samples[sample_index].answer_type,
                base_tokens=config.max_new_tokens,
                hard_cap=config.max_new_tokens_hard_cap,
                by_answer_type=config.max_new_tokens_by_answer_type,
            )
        )
        raw_batch = collator([dataset[sample_index]])
        prompt_batch = posttraining_prompt_batch(raw_batch, device)
        with _autocast_context(device, config.precision):
            completion_ids, completion_mask, texts = sample_completion_group(
                policy,
                prompt_batch,
                tokenizer,
                replace(
                    config,
                    max_new_tokens=generation_token_budget,
                    max_new_tokens_by_answer_type=(),
                ),
            )
        reward_results: list[RewardResult] = [
            score_structured_response(
                text,
                dataset.contexts[sample_index],
                reward_config,
            )
            for text in texts
        ]
        rewards = torch.tensor(
            [result.total for result in reward_results],
            dtype=torch.float32,
            device=device,
        )
        with _autocast_context(device, config.precision):
            policy_log_probs = completion_log_probs(
                policy,
                prompt_batch,
                completion_ids,
                completion_mask,
            )
            with torch.no_grad():
                reference_log_probs = completion_log_probs(
                    reference,
                    prompt_batch,
                    completion_ids,
                    completion_mask,
                )
            loss, tensors = group_relative_policy_loss(
                policy_log_probs,
                reference_log_probs,
                completion_mask,
                rewards,
                kl_coefficient=config.kl_coefficient,
                advantage_epsilon=config.advantage_epsilon,
                advantage_estimator=config.advantage_estimator,
            )
        replay_applied = (
            config.supervised_replay_every_steps > 0
            and (state.rollout_step + 1)
            % config.supervised_replay_every_steps
            == 0
        )
        policy_signal = bool(
            float(tensors["advantage_abs_mean"].detach()) > 0.0
        )
        replay_only = replay_applied and not policy_signal
        replay_loss = torch.zeros((), dtype=loss.dtype, device=device)
        replay_tokens = 0
        replay_text_tokens: int | None = None
        replay_sample_id = ""
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if replay_applied:
            replay_index = random.randrange(len(active_replay_dataset))
            with _autocast_context(device, config.precision):
                (
                    replay_loss,
                    replay_tokens,
                    replay_text_tokens,
                ) = supervised_replay_loss(
                    policy,
                    active_replay_dataset,
                    collator,
                    replay_index,
                    device,
                )
            replay_sample_id = active_replay_dataset.samples[
                replay_index
            ].sample_id
            scaler.scale(
                config.supervised_replay_loss_coefficient * replay_loss
            ).backward()
        total_loss = (
            loss.detach()
            + config.supervised_replay_loss_coefficient * replay_loss.detach()
        )
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        vision_tokens = _prompt_vision_tokens(prompt_batch, policy)
        step_flops = estimate_rlvr_step_flops(
            policy.config,
            vision_tokens=vision_tokens,
            prompt_tokens=int(prompt_batch["input_ids"].shape[1]),
            completion_tokens=int(completion_ids.shape[1]),
            group_size=config.group_size,
            replay_text_tokens=replay_text_tokens,
            use_kv_cache=config.use_kv_cache,
            checkpoint_components=(
                config.gradient_checkpointing_components
                if config.gradient_checkpointing
                else ()
            ),
        )
        state.student_flops_seen += step_flops["total"]
        state.checkpoint_recompute_flops_seen += step_flops[
            "checkpoint_recompute"
        ]
        state.rollout_step += 1
        state.optimizer_step += 1
        state.policy_signal_steps += int(policy_signal)
        state.replay_only_steps += int(replay_only)
        valid_fraction = sum(
            result.structurally_valid for result in reward_results
        ) / len(reward_results)
        malformed_fraction = 1.0 - valid_fraction
        final_metrics = {
            f"rlvr/{name}": float(value.detach())
            for name, value in tensors.items()
        }
        final_metrics.update(
            {
                "rlvr/gradient_norm": float(gradient_norm),
                "rlvr/valid_structure_fraction": valid_fraction,
                "rlvr/rollout_step": float(state.rollout_step),
                "rlvr/optimizer_step": float(state.optimizer_step),
                "rlvr/policy_signal_step": float(policy_signal),
                "rlvr/policy_signal_steps": float(
                    state.policy_signal_steps
                ),
                "rlvr/replay_only_step": float(replay_only),
                "rlvr/replay_only_steps": float(
                    state.replay_only_steps
                ),
                "rlvr/malformed_fraction": malformed_fraction,
                "rlvr/generation_token_budget": float(
                    generation_token_budget
                ),
                "rlvr/generation_budget_escalated": float(
                    generation_budget_source != "default"
                ),
                "rlvr/student_flops_seen": float(
                    state.student_flops_seen
                ),
                "rlvr/step_student_flops": float(step_flops["total"]),
                "rlvr/checkpoint_recompute_flops_seen": float(
                    state.checkpoint_recompute_flops_seen
                ),
                "rlvr/executed_student_flops_seen": float(
                    state.student_flops_seen
                    + state.checkpoint_recompute_flops_seen
                ),
                "rlvr/step_checkpoint_recompute_flops": float(
                    step_flops["checkpoint_recompute"]
                ),
                "rlvr/step_executed_student_flops": float(
                    step_flops["executed_total"]
                ),
                "rlvr/total_loss": float(total_loss),
                "rlvr/supervised_replay_applied": float(replay_applied),
                "rlvr/supervised_replay_loss": float(replay_loss.detach()),
                "rlvr/supervised_replay_tokens": float(replay_tokens),
                "rlvr/preference_warm_start": float(
                    config.policy_start_stage.startswith("preference:")
                ),
            }
        )
        for name in reward_config.weights:
            values = [
                result.components[name]
                for result in reward_results
                if name in result.applicable
            ]
            if values:
                final_metrics[f"reward/{name}"] = sum(values) / len(values)
        for diagnostic in (
            "malformed_recovery_similarity",
            "malformed_recovery_reward",
            "rationale_text_similarity",
            "rationale_program_fact_score",
            "program_trace_consistency",
        ):
            values = [
                result.components[diagnostic]
                for result in reward_results
                if diagnostic in result.applicable
            ]
            if values:
                final_metrics[f"reward_diagnostic/{diagnostic}"] = (
                    sum(values) / len(values)
                )
        if (
            state.rollout_step == 1
            or state.rollout_step % config.log_every_steps == 0
        ):
            _record_metric(
                output_dir,
                {
                    "kind": "rlvr",
                    "sample_id": dataset.samples[sample_index].sample_id,
                    "supervised_replay_sample_id": replay_sample_id,
                    "advantage_estimator": config.advantage_estimator,
                    "generation_token_budget_source": (
                        generation_budget_source
                    ),
                    **final_metrics,
                },
                metric_callback,
            )
            print(
                f"[rlvr] step={state.rollout_step} "
                f"student_flops={state.student_flops_seen:,} "
                f"reward={float(tensors['reward_mean'].detach()):.4f} "
                f"valid={valid_fraction:.2f} "
                f"kl={float(tensors['reference_kl'].detach()):.5f} "
                f"replay={int(replay_applied)}",
                flush=True,
            )
        if (
            config.checkpoint_every_steps > 0
            and state.rollout_step % config.checkpoint_every_steps == 0
        ):
            last_checkpoint = _save_rlvr_checkpoint(
                policy,
                optimizer,
                scaler,
                state,
                config,
                reward_config,
            )
    if (
        last_checkpoint is None
        or last_checkpoint.name != f"step-{state.rollout_step:08d}"
    ):
        last_checkpoint = _save_rlvr_checkpoint(
            policy,
            optimizer,
            scaler,
            state,
            config,
            reward_config,
        )
    return RLVRResult(
        output_dir=str(output_dir),
        rollout_step=state.rollout_step,
        optimizer_step=state.optimizer_step,
        policy_signal_steps=state.policy_signal_steps,
        replay_only_steps=state.replay_only_steps,
        student_flops_seen=state.student_flops_seen,
        checkpoint_recompute_flops_seen=(
            state.checkpoint_recompute_flops_seen
        ),
        executed_student_flops_seen=(
            state.student_flops_seen
            + state.checkpoint_recompute_flops_seen
        ),
        last_checkpoint=str(last_checkpoint),
        final_metrics=final_metrics,
    )
