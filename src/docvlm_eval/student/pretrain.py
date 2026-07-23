"""Token-scheduled, mixed-precision, resumable pretraining for the native student."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .data import student_model_inputs
from .curriculum import CurriculumSchedule, planned_optimizer_steps
from .distillation import DistillationLoss, NativeStudentTeacher, TeacherSignals
from .model import DocumentVLMStudent


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
    token_unit: str = "supervised"
    visual_tokens_per_image: int = 0
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    precision: str = "auto"
    checkpoint_every_steps: int = 1000
    eval_every_steps: int = 1000
    log_every_steps: int = 10
    seed: int = 7
    device: str = "auto"
    resume_from: str | None = None
    tokenizer_fingerprint: str | None = None
    run_stage: str = "pretraining"
    loss_weights: dict[str, float] = field(default_factory=dict)
    curriculum: CurriculumSchedule = field(default_factory=CurriculumSchedule)

    def __post_init__(self) -> None:
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be positive when set")
        if self.epochs is None and not self.stop_at_total_tokens:
            raise ValueError("epochs can be null only when stop_at_total_tokens is true")
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
        if self.checkpoint_every_steps < 0 or self.eval_every_steps < 0:
            raise ValueError("checkpoint and evaluation intervals must be non-negative")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        if any(weight < 0 for weight in self.loss_weights.values()):
            raise ValueError("pretraining loss weights must be non-negative")
        if not self.run_stage.strip():
            raise ValueError("run_stage cannot be empty")
        self.curriculum.validate()
        if (
            self.stop_at_total_tokens
            and self.curriculum.stages
            and self.curriculum.unit != "training_token_fraction"
            and self.epochs is None
        ):
            raise ValueError(
                "an unbounded token-budget run requires a "
                "training_token_fraction curriculum"
            )

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        output_dir: str | Path,
        **overrides: Any,
    ) -> "PretrainConfig":
        raw = blueprint["training"]["pretraining"]["optimizer"]
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
            "token_unit": str(raw.get("token_unit", "supervised")),
            "visual_tokens_per_image": int(
                blueprint["student"]["connector"]["latent_tokens"]
            ),
            "grad_accum_steps": int(raw["grad_accum_steps"]),
            "max_grad_norm": float(raw["max_grad_norm"]),
            "precision": str(raw["precision"]),
            "checkpoint_every_steps": int(raw["checkpoint_every_steps"]),
            "eval_every_steps": int(raw["eval_every_steps"]),
            "log_every_steps": int(raw["log_every_steps"]),
            "seed": int(raw["seed"]),
            "loss_weights": {
                str(name): float(weight)
                for name, weight in blueprint["training"]["pretraining"]["losses"].items()
            },
            "curriculum": CurriculumSchedule.from_blueprint(blueprint),
        }
        values.update(overrides)
        return cls(**values)


@dataclass
class TrainerState:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    text_tokens_seen: int = 0
    effective_tokens_seen: int = 0


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    global_step: int
    tokens_seen: int
    text_tokens_seen: int
    effective_tokens_seen: int
    budget_tokens_seen: int
    token_unit: str
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
    ):
        super().__init__()
        self.student = student
        self.distillation_loss = distillation_loss

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
        else 0
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


def _save_checkpoint(
    module: PretrainingModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: TrainerState,
    config: PretrainConfig,
    context: _DistributedContext,
    curriculum_total_steps: int,
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
            "curriculum_fingerprint": config.curriculum.fingerprint,
            "curriculum_total_steps": (
                curriculum_total_steps if config.curriculum.stages else None
            ),
            "token_budget": {
                "stop_at_total_tokens": config.stop_at_total_tokens,
                "total_tokens": config.total_tokens,
                "token_unit": config.token_unit,
                "visual_tokens_per_image": config.visual_tokens_per_image,
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
    expected_curriculum_fingerprint: str | None,
    expected_curriculum_total_steps: int,
    expected_token_budget: dict[str, Any],
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
    return TrainerState(
        **json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    )


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
    if getattr(train_loader, "persistent_workers", False):
        raise ValueError("exact-resume augmentation requires persistent_workers=False")
    context = _distributed_context(config.device)
    random.seed(config.seed + context.rank)
    torch.manual_seed(config.seed + context.rank)
    output_dir = Path(config.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    module = PretrainingModule(student, distillation_loss).to(context.device)
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
    scheduler = TokenCosineScheduler(
        optimizer,
        config.learning_rate,
        config.warmup_tokens,
        config.total_tokens,
        config.min_lr_ratio,
    )
    if config.curriculum.unit == "training_token_fraction":
        curriculum_horizon = config.total_tokens
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
            config.curriculum.fingerprint,
            curriculum_horizon,
            {
                "stop_at_total_tokens": config.stop_at_total_tokens,
                "total_tokens": config.total_tokens,
                "token_unit": config.token_unit,
                "visual_tokens_per_image": config.visual_tokens_per_image,
            },
        )
        scheduler.step(_state_token_count(state, config.token_unit))

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
    epoch = state.epoch
    while not stop and (config.epochs is None or epoch < config.epochs):
        _set_loader_epoch(train_loader, epoch, config.seed, context.rank)
        module.train()
        loader_length = len(train_loader)
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == state.epoch and batch_index < state.batch_in_epoch:
                continue
            batch = _move_batch(raw_batch, context.device)
            if config.curriculum.unit == "training_token_fraction":
                curriculum_progress = min(
                    _state_token_count(state, config.token_unit)
                    / config.total_tokens,
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
            with sync_context:
                with _autocast_context(context.device, config.precision):
                    teacher_signals = teacher(batch) if teacher is not None else None
                    total, losses = wrapped(
                        batch,
                        teacher_signals,
                        active_loss_weights,
                    )
                    scaled_loss = total / config.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            accumulated_microbatches = microbatch_number
            batch_token_counts = _batch_token_counts(
                batch,
                config.visual_tokens_per_image,
            )
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
            learning_rate = scheduler.step(budget_tokens_seen)
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
                    "train/global_step": float(state.global_step),
                    "train/curriculum_stage": (
                        curriculum_stage.id if curriculum_stage is not None else "base"
                    ),
                    "train/curriculum_progress": curriculum_progress,
                    **{
                        f"train/loss_weight/{name}": weight
                        for name, weight in sorted(active_loss_weights.items())
                    },
                }
            )
            accumulated_token_counts = {
                "supervised": 0,
                "text": 0,
                "effective": 0,
            }
            accumulated_losses = {}
            accumulated_microbatches = 0

            if context.is_main and (
                state.global_step == 1
                or state.global_step % config.log_every_steps == 0
            ):
                _append_metric(output_dir, {"kind": "train", **means})
                print(
                    f"[student] step={state.global_step} "
                    f"{config.token_unit}_tokens={budget_tokens_seen:,} "
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
                if context.is_main:
                    _append_metric(
                        output_dir,
                        {
                            "kind": "eval",
                            "train/global_step": state.global_step,
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
                )
            reached_step_limit = (
                config.max_steps is not None
                and state.global_step >= config.max_steps
            )
            reached_token_limit = (
                config.stop_at_total_tokens
                and budget_tokens_seen >= config.total_tokens
            )
            if reached_step_limit or reached_token_limit:
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
        budget_tokens_seen=budget_tokens_seen,
        token_unit=config.token_unit,
        last_checkpoint=str(last_checkpoint),
        final_metrics=final_metrics,
    )
