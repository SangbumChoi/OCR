"""Full-model multimodal training-step feasibility benchmark."""

from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .config import StudentConfig, student_config_fingerprint
from .compute import estimate_batch_training_flops_breakdown
from .model import DocumentVLMStudent
from .pretrain import (
    PretrainingModule,
    _autocast_context,
    _parameter_groups,
    _uses_fp16,
)


Precision = Literal["auto", "float32", "float16", "bfloat16"]
Backend = Literal["loop", "auto", "flex"]


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    patch_grid: tuple[int, int] = (40, 63)
    text_tokens: int = 2048
    micro_batch_size: int = 1
    warmup_steps: int = 1
    measured_steps: int = 2
    packed_attention_backend: Backend = "auto"
    precision: Precision = "auto"
    gradient_checkpointing: bool = False
    gradient_checkpointing_components: tuple[str, ...] = (
        "vision",
        "connector",
        "language",
    )
    gradient_checkpointing_use_reentrant: bool = False
    device: str = "auto"
    seed: int = 7

    @property
    def visual_tokens_per_sample(self) -> int:
        return self.patch_grid[0] * self.patch_grid[1]

    def validate(self, student: StudentConfig) -> None:
        height, width = self.patch_grid
        grid_side = math.isqrt(student.vision.max_position_tokens)
        if height <= 0 or width <= 0:
            raise ValueError("patch_grid dimensions must be positive")
        if height > grid_side or width > grid_side:
            raise ValueError("patch_grid exceeds the visual position grid")
        if self.text_tokens < 2:
            raise ValueError("text_tokens must be at least two")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.measured_steps <= 0:
            raise ValueError("measured_steps must be positive")
        if self.packed_attention_backend not in {"loop", "auto", "flex"}:
            raise ValueError("unsupported packed_attention_backend")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("unsupported precision")
        if (
            not self.gradient_checkpointing_components
            or len(set(self.gradient_checkpointing_components))
            != len(self.gradient_checkpointing_components)
            or not set(self.gradient_checkpointing_components)
            <= {"vision", "connector", "language"}
        ):
            raise ValueError(
                "gradient checkpointing components are invalid"
            )


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolved_precision(name: Precision, device: torch.device) -> str:
    if device.type != "cuda":
        return "float32"
    if name == "auto":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return name


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_type": device.type,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_total_memory_bytes": int(properties.total_memory),
                "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return result


def _memory_snapshot(device: torch.device) -> dict[str, int | float | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
            "peak_reserved_fraction": None,
        }
    free, total = torch.cuda.mem_get_info(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(peak_reserved),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "peak_reserved_fraction": float(peak_reserved / total),
    }


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _unique_parameter_count(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    return total


def _optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
) -> dict[str, int | float]:
    tensor_bytes = 0
    state_steps: list[float] = []
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                tensor_bytes += value.numel() * value.element_size()
                if name == "step" and value.numel() == 1:
                    state_steps.append(float(value.item()))
            elif name == "step":
                state_steps.append(float(value))
    return {
        "parameter_states": len(optimizer.state),
        "tensor_bytes": tensor_bytes,
        "min_step": min(state_steps, default=0.0),
        "max_step": max(state_steps, default=0.0),
    }


def _synthetic_batch(
    student: StudentConfig,
    config: TrainingBenchmarkConfig,
    device: torch.device,
    *,
    contrastive: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batch_size = config.micro_batch_size
    height, width = config.patch_grid
    tokens_per_sample = height * width
    patch = student.vision.patch_size
    patches = torch.randn(
        batch_size * tokens_per_sample,
        3,
        patch,
        patch,
        generator=generator,
    )
    grid_side = math.isqrt(student.vision.max_position_tokens)
    rows = torch.arange(height)[:, None]
    columns = torch.arange(width)[None, :]
    positions = (rows * grid_side + columns).flatten().repeat(batch_size)
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * tokens_per_sample,
        tokens_per_sample,
        dtype=torch.long,
    )
    input_ids = torch.randint(
        0,
        student.language.vocab_size,
        (batch_size, config.text_tokens),
        generator=generator,
    )
    labels = input_ids.clone()
    labels[:, : max(1, config.text_tokens // 2)] = -100
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool).to(device),
        "labels": labels.to(device),
        "packed_pixel_values": patches.to(device),
        "packed_position_ids": positions.to(device),
        "packed_cu_seqlens": cu_seqlens.to(device),
        "packed_attention_backend": config.packed_attention_backend,
        "box_targets": torch.tensor(
            [[0.1, 0.15, 0.8, 0.85]] * batch_size,
            dtype=torch.float32,
            device=device,
        ),
        "box_target_mask": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "box_query_positions": torch.full(
            (batch_size,),
            config.text_tokens - 1,
            dtype=torch.long,
            device=device,
        ),
        "orientation_labels": torch.arange(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        % 4,
        "contrastive": contrastive,
        "contrastive_ids": torch.arange(
            batch_size,
            dtype=torch.long,
            device=device,
        ),
    }


def _run_step(
    module: PretrainingModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: dict[str, Any],
    loss_weights: dict[str, float],
    *,
    precision: str,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> dict[str, Any]:
    before_state = _optimizer_state_summary(optimizer)
    started = time.perf_counter()
    with _autocast_context(device, precision):
        total, losses = module(batch, None, loss_weights)
        scaled_loss = total / grad_accum_steps
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer)
    if grad_accum_steps > 1:
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(grad_accum_steps)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        module.parameters(),
        max_grad_norm,
    )
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after_state = _optimizer_state_summary(optimizer)
    scalar_losses = {
        name: float(value.detach().float())
        for name, value in losses.items()
    }
    total_value = float(total.detach().float())
    finite = (
        math.isfinite(total_value)
        and math.isfinite(float(gradient_norm))
        and all(math.isfinite(value) for value in scalar_losses.values())
    )
    optimizer_advanced = (
        after_state["max_step"] > before_state["max_step"]
    )
    return {
        "elapsed_ms": elapsed_ms,
        "loss": total_value,
        "losses": scalar_losses,
        "gradient_norm": float(gradient_norm),
        "finite": finite,
        "optimizer_advanced": optimizer_advanced,
        "grad_scale_before": scale_before,
        "grad_scale_after": float(scaler.get_scale()),
        "optimizer_state": after_state,
    }


def run_training_feasibility_benchmark(
    student_config: StudentConfig,
    config: TrainingBenchmarkConfig,
    *,
    loss_weights: dict[str, float],
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    grad_accum_steps: int,
    max_grad_norm: float,
    contrastive: bool,
    box_iou_loss: str = "giou",
) -> dict[str, Any]:
    """Run real full-student micro-steps and retain evidence even on OOM."""

    config.validate(student_config)
    device = _resolve_device(config.device)
    precision = _resolved_precision(config.precision, device)
    environment = _environment(device)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "full_student_multimodal_training_step",
        "student_config": student_config.to_dict(),
        "student_config_fingerprint": student_config_fingerprint(student_config),
        "benchmark_config": {
            **asdict(config),
            "patch_grid": list(config.patch_grid),
            "resolved_precision": precision,
            "grad_accum_steps": grad_accum_steps,
            "microbatches_per_probe_step": 1,
            "short_final_batch_gradient_correction": True,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "max_grad_norm": max_grad_norm,
            "contrastive": contrastive,
            "box_iou_loss": box_iou_loss,
        },
        "environment": environment,
        "parameter_count": None,
        "status": "error",
        "error_type": None,
        "error": None,
        "oom": False,
        "resolved_visual_attention_backend": None,
        "gradient_checkpointing": None,
        "training_flops_per_microbatch": None,
        "setup_memory": None,
        "materialization_memory": None,
        "steady_state_memory": None,
        "effective_peak_memory": None,
        "warmup_steps": [],
        "measured_steps": [],
        "median_step_ms": None,
        "p95_step_ms": None,
        "steps_per_second": None,
        "all_finite": False,
        "all_optimizer_steps_succeeded": False,
        "optimizer_state": None,
    }
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        _reset_peak_memory(device)
    try:
        student = DocumentVLMStudent(student_config)
        student.configure_gradient_checkpointing(
            enabled=config.gradient_checkpointing,
            components=config.gradient_checkpointing_components,
            use_reentrant=(
                config.gradient_checkpointing_use_reentrant
            ),
        )
        report["gradient_checkpointing"] = (
            student.gradient_checkpointing_state
        )
        module = PretrainingModule(
            student,
            box_iou_loss_kind=box_iou_loss,
        ).to(device)
        module.train()
        report["parameter_count"] = _unique_parameter_count(module)
        optimizer = torch.optim.AdamW(
            _parameter_groups(module, weight_decay),
            lr=learning_rate,
            betas=betas,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=_uses_fp16(device, precision),
        )
        batch = _synthetic_batch(
            student_config,
            config,
            device,
            contrastive=contrastive,
        )
        training_flops = estimate_batch_training_flops_breakdown(
            student_config,
            batch,
            checkpoint_components=(
                config.gradient_checkpointing_components
                if config.gradient_checkpointing
                else ()
            ),
        )
        report["training_flops_per_microbatch"] = (
            training_flops.to_dict()
        )
        _synchronize(device)
        report["setup_memory"] = _memory_snapshot(device)

        _reset_peak_memory(device)
        warmup_records = []
        for _ in range(config.warmup_steps):
            warmup_records.append(
                _run_step(
                    module,
                    optimizer,
                    scaler,
                    batch,
                    loss_weights,
                    precision=precision,
                    device=device,
                    grad_accum_steps=grad_accum_steps,
                    max_grad_norm=max_grad_norm,
                )
            )
        report["warmup_steps"] = warmup_records
        report["materialization_memory"] = _memory_snapshot(device)

        _reset_peak_memory(device)
        measured_records = []
        for _ in range(config.measured_steps):
            measured_records.append(
                _run_step(
                    module,
                    optimizer,
                    scaler,
                    batch,
                    loss_weights,
                    precision=precision,
                    device=device,
                    grad_accum_steps=grad_accum_steps,
                    max_grad_norm=max_grad_norm,
                )
            )
        report["measured_steps"] = measured_records
        report["steady_state_memory"] = _memory_snapshot(device)
        memory_records = [
            record
            for record in (
                report["setup_memory"],
                report["materialization_memory"],
                report["steady_state_memory"],
            )
            if isinstance(record, dict)
            and record.get("peak_reserved_bytes") is not None
        ]
        if memory_records:
            effective = max(
                memory_records,
                key=lambda record: int(record["peak_reserved_bytes"]),
            )
            report["effective_peak_memory"] = dict(effective)
        elapsed = [record["elapsed_ms"] for record in measured_records]
        report["median_step_ms"] = statistics.median(elapsed)
        report["p95_step_ms"] = sorted(elapsed)[
            max(0, math.ceil(0.95 * len(elapsed)) - 1)
        ]
        report["steps_per_second"] = 1000.0 / report["median_step_ms"]
        all_records = [*warmup_records, *measured_records]
        report["all_finite"] = all(record["finite"] for record in all_records)
        report["all_optimizer_steps_succeeded"] = all(
            record["optimizer_advanced"] for record in all_records
        )
        report["optimizer_state"] = _optimizer_state_summary(optimizer)
        report["resolved_visual_attention_backend"] = (
            student.last_visual_attention_backend
        )
        report["status"] = "ok"
    except Exception as error:  # Evidence must survive CUDA OOM and backend failures.
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["oom"] = isinstance(error, torch.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )
        if device.type == "cuda":
            try:
                report["failure_memory"] = _memory_snapshot(device)
            except Exception:
                report["failure_memory"] = None
            torch.cuda.empty_cache()
    return report
