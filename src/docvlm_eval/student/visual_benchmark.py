"""Reproducible benchmark for packed student vision attention backends."""

from __future__ import annotations

import hashlib
import math
import platform
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .config import StudentConfig, student_config_fingerprint
from .model import DocumentVLMStudent, GatedResampler, VisionTower


Backend = Literal["loop", "auto", "flex"]
Mode = Literal["forward", "training"]
Precision = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class VisualBenchmarkConfig:
    sequence_lengths: tuple[int, ...]
    backends: tuple[Backend, ...] = ("loop", "auto", "flex")
    warmup_iterations: int = 3
    measured_iterations: int = 10
    mode: Mode = "training"
    precision: Precision = "auto"
    device: str = "auto"
    seed: int = 7
    require_flex: bool = False
    parity_atol: float | None = None

    def validate(self, student: StudentConfig) -> None:
        if not self.sequence_lengths or any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("sequence_lengths must contain positive integers")
        if max(self.sequence_lengths) > student.vision.max_position_tokens:
            raise ValueError("each sequence length must be at most vision.max_position_tokens")
        if not self.backends or any(
            backend not in {"loop", "auto", "flex"} for backend in self.backends
        ):
            raise ValueError("backends must contain loop, auto, or flex")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must not contain duplicates")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be positive")
        if self.mode not in {"forward", "training"}:
            raise ValueError("mode must be forward or training")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("unsupported precision")


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_precision(name: Precision, device: torch.device) -> str:
    if name != "auto":
        return name
    if device.type == "cuda":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def _autocast(precision: str, device: torch.device):
    if precision == "float32":
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(
            f"{precision} autocast is not supported by this benchmark on {device.type}"
        )
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_type": device.type,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        result.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return result


def _default_parity_atol(precision: str) -> float:
    return {
        "float32": 1e-4,
        "float16": 5e-3,
        "bfloat16": 2e-2,
    }[precision]


def _packed_inputs(
    student: StudentConfig,
    lengths: tuple[int, ...],
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total = sum(lengths)
    patch = student.vision.patch_size
    pixels = torch.randn(
        total,
        3,
        patch,
        patch,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    positions = torch.cat([torch.arange(length, dtype=torch.long) for length in lengths]).to(device)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu_seqlens = torch.tensor(cumulative, dtype=torch.int32, device=device)
    return pixels, positions, cu_seqlens


def _resolved_backend(vision: VisionTower, connector: GatedResampler) -> str:
    if (
        vision.last_packed_attention_backend == "flex"
        and connector.last_packed_attention_backend == "flex"
    ):
        return "flex"
    return "loop"


def _forward(
    vision: VisionTower,
    connector: GatedResampler,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    backend: Backend,
    precision: str,
    device: torch.device,
) -> torch.Tensor:
    pixels, positions, cu_seqlens = inputs
    with _autocast(precision, device):
        vision_tokens, _ = vision.forward_packed(
            pixels,
            positions,
            cu_seqlens,
            attention_backend=backend,
        )
        return connector.forward_packed(
            vision_tokens,
            cu_seqlens,
            attention_backend=backend,
        )


def _parity_output(
    vision: VisionTower,
    connector: GatedResampler,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    backend: Backend,
    precision: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    vision.eval()
    connector.eval()
    with torch.no_grad():
        output = _forward(vision, connector, inputs, backend, precision, device)
    _synchronize(device)
    return output.detach().float().cpu(), _resolved_backend(vision, connector)


def _measure_backend(
    vision: VisionTower,
    connector: GatedResampler,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    backend: Backend,
    config: VisualBenchmarkConfig,
    precision: str,
    device: torch.device,
    reference: torch.Tensor,
    parity_atol: float,
) -> dict[str, Any]:
    output, resolved = _parity_output(
        vision,
        connector,
        inputs,
        backend,
        precision,
        device,
    )
    max_abs_delta = float((output - reference).abs().max().item())
    if max_abs_delta > parity_atol:
        raise RuntimeError(
            f"{backend} parity delta {max_abs_delta:.6g} exceeds atol={parity_atol:.6g}"
        )

    is_training = config.mode == "training"
    vision.train(is_training)
    connector.train(is_training)

    def iteration() -> None:
        if is_training:
            vision.zero_grad(set_to_none=True)
            connector.zero_grad(set_to_none=True)
            result = _forward(
                vision,
                connector,
                inputs,
                backend,
                precision,
                device,
            )
            result.float().square().mean().backward()
        else:
            with torch.no_grad():
                _forward(
                    vision,
                    connector,
                    inputs,
                    backend,
                    precision,
                    device,
                )

    for _ in range(config.warmup_iterations):
        iteration()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    for _ in range(config.measured_iterations):
        _synchronize(device)
        started = time.perf_counter()
        iteration()
        _synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1_000.0)

    resolved = _resolved_backend(vision, connector)
    median_ms = statistics.median(latencies_ms)
    record: dict[str, Any] = {
        "status": "ok",
        "requested_backend": backend,
        "resolved_backend": resolved,
        "mode": config.mode,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": median_ms,
        "p95_ms": _percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "tokens_per_second": sum(config.sequence_lengths) / (median_ms / 1_000.0),
        "max_abs_delta_vs_loop": max_abs_delta,
        "parity_atol": parity_atol,
        "output_checksum": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
    }
    if device.type == "cuda":
        record["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        record["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return record


def run_visual_backend_benchmark(
    student: StudentConfig,
    config: VisualBenchmarkConfig,
) -> dict[str, Any]:
    """Benchmark exact blueprint vision modules and return a JSON-safe report."""

    config.validate(student)
    errors = student.validate()
    if errors:
        raise ValueError("; ".join(errors))
    device = _resolve_device(config.device)
    precision = _resolve_precision(config.precision, device)
    parity_atol = config.parity_atol
    if parity_atol is None:
        parity_atol = _default_parity_atol(precision)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    vision = VisionTower(student.vision).to(device)
    connector = GatedResampler(student.connector).to(device)
    vision.apply(DocumentVLMStudent._init_weights)
    connector.apply(DocumentVLMStudent._init_weights)
    torch.nn.init.normal_(connector.latents, std=0.02)
    torch.nn.init.zeros_(vision.position_embedding)
    inputs = _packed_inputs(
        student,
        config.sequence_lengths,
        device=device,
        seed=config.seed,
    )
    reference, _ = _parity_output(
        vision,
        connector,
        inputs,
        "loop",
        precision,
        device,
    )

    results: list[dict[str, Any]] = []
    for backend in config.backends:
        try:
            result = _measure_backend(
                vision,
                connector,
                inputs,
                backend=backend,
                config=config,
                precision=precision,
                device=device,
                reference=reference,
                parity_atol=parity_atol,
            )
        except Exception as error:
            result = {
                "status": "error",
                "requested_backend": backend,
                "resolved_backend": None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        results.append(result)

    loop_record = next(
        (
            record
            for record in results
            if record["requested_backend"] == "loop" and record["status"] == "ok"
        ),
        None,
    )
    if loop_record is not None:
        for record in results:
            if record["status"] != "ok":
                continue
            record["median_speedup_vs_loop"] = loop_record["median_ms"] / record["median_ms"]
            loop_memory = loop_record["peak_memory_allocated_bytes"]
            record_memory = record["peak_memory_allocated_bytes"]
            if loop_memory is not None and record_memory is not None:
                record["peak_memory_ratio_vs_loop"] = record_memory / loop_memory
                record["peak_memory_reduction_fraction_vs_loop"] = 1.0 - record_memory / loop_memory
            else:
                record["peak_memory_ratio_vs_loop"] = None
                record["peak_memory_reduction_fraction_vs_loop"] = None

    flex_records = [record for record in results if record["requested_backend"] in {"auto", "flex"}]
    flex_gate_passed = bool(flex_records) and all(
        record["status"] == "ok" and record["resolved_backend"] == "flex" for record in flex_records
    )
    report = {
        "schema_version": 1,
        "scope": "student_vision_tower_and_gated_resampler",
        "language_decoder_included": False,
        "student_config_fingerprint": student_config_fingerprint(student),
        "student_config": student.to_dict(),
        "benchmark_config": asdict(config),
        "resolved_precision": precision,
        "environment": _environment(device),
        "visual_tokens": sum(config.sequence_lengths),
        "batch_size": len(config.sequence_lengths),
        "gates": {
            "require_flex": config.require_flex,
            "flex_resolved": flex_gate_passed,
            "passed": not config.require_flex or flex_gate_passed,
        },
        "results": results,
    }
    return report
