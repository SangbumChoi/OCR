"""Matched benchmark for packed and dense student vision execution policies."""

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


Backend = Literal[
    "loop",
    "auto",
    "flex",
    "dense_adaptive",
    "dense_fixed_square",
]
Mode = Literal["forward", "training"]
Precision = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class VisualBenchmarkConfig:
    sequence_lengths: tuple[int, ...] | None = None
    patch_grids: tuple[tuple[int, int], ...] | None = None
    backends: tuple[Backend, ...] = (
        "loop",
        "auto",
        "flex",
        "dense_adaptive",
        "dense_fixed_square",
    )
    warmup_iterations: int = 3
    measured_iterations: int = 10
    mode: Mode = "training"
    precision: Precision = "auto"
    device: str = "auto"
    seed: int = 7
    require_flex: bool = False
    parity_atol: float | None = None

    @property
    def resolved_sequence_lengths(self) -> tuple[int, ...]:
        if self.patch_grids is not None:
            return tuple(height * width for height, width in self.patch_grids)
        return tuple(self.sequence_lengths or ())

    def validate(self, student: StudentConfig) -> None:
        lengths = self.resolved_sequence_lengths
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("sequence_lengths must contain positive integers")
        if self.sequence_lengths is not None and self.patch_grids is not None:
            raise ValueError("set only one of sequence_lengths or patch_grids")
        if max(lengths) > student.vision.max_position_tokens:
            raise ValueError("each sequence length must be at most vision.max_position_tokens")
        grid_side = math.isqrt(student.vision.max_position_tokens)
        if self.patch_grids is not None and any(
            height <= 0
            or width <= 0
            or height > grid_side
            or width > grid_side
            for height, width in self.patch_grids
        ):
            raise ValueError("patch_grids exceed the visual position grid")
        if not self.backends or any(
            backend
            not in {
                "loop",
                "auto",
                "flex",
                "dense_adaptive",
                "dense_fixed_square",
            }
            for backend in self.backends
        ):
            raise ValueError("unsupported visual benchmark backend")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must not contain duplicates")
        if any(backend.startswith("dense_") for backend in self.backends) and (
            self.patch_grids is None
        ):
            raise ValueError("dense policies require patch_grids")
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


@dataclass(frozen=True)
class _CanonicalInputs:
    patches: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    patch_grids: tuple[tuple[int, int], ...] | None


def _canonical_inputs(
    student: StudentConfig,
    lengths: tuple[int, ...],
    *,
    patch_grids: tuple[tuple[int, int], ...] | None,
    seed: int,
) -> _CanonicalInputs:
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
    )
    if patch_grids is None:
        positions = torch.cat(
            [torch.arange(length, dtype=torch.long) for length in lengths]
        )
    else:
        grid_side = math.isqrt(student.vision.max_position_tokens)
        positions = torch.cat(
            [
                (
                    torch.arange(height)[:, None] * grid_side
                    + torch.arange(width)[None, :]
                ).flatten()
                for height, width in patch_grids
            ]
        )
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return _CanonicalInputs(
        patches=pixels,
        position_ids=positions,
        cu_seqlens=torch.tensor(cumulative, dtype=torch.int32),
        patch_grids=patch_grids,
    )


def _materialize_inputs(
    student: StudentConfig,
    canonical: _CanonicalInputs,
    backend: Backend,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend in {"loop", "auto", "flex"}:
        return (
            canonical.patches.to(device),
            canonical.position_ids.to(device),
            canonical.cu_seqlens.to(device),
        )
    if canonical.patch_grids is None:
        raise ValueError("dense policies require patch_grids")
    grid_side = math.isqrt(student.vision.max_position_tokens)
    if backend == "dense_fixed_square":
        canvas_height = grid_side
        canvas_width = grid_side
    else:
        canvas_height = max(height for height, _ in canonical.patch_grids)
        canvas_width = max(width for _, width in canonical.patch_grids)
    patch = student.vision.patch_size
    images = torch.zeros(
        len(canonical.patch_grids),
        3,
        canvas_height * patch,
        canvas_width * patch,
        dtype=canonical.patches.dtype,
    )
    masks = torch.zeros(
        len(canonical.patch_grids),
        canvas_height * patch,
        canvas_width * patch,
        dtype=torch.bool,
    )
    offset = 0
    for index, (height, width) in enumerate(canonical.patch_grids):
        count = height * width
        document = (
            canonical.patches[offset : offset + count]
            .view(height, width, 3, patch, patch)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, height * patch, width * patch)
        )
        images[index, :, : height * patch, : width * patch] = document
        masks[index, : height * patch, : width * patch] = True
        offset += count
    return images.to(device), masks.to(device), canonical.cu_seqlens.to(device)


def _resolved_backend(
    vision: VisionTower,
    connector: GatedResampler,
    backend: Backend,
) -> str:
    if backend in {"dense_adaptive", "dense_fixed_square"}:
        return "dense"
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
    pixels, second, cu_seqlens = inputs
    with _autocast(precision, device):
        if backend in {"dense_adaptive", "dense_fixed_square"}:
            vision_tokens, vision_mask = vision(
                pixels,
                second,
                return_mask=True,
            )
            return connector(vision_tokens, vision_mask)
        vision_tokens, _ = vision.forward_packed(
            pixels,
            second,
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
    return (
        output.detach().float().cpu(),
        _resolved_backend(vision, connector, backend),
    )


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

    resolved = _resolved_backend(vision, connector, backend)
    median_ms = statistics.median(latencies_ms)
    lengths = config.resolved_sequence_lengths
    if backend == "dense_fixed_square":
        side = math.isqrt(vision.config.max_position_tokens)
        executed_tokens = len(lengths) * side * side
    elif backend == "dense_adaptive":
        grids = config.patch_grids or ()
        executed_tokens = (
            len(grids)
            * max(height for height, _ in grids)
            * max(width for _, width in grids)
        )
    else:
        executed_tokens = sum(lengths)
    record: dict[str, Any] = {
        "status": "ok",
        "requested_backend": backend,
        "resolved_backend": resolved,
        "mode": config.mode,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": median_ms,
        "p95_ms": _percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "tokens_per_second": sum(lengths) / (median_ms / 1_000.0),
        "executed_visual_tokens": executed_tokens,
        "valid_visual_token_fraction": sum(lengths) / executed_tokens,
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
    lengths = config.resolved_sequence_lengths
    canonical = _canonical_inputs(
        student,
        lengths,
        patch_grids=config.patch_grids,
        seed=config.seed,
    )
    reference_inputs = _materialize_inputs(
        student,
        canonical,
        "loop",
        device,
    )
    reference, _ = _parity_output(
        vision,
        connector,
        reference_inputs,
        "loop",
        precision,
        device,
    )
    del reference_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    for backend in config.backends:
        inputs = _materialize_inputs(
            student,
            canonical,
            backend,
            device,
        )
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
        finally:
            del inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
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

    dense_record = next(
        (
            record
            for record in results
            if record["requested_backend"] == "dense_adaptive"
            and record["status"] == "ok"
        ),
        None,
    )
    if dense_record is not None:
        for record in results:
            if record["status"] != "ok":
                continue
            record["median_speedup_vs_dense_adaptive"] = (
                dense_record["median_ms"] / record["median_ms"]
            )
            dense_memory = dense_record["peak_memory_allocated_bytes"]
            record_memory = record["peak_memory_allocated_bytes"]
            if dense_memory is not None and record_memory is not None:
                record["peak_memory_ratio_vs_dense_adaptive"] = (
                    record_memory / dense_memory
                )
                record[
                    "peak_memory_reduction_fraction_vs_dense_adaptive"
                ] = (1.0 - record_memory / dense_memory)
            else:
                record["peak_memory_ratio_vs_dense_adaptive"] = None
                record[
                    "peak_memory_reduction_fraction_vs_dense_adaptive"
                ] = None

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
        "visual_tokens": sum(lengths),
        "batch_size": len(lengths),
        "patch_grids": (
            [list(grid) for grid in config.patch_grids]
            if config.patch_grids is not None
            else None
        ),
        "gates": {
            "require_flex": config.require_flex,
            "flex_resolved": flex_gate_passed,
            "passed": not config.require_flex or flex_gate_passed,
        },
        "results": results,
    }
    return report
