"""Fail-closed optimizer construction for native student training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Iterable, Mapping


OPTIMIZER_NAMES = {"adamw", "adamw_8bit"}


@dataclass(frozen=True)
class OptimizerSpec:
    """Optimizer identity and state-precision controls."""

    name: str = "adamw"
    eps: float = 1e-8
    min_8bit_size: int = 4096
    block_wise: bool = True

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZER_NAMES:
            raise ValueError(
                f"optimizer name must be one of {sorted(OPTIMIZER_NAMES)}"
            )
        if self.eps <= 0:
            raise ValueError("optimizer eps must be positive")
        if self.min_8bit_size <= 0:
            raise ValueError("optimizer min_8bit_size must be positive")
        if not isinstance(self.block_wise, bool):
            raise ValueError("optimizer block_wise must be a boolean")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OptimizerSpec":
        return cls(
            name=str(raw.get("name", "adamw")),
            eps=float(raw.get("eps", 1e-8)),
            min_8bit_size=int(raw.get("min_8bit_size", 4096)),
            block_wise=raw.get("block_wise", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def build_optimizer(
    parameters: Iterable[Any],
    spec: OptimizerSpec,
    *,
    learning_rate: float,
    betas: tuple[float, float],
) -> Any:
    """Build the exact requested optimizer without silent fallback."""

    if spec.name == "adamw":
        import torch

        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=spec.eps,
        )
    try:
        from bitsandbytes.optim import AdamW8bit
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "adamw_8bit requires a working bitsandbytes installation; "
            "install `docvlm-eval[student-gpu]` or select optimizer name adamw"
        ) from error
    return AdamW8bit(
        parameters,
        lr=learning_rate,
        betas=betas,
        eps=spec.eps,
        min_8bit_size=spec.min_8bit_size,
        block_wise=spec.block_wise,
    )


def optimizer_runtime_contract(
    optimizer: Any,
    spec: OptimizerSpec,
) -> dict[str, Any]:
    """Record requested and realized optimizer identity."""

    implementation = type(optimizer)
    return {
        "schema_version": 1,
        "spec": spec.to_dict(),
        "implementation": (
            f"{implementation.__module__}.{implementation.__qualname__}"
        ),
        "bitsandbytes_version": (
            _package_version("bitsandbytes")
            if spec.name == "adamw_8bit"
            else None
        ),
    }
