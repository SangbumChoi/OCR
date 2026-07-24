from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest
import torch

from docvlm_eval.student.optim import (
    OptimizerSpec,
    build_optimizer,
    optimizer_runtime_contract,
)


def test_standard_adamw_builds_and_records_exact_runtime():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    spec = OptimizerSpec(eps=1e-7)

    optimizer = build_optimizer(
        [parameter],
        spec,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
    )
    contract = optimizer_runtime_contract(optimizer, spec)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["eps"] == pytest.approx(1e-7)
    assert contract["spec"] == spec.to_dict()
    assert contract["implementation"].endswith(".AdamW")
    assert contract["bitsandbytes_version"] is None


def test_adamw_8bit_forwards_precision_controls(monkeypatch):
    captured = {}

    class FakeAdamW8bit(torch.optim.AdamW):
        def __init__(self, parameters, **kwargs):
            captured.update(kwargs)
            kwargs.pop("min_8bit_size")
            kwargs.pop("block_wise")
            super().__init__(parameters, **kwargs)

    package = ModuleType("bitsandbytes")
    optim = ModuleType("bitsandbytes.optim")
    optim.AdamW8bit = FakeAdamW8bit
    package.optim = optim
    monkeypatch.setitem(sys.modules, "bitsandbytes", package)
    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", optim)
    spec = OptimizerSpec(
        name="adamw_8bit",
        min_8bit_size=8192,
        block_wise=False,
    )

    optimizer = build_optimizer(
        [torch.nn.Parameter(torch.tensor([1.0]))],
        spec,
        learning_rate=2e-4,
        betas=(0.8, 0.9),
    )

    assert isinstance(optimizer, FakeAdamW8bit)
    assert captured["min_8bit_size"] == 8192
    assert captured["block_wise"] is False
    assert captured["eps"] == pytest.approx(1e-8)


def test_adamw_8bit_never_silently_falls_back(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "bitsandbytes.optim":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="working bitsandbytes"):
        build_optimizer(
            [torch.nn.Parameter(torch.tensor([1.0]))],
            OptimizerSpec(name="adamw_8bit"),
            learning_rate=3e-4,
            betas=(0.9, 0.95),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "sgd"}, "optimizer name"),
        ({"eps": 0}, "optimizer eps"),
        ({"min_8bit_size": 0}, "min_8bit_size"),
        ({"block_wise": "yes"}, "block_wise"),
    ],
)
def test_optimizer_spec_rejects_invalid_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OptimizerSpec(**kwargs)
