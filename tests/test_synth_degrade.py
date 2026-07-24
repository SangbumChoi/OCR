"""Deterministic, fail-closed synthetic degradation retries."""

import importlib

import pytest
from PIL import Image

from docvlm_eval.synth.degrade import (
    RETRY_SEED_STRIDE,
    DegradationError,
    degrade_with_retries,
    derive_degradation_seed,
)


def test_degradation_seed_is_stable_and_variant_specific():
    first = derive_degradation_seed(7, "invoice", "0000", "en")
    repeated = derive_degradation_seed(7, "invoice", "0000", "en")
    second = derive_degradation_seed(7, "invoice", "0001", "en")

    assert first == repeated
    assert first != second


def test_degrade_retries_runtime_and_validator_failures(monkeypatch):
    degradation_module = importlib.import_module("docvlm_eval.synth.degrade")
    image = Image.new("RGB", (20, 20), "white")
    seeds = []

    def fake_degrade(img, _preset, seed=None):
        seeds.append(seed)
        if len(seeds) == 1:
            raise AssertionError("backend failure")
        return img.copy()

    validations = []

    def validator(_candidate):
        validations.append(True)
        if len(validations) == 1:
            raise ValueError("quality failure")

    monkeypatch.setattr(degradation_module, "degrade", fake_degrade)
    result, seed, attempts = degrade_with_retries(
        image,
        "fax",
        base_seed=7,
        max_attempts=3,
        validator=validator,
    )

    assert result.size == image.size
    assert seed == 7 + 2 * RETRY_SEED_STRIDE
    assert attempts == 3
    assert seeds == [7, 7 + RETRY_SEED_STRIDE, 7 + 2 * RETRY_SEED_STRIDE]


def test_degrade_fails_closed_after_bounded_attempts(monkeypatch):
    degradation_module = importlib.import_module("docvlm_eval.synth.degrade")
    image = Image.new("RGB", (20, 20), "white")
    seeds = []

    def unavailable(_img, _preset, seed=None):
        seeds.append(seed)
        return None

    monkeypatch.setattr(degradation_module, "degrade", unavailable)

    with pytest.raises(DegradationError, match="after 2 attempt"):
        degrade_with_retries(
            image,
            "scan",
            base_seed=11,
            max_attempts=2,
        )

    assert seeds == [11, 11 + RETRY_SEED_STRIDE]
