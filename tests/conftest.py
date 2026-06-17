"""Shared test fixtures + path bootstrap.

Tests run on CPU with no torch/transformers (they exercise metrics, schema, loaders,
registry wiring, robustness, the catalog, the dummy model and the full pipeline). Anything
needing heavy deps is guarded/skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def tiny_image(tmp_path) -> str:
    """A small valid PNG on disk (used where a real image path is required)."""
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (64, 48), (220, 220, 220)).save(p)
    return str(p)


@pytest.fixture
def tiny_benchmark(tmp_path, tiny_image):
    """A 3-sample normalised benchmark + its JSONL path."""
    from docvlm_eval.benchmarks import save_jsonl
    from docvlm_eval.schema import Sample

    samples = [
        Sample("s0", tiny_image, "What is the total?", ["100"], "form/total", "anls"),
        Sample("s1", tiny_image, "value?", ["42"], "numeric", "relaxed_acc"),
        Sample("s2", tiny_image, "name?", ["acme"], "free-text", "exact"),
    ]
    path = tmp_path / "bench.jsonl"
    save_jsonl(samples, path)
    return samples, str(path)
