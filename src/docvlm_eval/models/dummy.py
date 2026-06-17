"""A dependency-free dummy model for CI / smoke testing the pipeline on CPU.

It needs no weights, no GPU and no transformers - it just returns a deterministic answer
derived from the image+question and a pseudo-confidence. Its only purpose is to prove the
end-to-end plumbing (loading -> generate -> scoring -> aggregation -> table) works before
spending GPU time on real models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@register("dummy-echo")
@dataclass
class DummyEcho(ModelAdapter):
    family: str = "dummy"
    hf_id: str = "(none)"
    param_count_m: float = 0.0

    def load(self) -> None:
        self._loaded = True

    def generate(self, image_path: str, question: str):
        # deterministic "answer": first salient token of the question
        toks = [t for t in question.replace("?", " ").split() if t.isalnum()]
        answer = toks[-1] if toks else "n/a"
        h = int(hashlib.md5((image_path + question).encode()).hexdigest(), 16)
        confidence = 0.5 + (h % 1000) / 2000.0  # in [0.5, 1.0)
        return answer, confidence
