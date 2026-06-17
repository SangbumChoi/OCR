"""H2OVL-Mississippi-0.8B adapter.

An explicitly OCR/document-focused small VLM (H2O.ai), architecturally an InternVL-style
ViT + H2O-Danube LM with the same dynamic-tiling + ``model.chat`` interface, trained on ~19M
image-text pairs centred on OCR / document / chart / table understanding. Reported OCRBench
751, which makes it a strong document specialist in the sub-1B band.

Because it mirrors InternVL's interface, we reuse the InternVL adapter machinery.
Model card: https://huggingface.co/h2oai/h2ovl-mississippi-800m
"""

from __future__ import annotations

from dataclasses import dataclass

from .internvl import _InternVL
from .registry import register


@register("h2ovl-0.8b")
@dataclass
class H2OVLMississippi800M(_InternVL):
    family: str = "H2OVL-Mississippi"
    hf_id: str = "h2oai/h2ovl-mississippi-800m"
    param_count_m: float = 800.0
    max_tiles: int = 6
