"""Synthetic document generation with built-in ground truth.

Render HTML/CSS (single source of truth) -> raster + exact GT boxes, then optionally apply
photometric degradation that keeps the boxes valid. Heavy deps (weasyprint, pymupdf, augraphy,
faker) live in the ``[synth]`` extra and are imported lazily.

    from docvlm_eval.synth import DocBuilder, degrade
    b = DocBuilder("invoice", ["table"], "KIE F1")
    b.field("Invoice No", "INV-1", key="invoice_no", spot=True)
    img, gt = b.build(dpi=150)
    deg = degrade(img, "scan")
"""

from .degrade import PRESETS, degrade
from .derive import (
    Derivation, aggregate, count_occurrences, locate, region_box, resolve, union_box, word_boxes,
)
from .dto import (
    AblationSupport, BBox, Degradation, DocSample, Field, GenConfig, QAItem, RenderSpec,
    script_for,
)
from .patterns import DocBuilder, esc
from .render import RenderResult, render_html, resolve_boxes
from .to_samples import case_to_samples, load_case_dir, load_realistic_samples

__all__ = [
    "DocBuilder", "esc", "render_html", "resolve_boxes", "RenderResult", "degrade", "PRESETS",
    "case_to_samples", "load_case_dir", "load_realistic_samples",
    # structured GT + generation-config DTOs
    "DocSample", "GenConfig", "Field", "QAItem", "BBox", "RenderSpec", "Degradation",
    "AblationSupport", "script_for",
    # model-free understanding-GT derivation
    "Derivation", "resolve", "locate", "count_occurrences", "region_box", "union_box",
    "word_boxes", "aggregate",
]
