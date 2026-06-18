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
from .patterns import DocBuilder, esc
from .render import RenderResult, render_html, resolve_boxes

__all__ = [
    "DocBuilder", "esc", "render_html", "resolve_boxes", "RenderResult", "degrade", "PRESETS",
]
