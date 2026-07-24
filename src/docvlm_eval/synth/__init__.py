"""Synthetic document generation with built-in ground truth.

Render HTML/CSS (single source of truth) -> raster + exact GT boxes, optionally transform pixels
and boxes through one homography, then apply photometric degradation in that coordinate frame.
Heavy deps (weasyprint, pymupdf, augraphy, faker) live in the ``[synth]`` extra and are imported
lazily.

    from docvlm_eval.synth import DocBuilder, degrade
    b = DocBuilder("invoice", ["table"], "KIE F1")
    b.field("Invoice No", "INV-1", key="invoice_no", spot=True)
    img, gt = b.build(dpi=150)
    deg = degrade(img, "scan")
"""

from .degrade import (
    DegradationError, PRESETS, degrade, degrade_with_retries, derive_degradation_seed,
)
from .geometry import (
    GeometryAugmentationError,
    derive_perspective_seed,
    transform_box,
    transform_ground_truth,
    warp_perspective,
)
from .derive import (
    Derivation, aggregate, count_occurrences, locate, region_box, resolve, union_box, word_boxes,
)
from .dto import (
    AblationSupport, BBox, Degradation, DocSample, Field, GenConfig, QAItem, RenderSpec,
    script_for,
)
from .patterns import DocBuilder, esc
from .latent import (
    DifficultySpec, GraphEdge, GraphNode, GraphQuery, LatentDocumentGraph, ResolvedQuery,
)
from .render import RenderResult, render_html, resolve_boxes
from .quality import (
    EvidenceQualityError, audit_degraded_evidence, audit_render_evidence, collect_evidence_boxes,
    redact_evidence_quality_report,
)
from .splits import SplitPolicy, validate_split_leakage
from .supervision import apply_supervision_toggles
from .to_samples import case_to_samples, load_case_dir, load_realistic_samples

__all__ = [
    "DocBuilder", "esc", "render_html", "resolve_boxes", "RenderResult", "degrade",
    "degrade_with_retries", "derive_degradation_seed", "DegradationError", "PRESETS",
    "warp_perspective", "transform_box", "transform_ground_truth",
    "derive_perspective_seed", "GeometryAugmentationError",
    "case_to_samples", "load_case_dir", "load_realistic_samples",
    # structured GT + generation-config DTOs
    "DocSample", "GenConfig", "Field", "QAItem", "BBox", "RenderSpec", "Degradation",
    "AblationSupport", "script_for",
    # model-free understanding-GT derivation
    "Derivation", "resolve", "locate", "count_occurrences", "region_box", "union_box",
    "word_boxes", "aggregate",
    # executable semantic graphs and leakage-safe splitting
    "DifficultySpec", "GraphEdge", "GraphNode", "GraphQuery", "LatentDocumentGraph",
    "ResolvedQuery", "SplitPolicy", "validate_split_leakage",
    "apply_supervision_toggles",
    "EvidenceQualityError", "audit_render_evidence", "audit_degraded_evidence",
    "collect_evidence_boxes", "redact_evidence_quality_report",
]
