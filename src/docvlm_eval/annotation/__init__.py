"""Layered document annotation (DAR): one record per image carrying every annotation layer —
page/orientation, layout, OCR, tables, KIE, and understanding (caption, reading guide, reasoned
QA) — linked by id. See ``docs/report/annotation_format.md``.
"""

from .ablation import (
    ARMS,
    FAMILIES,
    LayerArm,
    annotation_cost,
    build_arm,
    eval_samples,
    value_per_hour,
)
from .diagnose import attribute, diagnostic_samples, failure_profile
from .export import EXPORT_LAYERS, REASONING_STYLES, export_samples, to_unified, write_jsonl
from .geometry import materialize_rotations, rotate_annotation, rotate_box, to_upright
from .schema import (
    LAYERS,
    SCHEMA_VERSION,
    Cell,
    DocAnnotation,
    GuideNote,
    KIEField,
    LayerProvenance,
    PageInfo,
    Provenance,
    ReasonedQA,
    ReasonStep,
    Region,
    Table,
    TextLine,
    Understanding,
    skeleton,
)
from .validate import Issue, assert_valid, validate

__all__ = [
    "ARMS",
    "EXPORT_LAYERS",
    "FAMILIES",
    "LAYERS",
    "REASONING_STYLES",
    "SCHEMA_VERSION",
    "Cell",
    "DocAnnotation",
    "GuideNote",
    "Issue",
    "KIEField",
    "LayerArm",
    "LayerProvenance",
    "PageInfo",
    "Provenance",
    "ReasonStep",
    "ReasonedQA",
    "Region",
    "Table",
    "TextLine",
    "Understanding",
    "annotation_cost",
    "assert_valid",
    "attribute",
    "build_arm",
    "diagnostic_samples",
    "eval_samples",
    "export_samples",
    "failure_profile",
    "materialize_rotations",
    "rotate_annotation",
    "rotate_box",
    "skeleton",
    "to_unified",
    "to_upright",
    "validate",
    "value_per_hour",
    "write_jsonl",
]
