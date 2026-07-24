"""Deterministic synthetic split assignment and semantic leakage checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SplitPolicy:
    """Hash-based split policy whose grouping key is explicit and reproducible."""

    seed: int = 7
    train: float = 0.8
    validation: float = 0.1
    heldout: float = 0.1
    group_by: str = "content"

    def __post_init__(self) -> None:
        if self.group_by not in {"content", "template", "layout", "document"}:
            raise ValueError(
                "group_by must be content, template, layout, or document"
            )
        values = (self.train, self.validation, self.heldout)
        if any(value < 0 for value in values):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")

    def assign(self, record: dict[str, Any]) -> str:
        graph = record.get("semantic_graph") or {}
        keys = {
            "content": graph.get("content_fingerprint"),
            "template": graph.get("template_fingerprint"),
            "layout": (record.get("render") or {}).get("layout_fingerprint"),
            "document": record.get("doc_id"),
        }
        group = keys[self.group_by]
        if not group:
            raise ValueError(f"record is missing the {self.group_by} split key")
        digest = hashlib.sha256(f"{self.seed}:{group}".encode("utf-8")).digest()
        point = int.from_bytes(digest[:8], "big") / 2**64
        if point < self.train:
            return "train"
        if point < self.train + self.validation:
            return "validation"
        return "heldout"


def validate_split_leakage(
    records: Iterable[dict[str, Any]],
    *,
    require_template_isolation: bool = False,
    require_layout_isolation: bool = False,
) -> dict[str, Any]:
    """Raise on semantic leakage and report program-template and visual-layout overlap."""

    seen_content: dict[str, str] = {}
    seen_template: dict[str, set[str]] = {}
    seen_layout: dict[str, set[str]] = {}
    missing_layout = 0
    counts = {"train": 0, "validation": 0, "heldout": 0}
    for record in records:
        split = str(record.get("split") or "")
        if split not in counts:
            raise ValueError(f"invalid or missing split {split!r}")
        graph = record.get("semantic_graph") or {}
        content = graph.get("content_fingerprint")
        template = graph.get("template_fingerprint")
        if not content or not template:
            raise ValueError("every split record requires semantic graph fingerprints")
        previous = seen_content.get(content)
        if previous is not None and previous != split:
            raise ValueError(
                f"semantic content leakage: fingerprint {content[:12]} appears in "
                f"{previous} and {split}"
            )
        seen_content[content] = split
        seen_template.setdefault(template, set()).add(split)
        layout = (record.get("render") or {}).get("layout_fingerprint")
        if layout:
            seen_layout.setdefault(str(layout), set()).add(split)
        else:
            missing_layout += 1
        counts[split] += 1

    template_overlaps = {
        fingerprint: sorted(splits)
        for fingerprint, splits in seen_template.items()
        if len(splits) > 1
    }
    layout_overlaps = {
        fingerprint: sorted(splits)
        for fingerprint, splits in seen_layout.items()
        if len(splits) > 1
    }
    if require_template_isolation and template_overlaps:
        first, splits = next(iter(template_overlaps.items()))
        raise ValueError(
            f"template leakage: fingerprint {first[:12]} appears in {', '.join(splits)}"
        )
    if require_layout_isolation and missing_layout:
        raise ValueError(
            f"layout isolation requires layout fingerprints; {missing_layout} record(s) are missing"
        )
    if require_layout_isolation and layout_overlaps:
        first, splits = next(iter(layout_overlaps.items()))
        raise ValueError(
            f"layout leakage: fingerprint {first[:12]} appears in {', '.join(splits)}"
        )
    return {
        "counts": counts,
        "unique_content": len(seen_content),
        "unique_templates": len(seen_template),
        "template_overlap_count": len(template_overlaps),
        "template_overlaps": template_overlaps,
        "unique_layouts": len(seen_layout),
        "missing_layout_fingerprints": missing_layout,
        "layout_overlap_count": len(layout_overlaps),
        "layout_overlaps": layout_overlaps,
    }
