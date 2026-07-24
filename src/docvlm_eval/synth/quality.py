"""Pixel-level quality gates for rendered spatial supervision."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


class EvidenceQualityError(ValueError):
    """Raised when emitted spatial supervision is not supported by visible pixels."""


def _box_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        return tuple(int(round(float(value[index]))) for index in range(4))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def collect_evidence_boxes(
    sample: Mapping[str, Any],
) -> dict[tuple[int, int, int, int], list[str]]:
    """Collect and deduplicate every emitted answer/evidence box in a sample."""
    boxes: dict[tuple[int, int, int, int], set[str]] = {}

    def add(source: str, value: Any) -> None:
        box = _box_tuple(value)
        if box is not None:
            boxes.setdefault(box, set()).add(source)

    for field in sample.get("fields_detailed") or []:
        if isinstance(field, Mapping):
            add(f"field:{field.get('key', '?')}", field.get("bbox"))

    for index, query in enumerate(sample.get("qa_detailed") or []):
        if not isinstance(query, Mapping):
            continue
        query_id = query.get("graph_query_id") or query.get("key") or index
        add(f"qa_answer:{query_id}", query.get("answer_bbox"))
        for evidence_index, box in enumerate(query.get("evidence_bboxes") or []):
            add(f"qa_evidence:{query_id}:{evidence_index}", box)

    for key, box in (sample.get("spotting") or {}).items():
        add(f"spotting:{key}", box)

    for index, query in enumerate(sample.get("qa") or []):
        if not isinstance(query, Mapping):
            continue
        query_id = query.get("graph_query_id") or query.get("key") or index
        add(f"qa_box:{query_id}", query.get("box"))
        for evidence_index, box in enumerate(query.get("evidence_bboxes") or []):
            add(f"qa_evidence_legacy:{query_id}:{evidence_index}", box)

    return {box: sorted(sources) for box, sources in sorted(boxes.items())}


def _background_pixels(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    pad = max(2, min(12, round(min(width, height) * 0.15)))
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
    surround = image[top:bottom, left:right]
    mask = np.ones(surround.shape[:2], dtype=bool)
    mask[y1 - top:y2 - top, x1 - left:x2 - left] = False
    ring = surround[mask]
    if ring.size:
        return ring.reshape(-1, 3)

    crop = image[y1:y2, x1:x2]
    border = np.concatenate(
        [
            crop[0, :, :],
            crop[-1, :, :],
            crop[:, 0, :],
            crop[:, -1, :],
        ],
        axis=0,
    )
    return border.reshape(-1, 3)


def audit_render_evidence(
    image: Image.Image,
    sample: Mapping[str, Any],
    *,
    required_spotting_keys: Sequence[str] = (),
    min_contrast: float = 8.0,
    min_foreground_fraction: float = 0.002,
    min_foreground_pixels: int = 4,
) -> dict[str, Any]:
    """Validate that every emitted box contains pixels distinct from its local background.

    The local-background comparison supports light, dark, and colored paper. A box passes only
    when it contains both enough foreground pixels and enough robust contrast. Geometry and
    requested-but-unresolved spotting keys are checked in the same fail-closed audit.
    """
    if min_contrast <= 0:
        raise ValueError("min_contrast must be positive")
    if not 0 <= min_foreground_fraction <= 1:
        raise ValueError("min_foreground_fraction must be within [0, 1]")
    if min_foreground_pixels < 1:
        raise ValueError("min_foreground_pixels must be positive")

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    image_height, image_width = rgb.shape[:2]
    evidence_boxes = collect_evidence_boxes(sample)
    spotting = sample.get("spotting") or {}
    missing_keys = sorted(set(required_spotting_keys) - set(spotting))
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    source_kinds: Counter[str] = Counter()

    for box, sources in evidence_boxes.items():
        for source in sources:
            source_kinds[source.split(":", 1)[0]] += 1
        x1, y1, x2, y2 = box
        geometry_ok = (
            0 <= x1 < x2 <= image_width
            and 0 <= y1 < y2 <= image_height
        )
        if not geometry_ok:
            failure = {
                "box": list(box),
                "sources": sources,
                "reason": "invalid_or_clipped_geometry",
            }
            failures.append(failure)
            observations.append(failure)
            continue

        crop = rgb[y1:y2, x1:x2]
        background = np.median(
            _background_pixels(rgb, box),
            axis=0,
        )
        color_distance = np.max(np.abs(crop - background), axis=2)
        foreground_pixels = int(np.count_nonzero(color_distance >= min_contrast))
        area = int(color_distance.size)
        required_pixels = max(
            min_foreground_pixels,
            int(math.ceil(area * min_foreground_fraction)),
        )
        contrast_p95 = float(np.percentile(color_distance, 95))
        foreground_fraction = foreground_pixels / area
        visible = (
            foreground_pixels >= required_pixels
            and contrast_p95 >= min_contrast
        )
        observation = {
            "box": list(box),
            "sources": sources,
            "area_px": area,
            "foreground_pixels": foreground_pixels,
            "foreground_fraction": round(foreground_fraction, 6),
            "contrast_p95": round(contrast_p95, 3),
            "passed": visible,
        }
        observations.append(observation)
        if not visible:
            failures.append(
                {
                    **observation,
                    "reason": "insufficient_visible_evidence",
                    "required_foreground_pixels": required_pixels,
                }
            )

    if missing_keys:
        failures.append(
            {
                "reason": "unresolved_required_spotting_keys",
                "keys": missing_keys,
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if evidence_boxes else "skipped_no_boxes",
        "image": "clean",
        "image_size_px": [image_width, image_height],
        "unique_boxes": len(evidence_boxes),
        "source_references": sum(len(sources) for sources in evidence_boxes.values()),
        "source_kinds": dict(sorted(source_kinds.items())),
        "required_spotting_keys": len(set(required_spotting_keys)),
        "thresholds": {
            "min_contrast": min_contrast,
            "min_foreground_fraction": min_foreground_fraction,
            "min_foreground_pixels": min_foreground_pixels,
        },
        "boxes": observations,
        "failure_count": len(failures),
    }
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        preview = "; ".join(
            str(failure.get("reason", "unknown"))
            + (
                f" {failure['box']}"
                if "box" in failure
                else f" {failure.get('keys', [])}"
            )
            for failure in failures[:5]
        )
        raise EvidenceQualityError(
            f"render evidence audit failed ({len(failures)} issue(s)): {preview}"
        )
    return report
