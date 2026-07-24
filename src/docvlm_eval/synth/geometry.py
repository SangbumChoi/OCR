"""Deterministic geometric augmentation with exact spatial-supervision transforms."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image


class GeometryAugmentationError(ValueError):
    """Raised when a geometric augmentation cannot preserve a valid coordinate frame."""


def derive_perspective_seed(
    base_seed: int,
    document_key: str,
    variant: str | None,
    language: str,
) -> int:
    """Derive a stable per-document seed for perspective sampling."""
    material = f"{base_seed}:{document_key}:{variant}:{language}:perspective"
    return int(hashlib.md5(material.encode()).hexdigest(), 16) % (2**30)


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous @ homography.T
    denominators = projected[:, 2:3]
    if np.any(np.abs(denominators) <= 1e-12):
        raise GeometryAugmentationError("homography maps a box corner to infinity")
    return projected[:, :2] / denominators


def transform_box(
    box: Sequence[float],
    homography: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[int]:
    """Transform an xyxy box and return its clipped axis-aligned envelope."""
    if len(box) < 4:
        raise GeometryAugmentationError(f"invalid box: {box!r}")
    x1, y1, x2, y2 = (float(box[index]) for index in range(4))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise GeometryAugmentationError(f"box is outside source image: {box!r}")
    points = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )
    transformed = _transform_points(points, homography)
    result = [
        max(0, min(width, int(np.floor(transformed[:, 0].min())))),
        max(0, min(height, int(np.floor(transformed[:, 1].min())))),
        max(0, min(width, int(np.ceil(transformed[:, 0].max())))),
        max(0, min(height, int(np.ceil(transformed[:, 1].max())))),
    ]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise GeometryAugmentationError(
            f"perspective collapsed box {box!r} to {result!r}"
        )
    return result


def _replace_rationale_box(rationale: str, box: Sequence[int]) -> str:
    return re.sub(
        r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
        f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
        rationale,
        count=1,
    )


def transform_ground_truth(
    ground_truth: Mapping[str, Any],
    homography: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[dict[str, Any], int]:
    """Deep-copy GT and transform every supported spatial-supervision view."""
    transformed = copy.deepcopy(dict(ground_truth))
    transformed_count = 0

    spotting = transformed.get("spotting")
    if isinstance(spotting, Mapping):
        transformed["spotting"] = {
            key: transform_box(box, homography, width=width, height=height)
            for key, box in spotting.items()
        }
        transformed_count += len(transformed["spotting"])

    for query in transformed.get("qa") or []:
        if not isinstance(query, dict):
            continue
        if query.get("box"):
            new_box = transform_box(
                query["box"],
                homography,
                width=width,
                height=height,
            )
            query["box"] = new_box
            transformed_count += 1
            if query.get("metric") == "grounding" or query.get("derived"):
                query["answers"] = [
                    f"{new_box[0]},{new_box[1]},{new_box[2]},{new_box[3]};{width},{height}"
                ]
            if query.get("rationale"):
                query["rationale"] = _replace_rationale_box(
                    query["rationale"],
                    new_box,
                )
        if query.get("evidence_bboxes"):
            query["evidence_bboxes"] = [
                transform_box(box, homography, width=width, height=height)
                for box in query["evidence_bboxes"]
            ]
            transformed_count += len(query["evidence_bboxes"])

    for field in transformed.get("fields_detailed") or []:
        if isinstance(field, dict) and field.get("bbox"):
            field["bbox"] = transform_box(
                field["bbox"],
                homography,
                width=width,
                height=height,
            )
            transformed_count += 1

    for query in transformed.get("qa_detailed") or []:
        if not isinstance(query, dict):
            continue
        if query.get("answer_bbox"):
            query["answer_bbox"] = transform_box(
                query["answer_bbox"],
                homography,
                width=width,
                height=height,
            )
            transformed_count += 1
        if query.get("evidence_bboxes"):
            query["evidence_bboxes"] = [
                transform_box(box, homography, width=width, height=height)
                for box in query["evidence_bboxes"]
            ]
            transformed_count += len(query["evidence_bboxes"])

    return transformed, transformed_count


def _sample_destination_corners(
    width: int,
    height: int,
    *,
    rng: np.random.Generator,
    max_inset_fraction: float,
    min_area_ratio: float,
) -> tuple[np.ndarray, int]:
    max_x = (width - 1) * max_inset_fraction
    max_y = (height - 1) * max_inset_fraction
    source_area = float((width - 1) * (height - 1))
    for attempt in range(1, 17):
        destination = np.asarray(
            [
                [rng.uniform(0, max_x), rng.uniform(0, max_y)],
                [width - 1 - rng.uniform(0, max_x), rng.uniform(0, max_y)],
                [
                    width - 1 - rng.uniform(0, max_x),
                    height - 1 - rng.uniform(0, max_y),
                ],
                [rng.uniform(0, max_x), height - 1 - rng.uniform(0, max_y)],
            ],
            dtype=np.float32,
        )
        if _polygon_area(destination) / source_area >= min_area_ratio:
            return destination, attempt
    raise GeometryAugmentationError(
        "could not sample a perspective quadrilateral above the minimum area ratio"
    )


def warp_perspective(
    image: Image.Image,
    ground_truth: Mapping[str, Any],
    *,
    seed: int,
    max_inset_fraction: float = 0.08,
    min_area_ratio: float = 0.70,
) -> tuple[Image.Image, dict[str, Any]]:
    """Warp image and GT into one same-size perspective coordinate frame."""
    if not 0 < max_inset_fraction < 0.5:
        raise ValueError("max_inset_fraction must be within (0, 0.5)")
    if not 0 < min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be within (0, 1]")
    width, height = image.size
    if width < 2 or height < 2:
        raise GeometryAugmentationError("perspective requires an image of at least 2x2 pixels")

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - synth extra guarantees it
        raise GeometryAugmentationError(
            "perspective augmentation requires opencv-python-headless"
        ) from error

    source = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination, sampling_attempts = _sample_destination_corners(
        width,
        height,
        rng=np.random.default_rng(seed),
        max_inset_fraction=max_inset_fraction,
        min_area_ratio=min_area_ratio,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    rgb = np.asarray(image.convert("RGB"))
    corner_pixels = np.concatenate(
        [
            rgb[:8, :8].reshape(-1, 3),
            rgb[:8, -8:].reshape(-1, 3),
            rgb[-8:, :8].reshape(-1, 3),
            rgb[-8:, -8:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_color = tuple(int(value) for value in np.median(corner_pixels, axis=0))
    warped = cv2.warpPerspective(
        rgb,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
    transformed_gt, transformed_count = transform_ground_truth(
        ground_truth,
        homography,
        width=width,
        height=height,
    )
    area_ratio = _polygon_area(destination) / float((width - 1) * (height - 1))
    transformed_gt.setdefault("render", {})["geometry"] = {
        "schema_version": 1,
        "kind": "perspective",
        "seed": seed,
        "max_inset_fraction": max_inset_fraction,
        "min_area_ratio": min_area_ratio,
        "sampled_area_ratio": round(area_ratio, 6),
        "sampling_attempts": sampling_attempts,
        "destination_corners_px": np.round(destination, 4).tolist(),
        "homography": np.round(homography, 10).tolist(),
        "box_enclosure": "axis_aligned_envelope",
        "transformed_box_references": transformed_count,
    }
    return Image.fromarray(warped), transformed_gt
