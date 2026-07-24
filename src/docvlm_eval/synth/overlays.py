"""Deterministic document marks with evidence-safe placement and exact provenance."""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OVERLAY_TYPES = ("handwriting", "stamp", "seal")

_OVERLAY_TEXT = {
    "handwriting": "OK",
    "stamp": "APPROVED",
    "seal": "VALID",
}
_OVERLAY_QUESTION = {
    "handwriting": "What does the handwritten margin note say?",
    "stamp": "What word appears in the rectangular document stamp?",
    "seal": "What word appears inside the circular document seal?",
}


@dataclass(frozen=True)
class OverlayMark:
    """One raster mark placed in the document coordinate frame."""

    kind: str
    text: str
    bbox: list[int]
    angle_degrees: float
    opacity: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def overlay_fingerprint(marks: Sequence[Mapping[str, Any]]) -> str:
    """Hash final mark geometry and style for corpus-level visual deduplication."""
    material = "|".join(
        f"{mark.get('kind')}:{mark.get('text')}:"
        f"{','.join(map(str, mark.get('bbox') or []))}:"
        f"{mark.get('angle_degrees')}:{mark.get('opacity')}"
        for mark in marks
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def derive_overlay_seed(
    base_seed: int,
    document_key: str,
    variant: str | None,
    language: str,
) -> int:
    """Derive a stable seed; callers may pass a pair ID to share overlay style."""
    material = f"{base_seed}:{document_key}:{variant}:{language}:document-overlay"
    return int(hashlib.md5(material.encode()).hexdigest(), 16) % (2**30)


def _box(value: Any) -> tuple[int, int, int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 4
    ):
        return None
    try:
        result = tuple(int(round(float(value[index]))) for index in range(4))
    except (TypeError, ValueError):
        return None
    return result if result[0] < result[2] and result[1] < result[3] else None


def _protected_boxes(
    ground_truth: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []

    def add(value: Any) -> None:
        parsed = _box(value)
        if parsed is None:
            return
        pad = max(3, round(min(width, height) * 0.006))
        boxes.append(
            (
                max(0, parsed[0] - pad),
                max(0, parsed[1] - pad),
                min(width, parsed[2] + pad),
                min(height, parsed[3] + pad),
            )
        )

    for value in (ground_truth.get("spotting") or {}).values():
        add(value)
    for query in ground_truth.get("qa") or []:
        if not isinstance(query, Mapping):
            continue
        add(query.get("box"))
        for value in query.get("evidence_bboxes") or []:
            add(value)
    return boxes


def _intersects(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _font(size: int, *, italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf")
        if italic
        else ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_metrics(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top, left, top


def _make_handwriting(
    text: str,
    base: int,
    rng: random.Random,
) -> tuple[Image.Image, float, int]:
    size = max(12, round(base * 0.031))
    font = _font(size, italic=True)
    text_width, text_height, left, top = _text_metrics(text, font)
    pad = max(5, round(size * 0.35))
    patch = Image.new(
        "RGBA",
        (text_width + 2 * pad, text_height + 2 * pad),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(patch)
    opacity = rng.randint(175, 225)
    draw.text(
        (pad - left, pad - top),
        text,
        font=font,
        fill=(25, 48, 105, opacity),
    )
    y = min(patch.height - 1, pad + text_height + 1)
    draw.line((pad, y, pad + text_width, y), fill=(25, 48, 105, opacity - 35))
    angle = rng.uniform(-11.0, 8.0)
    return (
        patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True),
        angle,
        opacity,
    )


def _make_stamp(
    text: str,
    base: int,
    rng: random.Random,
) -> tuple[Image.Image, float, int]:
    size = max(10, round(base * 0.022))
    font = _font(size)
    text_width, text_height, left, top = _text_metrics(text, font)
    pad_x, pad_y = max(7, size // 2), max(5, size // 3)
    width, height = text_width + 2 * pad_x, text_height + 2 * pad_y
    patch = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    opacity = rng.randint(120, 175)
    color = (165, 24, 31, opacity)
    line_width = max(2, round(base * 0.003))
    draw.rectangle(
        (line_width, line_width, width - line_width - 1, height - line_width - 1),
        outline=color,
        width=line_width,
    )
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=color)
    angle = rng.uniform(-9.0, 9.0)
    return (
        patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True),
        angle,
        opacity,
    )


def _make_seal(
    text: str,
    base: int,
    rng: random.Random,
) -> tuple[Image.Image, float, int]:
    diameter = max(54, round(base * 0.13))
    patch = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    opacity = rng.randint(115, 165)
    color = (150, 25, 35, opacity)
    line_width = max(2, round(base * 0.003))
    inset = line_width
    draw.ellipse(
        (inset, inset, diameter - inset - 1, diameter - inset - 1),
        outline=color,
        width=line_width,
    )
    inner = max(5, round(diameter * 0.10))
    draw.ellipse(
        (inner, inner, diameter - inner - 1, diameter - inner - 1),
        outline=color,
        width=max(1, line_width - 1),
    )
    font = _font(max(9, round(diameter * 0.16)))
    text_width, text_height, left, top = _text_metrics(text, font)
    draw.text(
        (
            (diameter - text_width) / 2 - left,
            (diameter - text_height) / 2 - top,
        ),
        text,
        font=font,
        fill=color,
    )
    angle = rng.uniform(-7.0, 7.0)
    return (
        patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True),
        angle,
        opacity,
    )


def _make_mark(
    kind: str,
    base: int,
    rng: random.Random,
) -> tuple[Image.Image, str, float, int]:
    text = _OVERLAY_TEXT[kind]
    maker = {
        "handwriting": _make_handwriting,
        "stamp": _make_stamp,
        "seal": _make_seal,
    }[kind]
    patch, angle, opacity = maker(text, base, rng)
    return patch, text, angle, opacity


def _ink_fraction(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2].astype(np.float32)
    if crop.size == 0:
        return 1.0
    median = float(np.median(crop))
    return float(np.mean(np.abs(crop - median) >= 22.0))


def _place_mark(
    image: Image.Image,
    patch: Image.Image,
    protected: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[Image.Image, list[int]] | None:
    width, height = image.size
    patch_width, patch_height = patch.size
    margin = max(2, round(min(width, height) * 0.015))
    if patch_width + 2 * margin >= width or patch_height + 2 * margin >= height:
        return None

    gray = np.asarray(image.convert("L"))
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    fixed = [
        (margin, margin),
        (width - patch_width - margin, margin),
        (margin, height - patch_height - margin),
        (width - patch_width - margin, height - patch_height - margin),
    ]
    random_positions = [
        (
            rng.randint(margin, width - patch_width - margin),
            rng.randint(margin, height - patch_height - margin),
        )
        for _ in range(96)
    ]
    for x, y in [*fixed, *random_positions]:
        box = (x, y, x + patch_width, y + patch_height)
        if any(_intersects(box, occupied) for occupied in protected):
            continue
        candidates.append((_ink_fraction(gray, box), box))
    if not candidates:
        return None

    best_score = min(score for score, _ in candidates)
    best = [box for score, box in candidates if score <= best_score + 1e-9]
    x1, y1, x2, y2 = rng.choice(best)
    output = image.convert("RGBA")
    output.alpha_composite(patch, (x1, y1))
    return output.convert("RGB"), [x1, y1, x2, y2]


def apply_document_overlays(
    image: Image.Image,
    ground_truth: Mapping[str, Any],
    *,
    seed: int,
    probability: float,
    overlay_types: Sequence[str] = OVERLAY_TYPES,
    max_count: int = 2,
    language: str = "en",
) -> tuple[Image.Image, dict[str, Any]]:
    """Place document marks without intersecting authored spatial evidence.

    Applied marks become explicit recognition QAs. Their boxes are transformed by later geometric
    augmentation and audited on both clean and degraded pixels.
    """
    if not 0 <= probability <= 1:
        raise ValueError("overlay probability must be within [0, 1]")
    if max_count < 1:
        raise ValueError("overlay max_count must be positive")
    unknown = sorted(set(overlay_types) - set(OVERLAY_TYPES))
    if unknown:
        raise ValueError(f"unknown document overlay types: {unknown}")
    if not overlay_types:
        raise ValueError("overlay_types cannot be empty")

    rng = random.Random(seed)
    output_gt = copy.deepcopy(dict(ground_truth))
    if rng.random() >= probability:
        return image, output_gt

    count = rng.randint(1, min(max_count, len(overlay_types)))
    selected = rng.sample(list(overlay_types), count)
    width, height = image.size
    protected = _protected_boxes(output_gt, width=width, height=height)
    output_image = image
    marks: list[OverlayMark] = []
    for kind in selected:
        patch, text, angle, opacity = _make_mark(kind, min(width, height), rng)
        placed = _place_mark(output_image, patch, protected, rng)
        if placed is None:
            continue
        output_image, bbox = placed
        protected.append(tuple(bbox))
        mark = OverlayMark(kind, text, bbox, round(angle, 3), opacity)
        marks.append(mark)
        output_gt.setdefault("qa", []).append(
            {
                "question": _OVERLAY_QUESTION[kind],
                "answers": [text],
                "metric": "ned",
                "answer_type": f"overlay-{kind}",
                "box": bbox,
                "languages": [language],
            }
        )
        stressor = f"overlay({kind})"
        if stressor not in output_gt.setdefault("stressors", []):
            output_gt["stressors"].append(stressor)

    if marks:
        render = output_gt.setdefault("render", {})
        render["overlay_seed"] = seed
        render["overlays"] = [mark.to_dict() for mark in marks]
        render["overlay_fingerprint"] = overlay_fingerprint(render["overlays"])
        fields = output_gt.get("fields")
        if isinstance(fields, dict) and fields.get("full_text"):
            fields["full_text"] = (
                f"{fields['full_text']}\n"
                + "\n".join(mark.text for mark in marks)
            )
            for query in output_gt.get("qa") or []:
                if (
                    isinstance(query, dict)
                    and query.get("answer_type") == "ocr-full"
                ):
                    query["answers"] = [fields["full_text"]]
    return output_image, output_gt
