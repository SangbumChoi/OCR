"""Compose independently rendered documents into one exact-provenance visual bundle."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class BundleDocument:
    """One independently rendered document entering a shared visual canvas."""

    document_id: str
    image: Image.Image
    ground_truth: dict[str, Any]
    required_spotting_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id or "." in self.document_id:
            raise ValueError("document_id must be non-empty and cannot contain '.'")
        render_size = (self.ground_truth.get("render") or {}).get("size_px")
        if render_size and list(self.image.size) != list(render_size):
            raise ValueError(
                f"document {self.document_id!r} image size {self.image.size} "
                f"does not match render size {render_size}"
            )


def _offset_box(box: Sequence[Any], origin: tuple[int, int]) -> list[int]:
    if len(box) < 4:
        raise ValueError(f"invalid bundle box: {box!r}")
    ox, oy = origin
    return [
        round(float(box[0])) + ox,
        round(float(box[1])) + oy,
        round(float(box[2])) + ox,
        round(float(box[3])) + oy,
    ]


def _prefix_key(document_id: str, key: str | None) -> str | None:
    if key is None:
        return None
    return f"{document_id}.{key}"


def _placement(
    sizes: Sequence[tuple[int, int]],
    *,
    mode: str,
    gap_px: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if mode not in {"grid", "vertical"}:
        raise ValueError("bundle mode must be grid or vertical")
    if gap_px < 0:
        raise ValueError("bundle gap_px cannot be negative")
    if not sizes:
        raise ValueError("at least one document is required")
    if mode == "vertical":
        width = max(size[0] for size in sizes)
        origins: list[tuple[int, int]] = []
        y = 0
        for width_i, height_i in sizes:
            origins.append(((width - width_i) // 2, y))
            y += height_i + gap_px
        return origins, (width, y - gap_px)

    columns = math.ceil(math.sqrt(len(sizes)))
    rows = math.ceil(len(sizes) / columns)
    column_widths = [
        max(
            (
                sizes[index][0]
                for index in range(column, len(sizes), columns)
            ),
            default=0,
        )
        for column in range(columns)
    ]
    row_heights = [
        max(
            (
                sizes[index][1]
                for index in range(row * columns, min((row + 1) * columns, len(sizes)))
            ),
            default=0,
        )
        for row in range(rows)
    ]
    column_x = [
        sum(column_widths[:column]) + column * gap_px
        for column in range(columns)
    ]
    row_y = [
        sum(row_heights[:row]) + row * gap_px
        for row in range(rows)
    ]
    origins = [
        (
            column_x[index % columns]
            + (column_widths[index % columns] - sizes[index][0]) // 2,
            row_y[index // columns]
            + (row_heights[index // columns] - sizes[index][1]) // 2,
        )
        for index in range(len(sizes))
    ]
    return origins, (
        sum(column_widths) + gap_px * (columns - 1),
        sum(row_heights) + gap_px * (rows - 1),
    )


def _merge_source_qa(
    query: dict[str, Any],
    *,
    document_id: str,
    origin: tuple[int, int],
    canvas_size: tuple[int, int],
) -> dict[str, Any]:
    merged = copy.deepcopy(query)
    merged["key"] = _prefix_key(document_id, merged.get("key"))
    merged["evidence_keys"] = [
        _prefix_key(document_id, key)
        for key in (merged.get("evidence_keys") or [])
    ]
    if merged.get("box"):
        merged["box"] = _offset_box(merged["box"], origin)
        if merged.get("metric") == "grounding" or merged.get("derived"):
            box = merged["box"]
            merged["answers"] = [
                f"{box[0]},{box[1]},{box[2]},{box[3]};"
                f"{canvas_size[0]},{canvas_size[1]}"
            ]
        if merged.get("rationale"):
            box = merged["box"]
            merged["rationale"] = re.sub(
                r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
                f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
                str(merged["rationale"]),
                count=1,
            )
    if merged.get("evidence_bboxes"):
        merged["evidence_bboxes"] = [
            _offset_box(box, origin)
            for box in merged["evidence_bboxes"]
        ]
    return merged


def compose_document_bundle(
    documents: Sequence[BundleDocument],
    *,
    mode: str = "grid",
    gap_px: int = 18,
    background: tuple[int, int, int] = (224, 228, 232),
    doc_type: str = "cross-document bundle",
    stressors: Sequence[str] = ("multi-document", "cross-document-reasoning"),
    anchor_metric: str = "relaxed accuracy + evidence IoU",
    task: str | None = None,
    qa: Sequence[dict[str, Any]] = (),
    probes: Sequence[dict[str, Any]] = (),
    include_source_qa: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """Pack documents and merge their GT into one image-coordinate frame.

    Every source field and spatial key is namespaced as ``<document_id>.<key>``.
    Page provenance remains flattened for existing page-aware consumers, while the
    document arrays preserve the higher-level source identity.
    """
    if not documents:
        raise ValueError("at least one bundle document is required")
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("bundle document IDs must be unique")

    sizes = [document.image.size for document in documents]
    origins, canvas_size = _placement(sizes, mode=mode, gap_px=gap_px)
    canvas = Image.new("RGB", canvas_size, background)
    fields: dict[str, Any] = {}
    spotting: dict[str, list[int]] = {}
    merged_qa: list[dict[str, Any]] = []
    merged_probes: list[dict[str, Any]] = []
    page_origins: list[list[int]] = []
    page_sizes: list[list[int]] = []
    page_document_indices: list[int] = []
    page_document_ids: list[str] = []
    document_records: list[dict[str, Any]] = []
    required_spotting_keys: list[str] = []
    page_cursor = 0

    for document_index, (document, origin) in enumerate(zip(documents, origins)):
        image = document.image.convert("RGB")
        canvas.paste(image, origin)
        source = document.ground_truth
        source_render = source.get("render") or {}
        for key, value in (source.get("fields") or {}).items():
            if str(key).startswith("_"):
                continue
            fields[f"{document.document_id}.{key}"] = value
        for key, box in (source.get("spotting") or {}).items():
            spotting[f"{document.document_id}.{key}"] = _offset_box(box, origin)
        required_spotting_keys.extend(
            f"{document.document_id}.{key}"
            for key in document.required_spotting_keys
        )
        if include_source_qa:
            merged_qa.extend(
                _merge_source_qa(
                    query,
                    document_id=document.document_id,
                    origin=origin,
                    canvas_size=canvas_size,
                )
                for query in (source.get("qa") or [])
            )
        merged_probes.extend(copy.deepcopy(source.get("probes") or []))

        relative_page_origins = source_render.get("page_origins_px") or [[0, 0]]
        relative_page_sizes = source_render.get("page_sizes_px") or [list(image.size)]
        if len(relative_page_origins) != len(relative_page_sizes):
            raise ValueError(
                f"document {document.document_id!r} has inconsistent page provenance"
            )
        first_page = page_cursor
        for page_origin, page_size in zip(relative_page_origins, relative_page_sizes):
            page_origins.append(
                [
                    round(float(page_origin[0])) + origin[0],
                    round(float(page_origin[1])) + origin[1],
                ]
            )
            page_sizes.append(
                [round(float(page_size[0])), round(float(page_size[1]))]
            )
            page_document_indices.append(document_index)
            page_document_ids.append(document.document_id)
            page_cursor += 1
        document_records.append(
            {
                "document_id": document.document_id,
                "document_type": source.get("type") or source.get("doc_type"),
                "origin_px": list(origin),
                "size_px": list(image.size),
                "page_indices": list(range(first_page, page_cursor)),
                "source_page_count": int(source_render.get("page_count") or 1),
                "rendered_page_count": len(relative_page_origins),
            }
        )

    available_keys = set(spotting)
    for query in qa:
        copied = copy.deepcopy(query)
        missing = set(copied.get("evidence_keys") or []) - available_keys
        if missing:
            raise ValueError(
                f"cross-document QA references unknown evidence keys: {sorted(missing)}"
            )
        merged_qa.append(copied)
    merged_probes.extend(copy.deepcopy(list(probes)))
    if task:
        fields["_task"] = task

    source_dpis = {
        (document.ground_truth.get("render") or {}).get("dpi")
        for document in documents
    }
    render = {
        "dpi": source_dpis.pop() if len(source_dpis) == 1 else None,
        "size_px": list(canvas_size),
        "page_size": "document-bundle",
        "page_count": sum(
            int((document.ground_truth.get("render") or {}).get("page_count") or 1)
            for document in documents
        ),
        "rendered_page_count": len(page_origins),
        "page_mode": f"bundle-{mode}",
        "page_gap_px": gap_px,
        "page_origins_px": page_origins,
        "page_sizes_px": page_sizes,
        "page_document_indices": page_document_indices,
        "page_document_ids": page_document_ids,
        "document_count": len(documents),
        "document_mode": mode,
        "document_gap_px": gap_px,
        "document_ids": ids,
        "document_origins_px": [list(origin) for origin in origins],
        "document_sizes_px": [list(size) for size in sizes],
        "documents": document_records,
        "box_resolver": "composed_exact_offsets",
        "required_spotting_keys": sorted(set(required_spotting_keys)),
        "color_probe_fallback_count": sum(
            int(
                (document.ground_truth.get("render") or {}).get(
                    "color_probe_fallback_count",
                    0,
                )
            )
            for document in documents
        ),
    }
    ground_truth: dict[str, Any] = {
        "type": doc_type,
        "stressors": list(stressors),
        "anchor_metric": anchor_metric,
        "fields": fields,
        "render": render,
        "source": (
            "SYNTHETIC (docvlm_eval.synth.bundle) - independently rendered "
            "documents with exact composition offsets"
        ),
    }
    if spotting:
        ground_truth["spotting"] = spotting
    if merged_qa:
        ground_truth["qa"] = merged_qa
    if merged_probes:
        ground_truth["probes"] = merged_probes
    return canvas, ground_truth
