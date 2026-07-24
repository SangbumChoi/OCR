"""Exact programmatic process diagrams with executable path reasoning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .hard_layout import HARD_LAYOUT_FAMILIES, layout_fingerprint
from .latent import (
    DifficultySpec,
    GraphEdge,
    GraphNode,
    GraphQuery,
    LatentDocumentGraph,
)


_FONT_CANDIDATES = {
    False: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
    True: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES[bold]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> list[int]:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = round(center[0] - width / 2)
    y = round(center[1] - height / 2 - box[1])
    draw.text((x, y), text, font=font, fill=fill)
    return [x, y + box[1], x + width, y + box[3]]


def _draw_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    label: str,
    *,
    width: int,
    height: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> tuple[list[int], list[int]]:
    x1 = center[0] - width // 2
    y1 = center[1] - height // 2
    x2 = center[0] + width // 2
    y2 = center[1] + height // 2
    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=8,
        fill=fill,
        outline=outline,
        width=3,
    )
    text_box = _centered_text(
        draw,
        center,
        label,
        font=font,
        fill=text_fill,
    )
    return [x1, y1, x2, y2], text_box


def _edge_points(
    source: tuple[int, int],
    target: tuple[int, int],
    *,
    node_width: int,
    node_height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    if abs(dx) >= abs(dy):
        direction = 1 if dx >= 0 else -1
        return (
            (source[0] + direction * node_width // 2, source[1]),
            (target[0] - direction * node_width // 2, target[1]),
        )
    direction = 1 if dy >= 0 else -1
    return (
        (source[0], source[1] + direction * node_height // 2),
        (target[0], target[1] - direction * node_height // 2),
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    source: tuple[int, int],
    target: tuple[int, int],
    *,
    node_width: int,
    node_height: int,
    color: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    start, end = _edge_points(
        source,
        target,
        node_width=node_width,
        node_height=node_height,
    )
    draw.line((start, end), fill=color, width=4)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    arrow = [
        end,
        (
            round(end[0] - ux * 15 + px * 7),
            round(end[1] - uy * 15 + py * 7),
        ),
        (
            round(end[0] - ux * 15 - px * 7),
            round(end[1] - uy * 15 - py * 7),
        ),
    ]
    draw.polygon(arrow, fill=color)
    return start, end


def _draw_edge_label(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> list[int]:
    center = (
        round((start[0] + end[0]) / 2),
        round((start[1] + end[1]) / 2),
    )
    raw = draw.textbbox((0, 0), text, font=font)
    width = raw[2] - raw[0]
    height = raw[3] - raw[1]
    box = [
        center[0] - width // 2 - 7,
        center[1] - height // 2 - 5,
        center[0] + (width + 1) // 2 + 7,
        center[1] + (height + 1) // 2 + 5,
    ]
    draw.rounded_rectangle(
        box,
        radius=5,
        fill=(255, 255, 255),
        outline=(77, 91, 105),
        width=2,
    )
    _centered_text(
        draw,
        center,
        text,
        font=font,
        fill=(25, 32, 38),
    )
    return box


def _layout(
    family: str,
) -> tuple[tuple[int, int], dict[str, tuple[int, int]], tuple[int, int, int, int]]:
    if family == "classic-v1":
        return (
            (1200, 760),
            {
                "input": (110, 340),
                "gate": (320, 340),
                "assay_a": (565, 205),
                "assay_b": (565, 475),
                "fusion": (820, 340),
                "decision": (1070, 340),
            },
            (70, 625, 530, 710),
        )
    if family == "compact-v1":
        return (
            (900, 900),
            {
                "input": (450, 145),
                "gate": (450, 300),
                "assay_a": (250, 470),
                "assay_b": (650, 470),
                "fusion": (450, 645),
                "decision": (450, 805),
            },
            (70, 735, 350, 850),
        )
    if family == "report-v1":
        return (
            (1200, 900),
            {
                "input": (170, 210),
                "gate": (170, 420),
                "assay_a": (470, 300),
                "assay_b": (470, 545),
                "fusion": (780, 420),
                "decision": (1060, 420),
            },
            (70, 720, 575, 830),
        )
    raise ValueError(
        f"unknown hard diagram layout {family!r}; choose from {HARD_LAYOUT_FAMILIES}"
    )


def _query_records(graph: LatentDocumentGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query in graph.queries:
        resolved = graph.resolve(query.query_id)
        records.append(
            {
                "question": query.question + " Answer concisely, no explanation.",
                "answers": [resolved.answer],
                "metric": query.metric,
                "answer_type": query.answer_type,
                "rationale": resolved.rationale,
                "evidence_keys": list(resolved.evidence_keys),
                "graph_query_id": query.query_id,
                "languages": ["en"],
            }
        )
    return records


def hard_process_diagram(
    rng: random.Random,
    *,
    level: int = 4,
    layout_family: str = "classic-v1",
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one directed parallel-path process with exact executable labels."""
    if not 1 <= level <= 5:
        raise ValueError("diagram difficulty level must be within [1, 5]")
    size, centers, audit_box = _layout(layout_family)
    weights = {
        "input_gate": rng.choice((0.88, 0.90, 0.92, 0.94, 0.96)),
        "gate_a": rng.choice((0.52, 0.55, 0.58, 0.60)),
        "a_fusion": rng.choice((0.76, 0.80, 0.84, 0.88)),
        "b_fusion": rng.choice((0.64, 0.68, 0.72, 0.76)),
        "fusion_decision": rng.choice((0.91, 0.93, 0.95, 0.97)),
    }
    weights["gate_b"] = round(1.0 - weights["gate_a"], 2)
    batch_size = rng.choice((800, 1000, 1200, 1500, 2000))
    path_a = (
        weights["input_gate"]
        * weights["gate_a"]
        * weights["a_fusion"]
        * weights["fusion_decision"]
    )
    path_b = (
        weights["input_gate"]
        * weights["gate_b"]
        * weights["b_fusion"]
        * weights["fusion_decision"]
    )
    combined = path_a + path_b
    stage_labels = {
        "input": "Sample Intake",
        "gate": "Quality Gate",
        "assay_a": "Assay A",
        "assay_b": "Assay B",
        "fusion": "Fusion Review",
        "decision": "Release Decision",
    }
    edge_specs = (
        ("input_gate", "input", "gate"),
        ("gate_a", "gate", "assay_a"),
        ("gate_b", "gate", "assay_b"),
        ("a_fusion", "assay_a", "fusion"),
        ("b_fusion", "assay_b", "fusion"),
        ("fusion_decision", "fusion", "decision"),
    )
    edges = [
        GraphEdge(edge_id, source, "flows_to", target, weights[edge_id])
        for edge_id, source, target in edge_specs
    ]
    nodes = [
        GraphNode(
            stage_id,
            "process-stage",
            label,
            label,
            attributes={"field_key": f"stage_{stage_id}"},
        )
        for stage_id, label in stage_labels.items()
    ]
    nodes.extend(
        GraphNode(
            f"label_{edge_id}",
            "edge-label",
            weights[edge_id] * 100,
            f"{source} to {target} flow",
            "percent",
            {"field_key": f"edge_{edge_id}"},
        )
        for edge_id, source, target in edge_specs
    )
    nodes.extend(
        [
            GraphNode(
                "path_a_percent",
                "latent-path",
                path_a * 100,
                "Assay A end-to-end yield",
                "percent",
            ),
            GraphNode(
                "path_b_percent",
                "latent-path",
                path_b * 100,
                "Assay B end-to-end yield",
                "percent",
            ),
            GraphNode(
                "batch_size",
                "audit-field",
                batch_size,
                "submitted batch size",
                "samples",
                {"field_key": "batch_size"},
            ),
        ]
    )
    all_edge_evidence = tuple(f"label_{edge_id}" for edge_id, _, _ in edge_specs)
    path_a_evidence = (
        "label_input_gate",
        "label_gate_a",
        "label_a_fusion",
        "label_fusion_decision",
    )
    queries = [
        GraphQuery(
            "gate_pass_rate",
            "What percentage of samples pass the Quality Gate?",
            "value",
            ("label_input_gate",),
            "T-diagram-edge-read",
            answer_format="percent",
        ),
        GraphQuery(
            "merge_stage",
            "Which stage merges the Assay A and Assay B branches?",
            "value",
            ("fusion",),
            "T-diagram-topology",
            metric="anls",
            answer_format="text",
        ),
        GraphQuery(
            "assay_a_path_yield",
            "What is the end-to-end yield through the Assay A branch?",
            "path_product",
            ("input_gate", "gate_a", "a_fusion", "fusion_decision"),
            "H-diagram-path",
            evidence=path_a_evidence,
            answer_format="fraction_percent",
        ),
        GraphQuery(
            "combined_release_yield",
            "What is the combined release yield across both branches?",
            "sum_products",
            (),
            "H-diagram-parallel-paths",
            evidence=all_edge_evidence,
            parameters={
                "paths": [
                    ["input_gate", "gate_a", "a_fusion", "fusion_decision"],
                    ["input_gate", "gate_b", "b_fusion", "fusion_decision"],
                ]
            },
            answer_format="fraction_percent",
        ),
        GraphQuery(
            "branch_yield_gap",
            "By how many percentage points does the Assay A path exceed the Assay B path?",
            "difference",
            ("path_a_percent", "path_b_percent"),
            "H-diagram-path-comparison",
            evidence=all_edge_evidence,
            answer_format="decimal:2",
        ),
        GraphQuery(
            "expected_releases",
            "For the submitted batch, how many released outputs are expected?",
            "weighted_sum",
            ("batch_size",),
            "H-diagram-expected-count",
            evidence=("batch_size",) + all_edge_evidence,
            parameters={"weights": [combined]},
            answer_format="decimal:2",
        ),
    ]
    active_queries = [
        queries[0],
        *([queries[1]] if level >= 2 else []),
        *([queries[2]] if level >= 3 else []),
        *([queries[3], queries[4]] if level >= 4 else []),
        *([queries[5]] if level >= 5 else []),
    ]
    graph = LatentDocumentGraph(
        graph_id=f"hard-process-diagram-{rng.randrange(1_000_000_000)}",
        template_family="hard-parallel-process-diagram-v1",
        nodes=nodes,
        edges=edges,
        queries=active_queries,
        metadata={
            "layout_family": layout_family,
            "path_count": 2,
            "merge_stage": "fusion",
        },
        language="en",
    )

    image = Image.new("RGB", size, (247, 249, 251))
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    subtitle_font = _font(17)
    node_font = _font(18, bold=True)
    label_font = _font(15, bold=True)
    small_font = _font(14)
    draw.text(
        (55, 32),
        "PARALLEL ASSAY RELEASE WORKFLOW",
        font=title_font,
        fill=(25, 49, 72),
    )
    draw.text(
        (57, 76),
        "Directed edge labels are conditional retention or routing rates.",
        font=subtitle_font,
        fill=(68, 79, 90),
    )
    node_width = 160
    node_height = 70
    edge_geometry: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for edge_id, source, target in edge_specs:
        edge_geometry[edge_id] = _draw_arrow(
            draw,
            centers[source],
            centers[target],
            node_width=node_width,
            node_height=node_height,
            color=(63, 86, 108),
        )

    spotting: dict[str, list[int]] = {}
    fields: dict[str, Any] = {}
    for stage_id, label in stage_labels.items():
        fill = (224, 239, 248)
        outline = (47, 104, 143)
        if stage_id in {"fusion", "decision"}:
            fill = (232, 242, 226)
            outline = (67, 119, 71)
        _, text_box = _draw_node(
            draw,
            centers[stage_id],
            label,
            width=node_width,
            height=node_height,
            fill=fill,
            outline=outline,
            text_fill=(24, 38, 49),
            font=node_font,
        )
        key = f"stage_{stage_id}"
        fields[key] = label
        spotting[key] = text_box
    for edge_id, _, _ in edge_specs:
        label = f"{weights[edge_id] * 100:.0f}%"
        key = f"edge_{edge_id}"
        fields[key] = label
        spotting[key] = _draw_edge_label(
            draw,
            *edge_geometry[edge_id],
            label,
            font=label_font,
        )

    draw.rounded_rectangle(
        audit_box,
        radius=8,
        fill=(255, 255, 255),
        outline=(119, 130, 139),
        width=2,
    )
    draw.text(
        (audit_box[0] + 16, audit_box[1] + 12),
        "RUN AUDIT",
        font=label_font,
        fill=(56, 69, 80),
    )
    batch_label = f"Submitted batch: {batch_size:,} samples"
    batch_origin = (audit_box[0] + 16, audit_box[1] + 45)
    draw.text(batch_origin, batch_label, font=small_font, fill=(25, 32, 38))
    batch_prefix = "Submitted batch: "
    prefix_width = draw.textlength(batch_prefix, font=small_font)
    number_text = f"{batch_size:,}"
    number_box = draw.textbbox(
        (batch_origin[0] + prefix_width, batch_origin[1]),
        number_text,
        font=small_font,
    )
    fields["batch_size"] = str(batch_size)
    spotting["batch_size"] = list(number_box)

    distractor_count = max(0, level - 3)
    distractor_labels = ("Calibration Log", "Archive Copy")
    for index in range(distractor_count):
        x = size[0] - 225
        y = size[1] - 115 + index * 34
        draw.rounded_rectangle(
            (x, y, x + 160, y + 25),
            radius=4,
            fill=(235, 237, 239),
            outline=(167, 172, 177),
        )
        draw.text(
            (x + 9, y + 5),
            distractor_labels[index],
            font=_font(12),
            fill=(100, 105, 110),
        )

    full_text = "\n".join(
        [
            "PARALLEL ASSAY RELEASE WORKFLOW",
            "Directed edge labels are conditional retention or routing rates.",
            *stage_labels.values(),
            *(f"{weights[edge_id] * 100:.0f}%" for edge_id, _, _ in edge_specs),
            "RUN AUDIT",
            batch_label,
            *distractor_labels[:distractor_count],
        ]
    )
    fields["full_text"] = full_text
    qa = _query_records(graph)
    qa.append(
        {
            "key": "full_text",
            "question": "Transcribe all text in this diagram in reading order.",
            "answers": [full_text],
            "metric": "ned",
            "answer_type": "ocr-full",
            "languages": ["en"],
        }
    )
    difficulty = DifficultySpec(
        level=level,
        reasoning_hops=min(level, 4),
        distractor_count=distractor_count,
        visual_density=min(1.0, 0.40 + level * 0.10),
        cross_region=level >= 4,
        skills=(
            "diagram-topology",
            "edge-reading",
            "path-product",
            "parallel-path-aggregation",
            "expected-count",
        ),
    )
    render = {
        "dpi": None,
        "size_px": list(size),
        "page_size": "programmatic-diagram",
        "page_count": 1,
        "rendered_page_count": 1,
        "page_mode": "first",
        "page_gap_px": 0,
        "page_origins_px": [[0, 0]],
        "page_sizes_px": [list(size)],
        "document_count": 1,
        "document_mode": "single",
        "document_ids": ["diagram"],
        "document_origins_px": [[0, 0]],
        "document_sizes_px": [list(size)],
        "layout_family": layout_family,
        "layout_fingerprint": layout_fingerprint("hard_diagram", layout_family),
        "box_resolver": "programmatic_exact",
        "required_spotting_keys": sorted(spotting),
        "color_probe_fallback_count": 0,
    }
    ground_truth = {
        "type": "parallel scientific process diagram",
        "stressors": [
            "directed-diagram",
            "parallel-paths",
            "small-edge-labels",
            "topology",
            "multi-step-probability",
        ],
        "anchor_metric": "relaxed accuracy + evidence IoU",
        "fields": fields,
        "spotting": spotting,
        "qa": qa,
        "probes": [
            {
                "kind": "abstain",
                "question": "What instrument serial number was used?",
                "expected": "not present - abstain",
            }
        ],
        "render": render,
        "semantic_graph": graph.to_dict(),
        "difficulty": difficulty.to_dict(),
        "source": (
            "SYNTHETIC (docvlm_eval.synth.hard_diagram) - exact PIL geometry "
            "and executable path programs"
        ),
    }
    return image, ground_truth
