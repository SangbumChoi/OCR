"""Render-aware quality gates for synthetic spatial supervision."""

import pytest
from PIL import Image, ImageDraw

from docvlm_eval.synth.quality import (
    EvidenceQualityError,
    audit_render_evidence,
    collect_evidence_boxes,
)


def _sample(box):
    return {
        "fields_detailed": [{"key": "total", "bbox": box}],
        "qa_detailed": [
            {
                "key": "total",
                "answer_bbox": box,
                "evidence_bboxes": [box],
            }
        ],
        "spotting": {"total": box},
    }


def test_collect_evidence_boxes_deduplicates_structured_and_legacy_views():
    boxes = collect_evidence_boxes(_sample([10, 10, 30, 30]))

    assert list(boxes) == [(10, 10, 30, 30)]
    assert boxes[(10, 10, 30, 30)] == [
        "field:total",
        "qa_answer:total",
        "qa_evidence:total:0",
        "spotting:total",
    ]


def test_render_evidence_audit_accepts_visible_local_contrast():
    image = Image.new("RGB", (80, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 31, 31), fill="black")

    report = audit_render_evidence(
        image,
        _sample([15, 15, 35, 35]),
        required_spotting_keys=["total"],
    )

    assert report["status"] == "passed"
    assert report["unique_boxes"] == 1
    assert report["failure_count"] == 0
    assert report["boxes"][0]["foreground_pixels"] >= 4
    assert report["boxes"][0]["passed"] is True


def test_render_evidence_audit_rejects_visually_blank_box():
    image = Image.new("RGB", (80, 60), "white")

    with pytest.raises(EvidenceQualityError, match="insufficient_visible_evidence"):
        audit_render_evidence(image, _sample([15, 15, 35, 35]))


def test_render_evidence_audit_rejects_clipped_geometry():
    image = Image.new("RGB", (80, 60), "white")

    with pytest.raises(EvidenceQualityError, match="invalid_or_clipped_geometry"):
        audit_render_evidence(image, _sample([15, 15, 85, 35]))


def test_render_evidence_audit_rejects_unresolved_requested_spot():
    image = Image.new("RGB", (80, 60), "white")

    with pytest.raises(EvidenceQualityError, match="unresolved_required_spotting_keys"):
        audit_render_evidence(
            image,
            {"fields_detailed": [], "qa_detailed": []},
            required_spotting_keys=["missing"],
        )


def test_render_evidence_audit_skips_samples_without_spatial_supervision():
    report = audit_render_evidence(
        Image.new("RGB", (20, 20), "white"),
        {"fields_detailed": [], "qa_detailed": []},
    )

    assert report["status"] == "skipped_no_boxes"
    assert report["unique_boxes"] == 0
