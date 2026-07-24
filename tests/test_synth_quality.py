"""Render-aware quality gates for synthetic spatial supervision."""

import pytest
from PIL import Image, ImageDraw

from docvlm_eval.synth.quality import (
    EvidenceQualityError,
    audit_degraded_evidence,
    audit_render_evidence,
    collect_evidence_boxes,
    redact_evidence_quality_report,
)
from docvlm_eval.synth.render import audit_html_render, render_html


_TWO_PAGE_HTML = """
<html><body>
  <table><tr><th>Account</th><td>Primary</td></tr></table>
  <div style="break-before: page">Second page text</div>
</body></html>
"""


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


def test_html_render_audit_reports_omitted_pages():
    result = render_html(_TWO_PAGE_HTML, page_mode="first", dpi=72)
    try:
        report = audit_html_render(
            result,
            _TWO_PAGE_HTML,
            require_all_pages=True,
        )
    finally:
        result.close()

    assert report["status"] == "fail"
    assert report["page_count"] == 2
    assert report["rendered_page_count"] == 1
    assert report["omitted_page_count"] == 1


def test_html_render_rejects_canvas_above_pixel_budget():
    with pytest.raises(RuntimeError, match="max_canvas_pixels"):
        render_html(
            "<html><body>Visible page</body></html>",
            page_mode="first",
            dpi=72,
            max_canvas_pixels=1,
        )


def test_quality_report_redaction_removes_spatial_supervision():
    report = {
        "schema_version": 1,
        "status": "passed",
        "image": "clean",
        "image_size_px": [80, 60],
        "thresholds": {"min_contrast": 8.0},
        "unique_boxes": 1,
        "source_references": 4,
        "source_kinds": {"field": 1},
        "boxes": [{"box": [1, 2, 3, 4], "sources": ["field:x"]}],
        "failure_count": 0,
    }

    redacted = redact_evidence_quality_report(report)

    assert redacted["status"] == "passed"
    assert redacted["supervision_redacted"] is True
    assert "boxes" not in redacted
    assert "unique_boxes" not in redacted
    assert "source_references" not in redacted
    assert "source_kinds" not in redacted


def test_degraded_evidence_audit_accepts_photometric_change():
    clean = Image.new("RGB", (80, 60), "white")
    clean_draw = ImageDraw.Draw(clean)
    clean_draw.rectangle((18, 18, 31, 31), fill="black")
    degraded = Image.new("RGB", (80, 60), (210, 210, 210))
    degraded_draw = ImageDraw.Draw(degraded)
    degraded_draw.rectangle((18, 18, 31, 31), fill=(40, 40, 40))

    report = audit_degraded_evidence(
        clean,
        degraded,
        _sample([15, 15, 35, 35]),
    )

    assert report["status"] == "passed"
    assert report["minimum_structure_correlation"] > 0.9
    assert report["boxes"][0]["visible"] is True


def test_degraded_evidence_audit_rejects_visible_unrelated_noise():
    clean = Image.new("RGB", (80, 60), "white")
    clean_draw = ImageDraw.Draw(clean)
    clean_draw.rectangle((18, 18, 31, 31), fill="black")
    rng = __import__("numpy").random.default_rng(7)
    degraded = Image.fromarray(
        rng.integers(0, 256, size=(60, 80, 3), dtype="uint8"),
        mode="RGB",
    )

    with pytest.raises(EvidenceQualityError, match="insufficient_structure_retention"):
        audit_degraded_evidence(
            clean,
            degraded,
            _sample([15, 15, 35, 35]),
        )


def test_degraded_evidence_audit_rejects_geometry_change():
    clean = Image.new("RGB", (80, 60), "white")
    degraded = Image.new("RGB", (79, 60), "white")

    with pytest.raises(EvidenceQualityError, match="image sizes differ"):
        audit_degraded_evidence(clean, degraded, _sample([15, 15, 35, 35]))
