"""Document overlays are deterministic, grounded, and evidence-safe."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from docvlm_eval.synth.dto import DocSample, GenConfig
from docvlm_eval.synth.overlays import (
    OVERLAY_TYPES,
    apply_document_overlays,
    derive_overlay_seed,
)
from docvlm_eval.synth.quality import audit_render_evidence
from docvlm_eval.synth.to_samples import case_to_samples


def _intersects(first: list[int], second: list[int]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _fixture() -> tuple[Image.Image, dict]:
    image = Image.new("RGB", (640, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((220, 300, 420, 340), fill="black")
    ground_truth = {
        "type": "overlay fixture",
        "stressors": [],
        "anchor_metric": "NED",
        "fields": {"target": "visible", "full_text": "Base text"},
        "spotting": {"target": [220, 300, 420, 340]},
        "qa": [
            {
                "question": "Transcribe all text.",
                "answers": ["Base text"],
                "metric": "ned",
                "answer_type": "ocr-full",
                "languages": ["en"],
            }
        ],
        "render": {"dpi": 96, "size_px": [640, 800], "page_count": 1},
    }
    return image, ground_truth


def test_overlay_seed_is_stable_and_variant_specific():
    first = derive_overlay_seed(7, "invoice", "0000", "en")

    assert first == derive_overlay_seed(7, "invoice", "0000", "en")
    assert first != derive_overlay_seed(7, "invoice", "0001", "en")
    assert first != derive_overlay_seed(7, "invoice", "0000", "ko")


def test_all_overlay_types_are_deterministic_grounded_and_evidence_safe():
    image, ground_truth = _fixture()
    first_image, first_gt = apply_document_overlays(
        image,
        ground_truth,
        seed=19,
        probability=1.0,
        overlay_types=OVERLAY_TYPES,
        max_count=3,
    )
    second_image, second_gt = apply_document_overlays(
        image,
        ground_truth,
        seed=19,
        probability=1.0,
        overlay_types=OVERLAY_TYPES,
        max_count=3,
    )

    assert first_image.tobytes() == second_image.tobytes()
    assert first_gt == second_gt
    marks = first_gt["render"]["overlays"]
    assert {mark["kind"] for mark in marks} == set(OVERLAY_TYPES)
    assert len(first_gt["render"]["overlay_fingerprint"]) == 64
    assert all(
        not _intersects(mark["bbox"], ground_truth["spotting"]["target"])
        for mark in marks
    )
    assert first_gt["fields"]["full_text"].endswith(
        "\n" + "\n".join(mark["text"] for mark in marks)
    )
    full_text_query = next(
        query
        for query in first_gt["qa"]
        if query["answer_type"] == "ocr-full"
    )
    assert full_text_query["answers"] == [first_gt["fields"]["full_text"]]

    record = DocSample.from_builder_gt(
        first_gt,
        gen_config=GenConfig(
            overlay_prob=1.0,
            overlay_types=list(OVERLAY_TYPES),
            overlay_max_count=3,
        ),
    ).to_dict()
    report = audit_render_evidence(first_image, record)
    assert report["status"] == "passed"
    assert len(
        [
            query
            for query in record["qa_detailed"]
            if query["answer_type"].startswith("overlay-")
        ]
    ) == 3
    samples = case_to_samples(record, "clean.png", "overlay")
    overlay_samples = [
        sample
        for sample in samples
        if sample.answer_type.startswith("overlay-")
    ]
    assert len(overlay_samples) == 3
    assert all(sample.meta["evidence_count"] == 1 for sample in overlay_samples)
    assert all("box" in sample.meta for sample in overlay_samples)
    assert all(sample.meta["overlay_count"] == 3 for sample in overlay_samples)


def test_overlay_probability_zero_preserves_pixels_and_adds_no_provenance():
    image, ground_truth = _fixture()
    output, result = apply_document_overlays(
        image,
        ground_truth,
        seed=23,
        probability=0.0,
    )

    assert output is image
    assert result == ground_truth
    assert "overlays" not in result["render"]


@pytest.mark.parametrize("kind", OVERLAY_TYPES)
def test_each_overlay_fits_a_short_meter_canvas(kind):
    image = Image.new("RGB", (520, 200), "white")
    ImageDraw.Draw(image).rectangle((145, 70, 375, 130), fill="black")
    ground_truth = {
        "type": "meter",
        "stressors": [],
        "anchor_metric": "exact",
        "fields": {"reading": "123.4"},
        "spotting": {"reading": [145, 70, 375, 130]},
        "qa": [],
        "render": {"dpi": 96, "size_px": [520, 200], "page_count": 1},
    }

    output, result = apply_document_overlays(
        image,
        ground_truth,
        seed=31,
        probability=1.0,
        overlay_types=[kind],
        max_count=1,
    )

    assert [mark["kind"] for mark in result["render"]["overlays"]] == [kind]
    record = DocSample.from_builder_gt(result).to_dict()
    assert audit_render_evidence(output, record)["status"] == "passed"
