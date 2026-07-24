"""Perspective geometry keeps rendered pixels and every spatial target in one frame."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from docvlm_eval.synth.geometry import (
    derive_perspective_seed,
    transform_box,
    transform_ground_truth,
    warp_perspective,
)
from docvlm_eval.synth.quality import audit_render_evidence
from docvlm_eval.synth.overlays import overlay_fingerprint


def test_perspective_seed_is_stable_and_variant_specific():
    first = derive_perspective_seed(7, "id_card", "0000", "en")

    assert first == derive_perspective_seed(7, "id_card", "0000", "en")
    assert first != derive_perspective_seed(7, "id_card", "0001", "en")
    assert first != derive_perspective_seed(7, "id_card", "0000", "ko")


def test_transform_box_applies_homography_and_clips_envelope():
    translation = np.asarray(
        [[1.0, 0.0, 7.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert transform_box([10, 12, 30, 32], translation, width=100, height=80) == [
        17,
        16,
        37,
        36,
    ]
    assert transform_box([90, 60, 100, 80], translation, width=100, height=80) == [
        97,
        64,
        100,
        80,
    ]


def test_transform_ground_truth_updates_all_legacy_spatial_views():
    translation = np.asarray(
        [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    gt = {
        "spotting": {"total": [10, 12, 30, 32]},
        "render": {
            "overlay_fingerprint": "source",
            "overlays": [
                {"kind": "stamp", "text": "APPROVED", "bbox": [60, 40, 80, 55]}
            ]
        },
        "qa": [
            {
                "key": "total",
                "metric": "grounding",
                "derived": True,
                "box": [10, 12, 30, 32],
                "answers": ["10,12,30,32;100,80"],
                "rationale": "The value is inside [10, 12, 30, 32].",
                "evidence_bboxes": [[40, 20, 50, 30]],
            }
        ],
    }

    transformed, count = transform_ground_truth(
        gt,
        translation,
        width=100,
        height=80,
    )

    assert transformed["spotting"]["total"] == [15, 15, 35, 35]
    assert transformed["qa"][0]["box"] == [15, 15, 35, 35]
    assert transformed["qa"][0]["answers"] == ["15,15,35,35;100,80"]
    assert "[15, 15, 35, 35]" in transformed["qa"][0]["rationale"]
    assert transformed["qa"][0]["evidence_bboxes"] == [[45, 23, 55, 33]]
    assert transformed["render"]["overlays"][0]["bbox"] == [65, 43, 85, 58]
    assert transformed["render"]["overlay_fingerprint"] == overlay_fingerprint(
        transformed["render"]["overlays"]
    )
    assert gt["spotting"]["total"] == [10, 12, 30, 32]
    assert count == 3


def test_warp_perspective_is_deterministic_and_preserves_visible_evidence():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((38, 38, 101, 81), fill="black")
    gt = {
        "spotting": {"value": [35, 35, 105, 85]},
        "render": {"size_px": [160, 120]},
    }

    first_image, first_gt = warp_perspective(image, gt, seed=17)
    second_image, second_gt = warp_perspective(image, gt, seed=17)
    other_image, other_gt = warp_perspective(image, gt, seed=18)

    assert first_image.size == image.size
    assert np.array_equal(np.asarray(first_image), np.asarray(second_image))
    assert first_gt == second_gt
    assert not np.array_equal(np.asarray(first_image), np.asarray(other_image))
    assert first_gt["spotting"]["value"] != other_gt["spotting"]["value"]
    geometry = first_gt["render"]["geometry"]
    assert geometry["kind"] == "perspective"
    assert geometry["seed"] == 17
    assert geometry["box_enclosure"] == "axis_aligned_envelope"
    assert geometry["transformed_box_references"] == 1

    report = audit_render_evidence(
        first_image,
        {"spotting": first_gt["spotting"]},
        required_spotting_keys=["value"],
    )
    assert report["status"] == "passed"
