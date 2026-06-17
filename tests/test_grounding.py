"""Grounding metric: box parsing (pixel / normalised / loc-tokens) + IoU scoring."""

from docvlm_eval.metrics.grounding import grounding_score, iou, parse_gold_box, parse_pred_box
from docvlm_eval.metrics.text import score_sample

GOLD = ["40,335,270,359;820,600"]


def test_parse_gold_box():
    box, size = parse_gold_box(GOLD[0])
    assert box == [40, 335, 270, 359]
    assert size == (820, 600)


def test_iou_identical():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_exact_box_scores_one():
    assert grounding_score("[40, 335, 270, 359]", GOLD) == 1.0


def test_close_box_partial():
    s = grounding_score("box: [45, 338, 265, 355]", GOLD)
    assert 0.5 < s < 1.0


def test_wrong_box_zero():
    assert grounding_score("[600, 50, 700, 90]", GOLD) == 0.0


def test_no_box_zero():
    assert grounding_score("I cannot determine the location", GOLD) == 0.0


def test_normalised_box_rescaled():
    # 0-1 normalised coords matching the gold should rescale and score high
    s = grounding_score("[0.05, 0.56, 0.33, 0.60]", GOLD)
    assert s > 0.8


def test_dispatch_via_score_sample():
    # the "grounding" metric must be reachable through the generic dispatcher
    assert score_sample("grounding", "[40,335,270,359]", GOLD) == 1.0
