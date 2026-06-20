"""Model-free understanding-GT derivation (docvlm_eval.synth.derive).

Pure geometry/arithmetic, so it runs without the [synth] render extra: render-dependent derivers
are exercised through a tiny fake RenderResult exposing ``image.size`` + ``search_boxes``.
"""

import pytest

from docvlm_eval.synth.derive import (
    Derivation, aggregate, count_occurrences, locate, region_box, resolve, union_box, word_boxes,
)


class _FakeImg:
    def __init__(self, size):
        self.size = size


class _FakeRR:
    """Stands in for RenderResult: maps a string to its pixel boxes (reading order)."""

    def __init__(self, boxes_by_text, size=(800, 600)):
        self._b = boxes_by_text
        self.image = _FakeImg(size)

    def search_boxes(self, text):
        return [list(b) for b in self._b.get(text, [])]


RR = _FakeRR({
    "TOTAL": [[40, 500, 140, 525]],
    "$": [[200, 100, 210, 120], [300, 100, 310, 120], [40, 500, 50, 520]],
    "Item": [[40, 180, 90, 200]],
    "Amount": [[520, 180, 600, 200]],
    "Widget": [[40, 220, 110, 240]],
})


# ---------------------------------------------------------------- arithmetic (pure)
def test_aggregate_sum_with_working_rationale():
    r, rat = aggregate([45, 80, 20.5], "sum")
    assert r == 145.5 and rat == "45 + 80 + 20.5 = 145.5"


def test_aggregate_ops():
    assert aggregate([3, 7, 5], "max")[0] == 7
    assert aggregate([3, 7, 5], "min")[0] == 3
    assert aggregate([2, 4], "mean")[0] == 3
    assert aggregate([2, 4, 9], "count")[0] == 3


def test_aggregate_rejects_unknown_op_and_empty():
    with pytest.raises(ValueError):
        aggregate([1, 2], "median")
    with pytest.raises(ValueError):
        aggregate([], "sum")


def test_union_box():
    from docvlm_eval.synth.dto import BBox
    u = union_box([BBox(10, 20, 30, 40), BBox(5, 25, 50, 35), None])
    assert u.to_list() == [5, 20, 50, 40]
    assert union_box([None]) is None


# ---------------------------------------------------------------- spatial (fake render)
def test_word_boxes_and_count():
    assert len(word_boxes(RR, "$")) == 3
    n, boxes = count_occurrences(RR, "$")
    assert n == 3 and len(boxes) == 3
    assert count_occurrences(RR, "absent")[0] == 0


def test_locate_occurrence():
    assert locate(RR, "$").to_list() == [200, 100, 210, 120]      # first hit
    assert locate(RR, "$", occurrence=2).to_list() == [40, 500, 50, 520]
    assert locate(RR, "$", occurrence=9) is None
    assert locate(RR, "absent") is None


def test_region_box_unions_member_strings():
    box = region_box(RR, ["Item", "Amount", "Widget"])
    assert box.to_list() == [40, 180, 600, 240]                   # encloses header + cell


# ---------------------------------------------------------------- resolve -> qa dicts
def test_resolve_locate_emits_grounding_qa_with_box_and_rationale():
    qa = resolve(RR, Derivation("locate", text="TOTAL", label="the TOTAL row"))
    assert qa["metric"] == "grounding" and qa["derived"]
    assert qa["answers"] == ["40,500,140,525;800,600"]
    assert qa["box"] == [40, 500, 140, 525]
    assert "[40, 500, 140, 525]" in qa["rationale"] and "800x600px" in qa["rationale"]


def test_resolve_count_emits_exact_qa():
    qa = resolve(RR, Derivation("count", text="$"))
    assert qa["metric"] == "exact" and qa["answers"] == ["3"]
    assert "3 time(s)" in qa["rationale"]


def test_resolve_region_emits_bbox():
    qa = resolve(RR, Derivation("region", texts=["Item", "Amount"], label="the table"))
    assert qa["metric"] == "grounding"
    assert qa["answers"] == ["40,180,600,200;800,600"]


def test_resolve_aggregate_emits_number_and_working():
    qa = resolve(RR, Derivation("aggregate", values=[45, 80, 20.5], op="sum",
                                label="the total amount"))
    assert qa["metric"] == "relaxed_acc"
    assert qa["answers"][0] == "145.5"
    assert "45 + 80 + 20.5 = 145.5" in qa["rationale"]


def test_resolve_returns_none_when_text_absent():
    assert resolve(RR, Derivation("locate", text="nope")) is None
    assert resolve(RR, Derivation("region", texts=["nope"])) is None
