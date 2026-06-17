"""Schema dataclasses: defaults and field integrity."""

from docvlm_eval.schema import Prediction, Sample


def test_sample_defaults():
    s = Sample("id0", "img.png", "q?", ["a"])
    assert s.answer_type == "default"
    assert s.metric == "anls"
    assert s.meta == {}


def test_sample_meta_is_independent():
    a = Sample("a", "i", "q", ["x"])
    b = Sample("b", "i", "q", ["y"])
    a.meta["k"] = 1
    assert b.meta == {}  # default_factory, not shared


def test_prediction_defaults():
    p = Prediction("id0", "answer")
    assert p.confidence is None
    assert p.raw == ""
