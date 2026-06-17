"""Aggregation: headline metrics, per-slice breakdown, ECE wiring, missing predictions."""

from docvlm_eval.metrics import aggregate
from docvlm_eval.schema import Prediction, Sample


def _samples():
    return [
        Sample("a", "i", "q", ["hello"], "typeA", "anls"),
        Sample("b", "i", "q", ["100"], "typeB", "relaxed_acc"),
        Sample("c", "i", "q", ["world"], "typeA", "exact"),
    ]


def test_perfect_predictions():
    s = _samples()
    preds = {
        "a": Prediction("a", "hello", 0.9),
        "b": Prediction("b", "100", 0.9),
        "c": Prediction("c", "world", 0.9),
    }
    out = aggregate(s, preds)
    assert out["summary"]["n_samples"] == 3
    assert out["summary"]["score"] == 1.0
    assert out["summary"]["accuracy"] == 1.0
    assert out["summary"]["answer_rate"] == 1.0
    assert out["summary"]["by_answer_type"]["typeA"]["n"] == 2


def test_missing_prediction_scores_zero():
    s = _samples()
    preds = {"a": Prediction("a", "hello", 0.5)}  # b, c missing
    out = aggregate(s, preds)
    assert out["summary"]["score"] < 1.0
    assert out["summary"]["answer_rate"] == round(1 / 3, 4)
    # per-sample rows exist for every sample even if unpredicted
    assert len(out["per_sample"]) == 3


def test_ece_present_when_confidences_given():
    s = _samples()
    preds = {k: Prediction(k, "wrong", 0.9) for k in ["a", "b", "c"]}
    out = aggregate(s, preds)
    assert out["summary"]["ece"] is not None  # confident but wrong -> high ece


def test_ece_none_when_no_confidence():
    s = _samples()
    preds = {k: Prediction(k, "hello", None) for k in ["a", "b", "c"]}
    out = aggregate(s, preds)
    assert out["summary"]["ece"] is None
