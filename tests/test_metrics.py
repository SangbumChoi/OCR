"""Unit tests for the scoring functions.

These validate the metric implementations against hand-computed values so that the numbers
the pipeline produces are trustworthy (the metrics are the part that must be exactly right,
even though we can't run the GPU models here).

Run: python -m pytest tests/ -q   (or python tests/test_metrics.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.metrics import (  # noqa: E402
    anls,
    exact_match,
    expected_calibration_error,
    ocrbench_score,
    relaxed_accuracy,
)
from docvlm_eval.metrics.text import levenshtein, normalize_text  # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_anls_exact_and_threshold():
    assert approx(anls("hello", ["hello"]), 1.0)
    # one char off out of 5 -> NLS 0.8 (>=0.5 threshold)
    assert approx(anls("hella", ["hello"]), 0.8)
    # very different -> below 0.5 -> scored 0
    assert anls("zzzzz", ["hello"]) == 0.0
    # best over multiple golds
    assert approx(anls("cat", ["dog", "cat"]), 1.0)


def test_relaxed_accuracy_numeric_tolerance():
    assert relaxed_accuracy("102", ["100"]) == 1.0  # within 5%
    assert relaxed_accuracy("120", ["100"]) == 0.0  # outside 5%
    assert relaxed_accuracy("$1,200", ["1200"]) == 1.0
    assert relaxed_accuracy("yes", ["Yes"]) == 1.0  # non-numeric exact


def test_exact_and_ocrbench():
    assert exact_match("The Total", ["total"]) == 1.0
    assert ocrbench_score("the invoice total is 1200 dollars", ["1200"]) == 1.0
    assert ocrbench_score("nothing here", ["1200"]) == 0.0


def test_normalize_text():
    assert normalize_text("The Total: $1,200.") == "total 1200"  # articles/$/commas/trailing-dot stripped


def test_ece_perfectly_calibrated():
    # confidence == empirical accuracy in every bin -> ECE 0
    conf = [0.95, 0.95, 0.05, 0.05]
    corr = [1.0, 1.0, 0.0, 0.0]
    ece = expected_calibration_error(conf, corr, n_bins=10)
    assert ece is not None and ece < 0.06


def test_ece_overconfident():
    # always 0.9 confident but only 50% correct -> ECE ~0.4
    conf = [0.9, 0.9, 0.9, 0.9]
    corr = [1.0, 0.0, 1.0, 0.0]
    ece = expected_calibration_error(conf, corr, n_bins=10)
    assert ece is not None and approx(ece, 0.4, tol=0.05)


def test_ece_none_when_no_confidence():
    assert expected_calibration_error([], []) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
