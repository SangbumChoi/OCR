"""Offline tests for the reading-order metric split (docvlm_eval.metrics.order)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.metrics import content_bag, order_tau, score_sample

GOLD = ["1. mix the flour\n2. add two eggs\n3. whisk until smooth\n4. pour into the pan"]


def test_perfect_reader():
    pred = "1. mix the flour 2. add two eggs 3. whisk until smooth 4. pour into the pan"
    assert content_bag(pred, GOLD) == 1.0 and order_tau(pred, GOLD) == 1.0


def test_reversed_reader_isolates_order_failure():
    pred = "4. pour into the pan 3. whisk until smooth 2. add two eggs 1. mix the flour"
    assert content_bag(pred, GOLD) == 1.0          # read EVERYTHING...
    assert order_tau(pred, GOLD) == 0.0            # ...in exactly the wrong order


def test_row_major_scramble_scores_between():
    pred = "1. mix the flour 3. whisk until smooth 2. add two eggs 4. pour into the pan"
    assert content_bag(pred, GOLD) == 1.0
    assert 0.5 < order_tau(pred, GOLD) < 1.0       # one swapped pair -> partial order credit


def test_half_reader_isolates_content_failure():
    pred = "1. mix the flour 2. add two eggs"
    assert content_bag(pred, GOLD) == 0.5          # half the elements missing...
    assert order_tau(pred, GOLD) == 1.0            # ...but what was read is in order


def test_garbage_and_dispatch():
    assert content_bag("lorem ipsum dolor", GOLD) == 0.0
    assert order_tau("lorem ipsum dolor", GOLD) == 0.0     # <2 found elements -> no order evidence
    assert score_sample("order_tau", "1. mix the flour 2. add two eggs", GOLD) == 1.0
    assert score_sample("content_bag", "1. mix the flour 2. add two eggs", GOLD) == 0.5


def test_ocr_noise_tolerance_on_long_elements():
    # one typo inside a long element: the half-substring fallback still finds it
    pred = "1. mix the flour 2. add two eggs 3. whisk umtil smooth 4. pour into the pan"
    assert content_bag(pred, GOLD) == 1.0
    assert order_tau(pred, GOLD) == 1.0
