"""Offline tests for the metric bank (docvlm_eval.metrics.bank)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.metrics import (METRIC_BANK, cer_sim, drop_em, drop_normalize, score_all,
                                 score_sample, semantic_match, token_f1)


def test_token_f1_partial_credit_and_normalization():
    assert token_f1("5 days", ["The default period is 5 days"]) > 0.4
    assert token_f1("The 5 days", ["5 days"]) == 1.0        # articles normalized away
    assert token_f1("red", ["blue"]) == 0.0
    assert token_f1("b a", ["a b"]) == 1.0                  # bag-of-tokens: order-free


def test_drop_em_numbers_dates_separators():
    assert drop_em("five", ["5"]) == 1.0
    assert drop_em("twenty-one", ["21"]) == 1.0
    assert drop_em("1,000", ["1000"]) == 1.0
    assert drop_em("50%", ["50"]) == 1.0
    assert drop_em("$339,000", ["339000"]) == 1.0
    assert drop_em("January 5, 2020", ["5 Jan 2020"]) == 1.0
    assert drop_em("6", ["5"]) == 0.0                       # numeric EQUALITY, not tolerance
    assert drop_normalize("Five Days") == "5 days"


def test_cer_sim_similarity_semantics():
    assert cer_sim("hello world", ["hello world"]) == 1.0
    assert 0.7 < cer_sim("helo world", ["hello world"]) < 1.0
    assert cer_sim("", ["hello"]) == 0.0


def test_semantic_match_layered():
    assert semantic_match("ITC Limited.", ["itc limited"]) == 1.0      # layer 1: normalized EM
    assert semantic_match("five days", ["5 days"]) == 1.0              # layer 2: DROP canon
    assert 0 < semantic_match("5", ["The answer is 5 days"]) < 1.0     # layer 3: token-F1 partial
    assert semantic_match("red", ["blue"]) == 0.0


def test_case_tolerance_is_explicit_across_the_bank():
    # "Total" vs "total": every answer-style metric must treat pure case difference as correct —
    # this is the metric-specialty row the tendency matrix ("case change") makes visible.
    from docvlm_eval.metrics import METRIC_BANK
    forgiving = {m: fn("total", ["Total"]) for m, fn in METRIC_BANK.items()}
    for m in ("exact", "anls", "ned", "relaxed_acc", "ocrbench", "token_f1", "drop_em",
              "semantic_match"):
        assert forgiving[m] == 1.0, f"{m} punished a pure case change: {forgiving[m]}"
    # cer_sim is the DELIBERATE exception: recognition CER is case-sensitive by convention
    assert forgiving["cer_sim"] < 1.0


def test_score_all_covers_bank_and_registry_dispatch():
    s = score_all("five", ["5"])
    assert set(s) == set(METRIC_BANK)
    assert s["drop_em"] == 1.0 and s["exact"] == 0.0        # the tendency difference, in one row
    # bank metrics are also dispatchable as a Sample.metric
    assert score_sample("semantic_match", "five", ["5"]) == 1.0
    assert score_sample("token_f1", "b a", ["a b"]) == 1.0
