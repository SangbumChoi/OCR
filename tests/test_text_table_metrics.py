"""New metrics for the proposed format: CER/WER/NED + TEDS table metric."""

from docvlm_eval.metrics import cer, ned_similarity, teds, teds_struct, wer
from docvlm_eval.metrics.text import score_sample


def test_cer():
    assert cer("hello", "hello") == 0.0
    assert abs(cer("helo", "hello") - 0.2) < 1e-9  # 1 deletion / 5 chars
    assert cer("", "abc") == 1.0


def test_wer():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the dog sat", "the cat sat") - 1 / 3) < 1e-9


def test_ned_similarity_partial_credit():
    # ANLS would zero this (>0.5 dist); NED gives graded credit
    s = ned_similarity("hello world", ["hella world"])
    assert 0.8 < s < 1.0
    assert ned_similarity("hello", ["hello"]) == 1.0


def test_ned_dispatch():
    assert score_sample("ned", "abc", ["abc"]) == 1.0


def test_teds_identical():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert teds(html, html) > 0.99
    assert teds_struct(html, html) == 1.0


def test_teds_structure_penalty():
    gold = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    wrong = "<table><tr><td>a</td></tr></table>"  # fewer rows + cols
    assert teds(wrong, gold) < teds(gold, gold)
    assert teds_struct(wrong, gold) < 1.0


def test_teds_content_matters():
    gold = "<table><tr><td>Item</td><td>Price</td></tr></table>"
    same_struct_diff_text = "<table><tr><td>XXXX</td><td>YYYY</td></tr></table>"
    # structure perfect but content wrong -> below 1.0
    assert teds_struct(same_struct_diff_text, gold) == 1.0
    assert teds(same_struct_diff_text, gold) < 0.9


def test_teds_dispatch():
    html = "<table><tr><td>a</td></tr></table>"
    assert score_sample("teds", html, [html]) > 0.99
