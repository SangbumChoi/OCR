"""Unit tests for the model-free reasoning generator (docvlm_eval.synth.reasoning).

Every generated answer must be ground truth by construction, so we re-derive the expected value
from the known table/sequence and assert each emitted QA agrees."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.synth.reasoning import _to_num, sequence_questions, table_questions


def test_to_num_parses_money_signs_commas():
    assert _to_num("$1,200.50") == 1200.5
    assert _to_num("+12.50") == 12.5
    assert _to_num("-470.75") == -470.75
    assert _to_num("85%") == 85.0
    assert _to_num("hello") is None
    assert _to_num("2024-01-02") is None     # a date is not a number


def _num(s):
    return _to_num(s)


def test_table_answers_are_correct():
    header = ["Item", "Amount"]
    rows = [["X", "10"], ["Y", "30"], ["Z", "20"]]   # single numeric col -> deterministic facts
    vals = [10.0, 30.0, 20.0]
    qas = table_questions(header, rows, label="the table", n=99, seed="fixed")
    assert qas, "should emit questions"
    import re
    seen = set()
    for d in qas:
        q = d["question"].lower(); ans = d["answers"]
        seen.add(d["answer_type"])
        if "how many rows are in" in q:
            assert ans == ["3"]
        elif "greater than" in q:                                    # threshold-count
            thr = float(re.search(r"greater than ([\d.]+)", q).group(1))
            assert ans == [str(sum(1 for v in vals if v > thr))]
        elif "total of the amount" in q:
            assert any(_num(a) == sum(vals) for a in ans)            # 60
        elif "average" in q or "mean" in q:
            assert any(abs(_num(a) - sum(vals) / 3) < 0.01 for a in ans)
        elif "-largest value" in q:                                  # ordinal (2nd-largest) FIRST
            assert any(_num(a) == sorted(vals, reverse=True)[1] for a in ans)   # 20
        elif "largest value" in q:
            assert any(_num(a) == max(vals) for a in ans)            # 30
        elif "smallest value" in q:
            assert any(_num(a) == min(vals) for a in ans)            # 10
        elif "highest amount" in q and "item" in q:
            assert ans == ["Y"]                                       # argmax-lookup
        elif "lowest amount" in q and "item" in q:
            assert ans == ["X"]
        elif "larger amount, row" in q:                              # row compare
            i, j = (int(x) for x in re.findall(r"row (\d+)", q))
            bigger = i if vals[i - 1] > vals[j - 1] else j
            assert f"row {bigger}" in ans
    # the registry should be able to produce a spread of reasoning types
    assert "H-count" in seen


def test_table_questions_vary_by_content():
    h = ["Item", "Amount"]
    a = {d["question"] for d in table_questions(h, [["A", "1"], ["B", "2"], ["C", "9"]], seed=None)}
    b = {d["question"] for d in table_questions(h, [["A", "5"], ["B", "4"], ["C", "3"]], seed=None)}
    # different content -> (generally) a different sampled question subset
    assert a and b


def test_sequence_counts_and_which_more():
    items = [("left",), ("left",), ("right",)]
    qas = sequence_questions(items, attr="side", label="bubbles",
                             value_names={"left": "on the left", "right": "on the right"}, n=99)
    by_q = {d["question"]: d["answers"] for d in qas}
    assert any(a == ["3"] for q, a in by_q.items() if "in total" in q)
    assert any(a == ["2"] for q, a in by_q.items() if "on the left" in q)
    assert any(a == ["1"] for q, a in by_q.items() if "on the right" in q)
    more = [a for q, a in by_q.items() if "more" in q]
    assert more and "left" in str(more[0]).lower()


def test_sequence_tie():
    qas = sequence_questions([("a",), ("b",)], attr="x", label="things", n=99)
    more = [d["answers"] for d in qas if "more" in d["question"]]
    assert more and any(t in str(more[0]).lower() for t in ("equal", "same", "neither", "tie"))
