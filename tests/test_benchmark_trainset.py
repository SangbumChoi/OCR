"""Offline tests for the benchmark->DTO adapters (docvlm_eval.benchmarks.trainset).

No network: we feed hand-built raw example dicts mimicking each benchmark's real schema and assert
the normalised QA matches our Sample DTO (question + non-empty answers + sane metric)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks.trainset import extract_qa, norm_metric


def _one(key, ex):
    qas = extract_qa(key, ex, {"key": key})
    assert qas, f"{key} produced no QA"
    for q in qas:
        assert q["question"].strip()
        assert q["answers"] and all(a.strip() for a in q["answers"])
        assert q["metric"] in {"anls", "exact", "relaxed_acc", "ocrbench", "ned"}
    return qas


def test_metric_normalisation():
    assert norm_metric("CER / WER / NED") == "ned"
    assert norm_metric("TEDS / TEDS-S") == "anls"
    assert norm_metric("relaxed_acc") == "relaxed_acc"
    assert norm_metric(None) == "anls"
    assert norm_metric("entity F1") == "anls"


def test_vqa_family():
    for key in ("docvqa", "infovqa", "textvqa", "stvqa", "ocrvqa"):
        qas = _one(key, {"question": "What is the total?", "answers": ["$12.00", "12.00"]})
        assert qas[0]["answers"][0] == "$12.00"


def test_chartqa_uses_relaxed_acc():
    qas = _one("chartqa", {"question": "Which bar is tallest?", "answer": "2021"})
    assert qas[0]["metric"] == "relaxed_acc"


def test_iam_transcription():
    qas = _one("iam", {"text": "the quick brown fox", "image": object()})
    assert qas[0]["answers"] == ["the quick brown fox"]
    assert qas[0]["metric"] == "ned"


def test_latex_formula():
    qas = _one("latexocr", {"text": "x^2 + y^2 = z^2"})
    assert "LaTeX" in qas[0]["question"]
    qas2 = _one("im2latex", {"latex_formula": "\\frac{a}{b}"})
    assert qas2[0]["answers"] == ["\\frac{a}{b}"]


def test_cord_kie_json():
    gt = json.dumps({"gt_parse": {"menu": {"nm": "Coffee", "price": "4.50"}, "total": "4.50"}})
    qas = _one("cord", {"ground_truth": gt})
    assert qas[0]["answer_type"] == "kie"
    payload = json.loads(qas[0]["answers"][0])      # the answer must be valid JSON of the fields
    assert payload["total"] == "4.50"


def test_ai2d_mcq_index_answer():
    qas = _one("ai2d", {"question": "Which is the root?", "options": ["leaf", "root", "stem"],
                        "answer": 1})
    assert qas[0]["answers"] == ["root"]
    assert "Options:" in qas[0]["question"]
    assert qas[0]["metric"] == "exact"


def test_table_to_html():
    qas = _one("pubtabnet", {"html_table": "<table><tr><td>a</td></tr></table>"})
    assert qas[0]["answer_type"] == "table"


def test_ocrvqa_parallel_lists():
    ex = {"image": object(),
          "questions": ["Who wrote this book?", "What is the title?"],
          "answers": ["The Times Mind Games", "Killer Su Doku 6"]}
    qas = extract_qa("ocrvqa", ex, {"key": "ocrvqa"})
    assert len(qas) == 2
    assert qas[0]["answers"] == ["The Times Mind Games"]
    assert qas[1]["question"].startswith("What is the title")


def test_charxiv_reasoning_pair():
    ex = {"image": object(), "descriptive_q1": 7, "descriptive_a1": "60",
          "reasoning_q": "Which model declines more?", "reasoning_a": "Joint-CNN"}
    qas = _one("charxiv", ex)
    assert qas[0]["answers"] == ["Joint-CNN"]
    assert qas[0]["answer_type"] == "sci-figure"


def test_conversations_fallback():
    ex = {"conversations": [{"from": "human", "value": "read it"},
                            {"from": "gpt", "value": "INVOICE #42"}]}
    qas = extract_qa("some_unregistered_bench", ex, {"key": "x", "metric": "anls"})
    assert qas and qas[0]["answers"] == ["INVOICE #42"]


def test_unmappable_returns_empty():
    # detection-only record (no question, no text target) -> skipped, not crashed
    assert extract_qa("pubtables1m", {"image": object(), "boxes": [[1, 2, 3, 4]]},
                      {"key": "pubtables1m"}) == []
