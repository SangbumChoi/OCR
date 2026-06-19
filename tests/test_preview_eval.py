"""Offline samples.jsonl (raw HF GT) -> eval Sample conversion. Schema-only, no network/data dep."""

from docvlm_eval.benchmarks.preview_eval import SPEC, case_samples


def _one(key, gt):
    return case_samples(key, gt, "img.jpg", SPEC[key], 0)


def test_vqa_question_answers():
    s = _one("docvqa", {"question": "What is the total?", "answers": ["$10", "10"]})
    assert len(s) == 1 and s[0].metric == "anls"
    assert s[0].answers == ["$10", "10"]
    assert s[0].question.startswith("What is the total?")


def test_ai2d_maps_option_index_to_text():
    s = _one("ai2d", {"question": "Which?", "options": ["c", "D", "b", "a"], "answer": 1})
    assert s[0].answers == ["D"]                  # options[1]
    assert "Options: c, D, b, a." in s[0].question
    assert s[0].metric == "exact"


def test_yesno_mapping():
    assert _one("pope", {"question": "Is there a cat?", "answer": "yes"})[0].answers == ["yes"]
    assert _one("hallusionbench", {"question": "Q?", "gt_answer": 0})[0].answers == ["no"]
    assert _one("hallusionbench", {"question": "Q?", "gt_answer": 1})[0].answers == ["yes"]


def test_fixed_instruction_transcription_tasks():
    s = _one("iam", {"text": "hello world"})
    assert s[0].metric == "ned" and s[0].answers == ["hello world"]
    assert "Transcribe" in s[0].question
    assert "concisely" not in s[0].question  # no concise suffix for ned/teds


def test_pubtabnet_teds_gold_is_html():
    s = _one("pubtabnet", {"html_table": "<table><tr><td>a</td></tr></table>"})
    assert s[0].metric == "teds" and s[0].answers[0].startswith("<table>")


def test_ocrvqa_multi_expands_parallel_qa_capped():
    s = _one("ocrvqa", {"questions": ["Q1", "Q2", "Q3", "Q4"],
                        "answers": ["A1", "A2", "A3", "A4"]})
    assert [x.answers[0] for x in s] == ["A1", "A2", "A3"]   # capped at 3


def test_no_gold_is_skipped():
    assert _one("docvqa", {"question": "Q?", "answers": []}) == []
    assert _one("textvqa", {"question": "Q?"}) == []  # missing answers field
