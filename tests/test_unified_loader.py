"""Offline tests for the unified task-typed loader (docvlm_eval.benchmarks.unified).

No network: feed hand-built raw records mimicking each benchmark's real schema and assert the
UnifiedSample carries the right task + structured payload (fields/regions/boxes/table), and that
to_sample() collapses back to a trainable flat Sample."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks.unified import (Task, UnifiedSample, extract_unified,
                                            to_training_samples)


def _one(key, ex):
    recs = extract_unified(key, ex, {"key": key, "metric": "anls"})
    assert recs, f"{key} produced no unified record"
    return recs


def test_cord_kie_fields_and_boxes():
    gt = json.dumps({
        "gt_parse": {"menu": {"nm": "Coffee", "price": "4.50"}, "total": "4.50"},
        "valid_line": [{"category": "menu.nm",
                        "words": [{"text": "Coffee",
                                   "quad": {"x1": 10, "y1": 20, "x2": 60, "y2": 20,
                                            "x3": 60, "y3": 40, "x4": 10, "y4": 40}}]}],
    })
    r = _one("cord", {"ground_truth": gt})[0]
    assert r.task == Task.KIE
    keys = {f.key for f in r.fields}
    assert "total" in keys and any("nm" in k for k in keys)        # flattened gt_parse
    boxed = [f for f in r.fields if f.bbox]
    assert boxed and boxed[0].bbox.to_list() == [10.0, 20.0, 60.0, 40.0]   # quad -> enclosing box
    payload = json.loads(r.answers[0])
    assert payload["total"] == "4.50"


def test_funsd_kie_with_word_boxes():
    ex = {"words": ["Name", "Bob"], "bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "ner_tags": [3, 5]}
    r = _one("funsd", ex)[0]
    assert r.task == Task.KIE
    assert [f.key for f in r.fields] == ["B-QUESTION", "B-ANSWER"]
    assert r.fields[0].bbox.normalized is True
    assert r.full_text == "Name Bob"


def test_ocrvqa_vqa_with_regions():
    ex = {"questions": ["Who wrote this?"], "answers": ["A. Author"],
          "ocr_info": [{"word": "TITLE",
                        "bounding_box": {"top_left_x": 0.1, "top_left_y": 0.2,
                                         "width": 0.3, "height": 0.05}}]}
    r = _one("ocrvqa", ex)[0]
    assert r.task == Task.VQA and r.answers == ["A. Author"]
    assert r.regions and r.regions[0].bbox.normalized
    assert abs(r.regions[0].bbox.x2 - 0.4) < 1e-6           # 0.1 + 0.3


def test_table_task():
    r = _one("pubtabnet", {"html_table": "<table><tr><td>a</td></tr></table>"})[0]
    assert r.task == Task.TABLE and r.table_html.startswith("<table")


def test_fallback_tasks_via_trainset():
    # docvqa -> VQA; iam -> recognition; chartqa -> reasoning (TASK_BY_BENCHMARK)
    assert _one("docvqa", {"question": "Total?", "answers": ["$5"]})[0].task == Task.VQA
    assert _one("iam", {"text": "hello world"})[0].task == Task.RECOGNITION
    assert _one("chartqa", {"question": "Tallest bar?", "answer": "2021"})[0].task == Task.REASONING


def test_to_sample_roundtrip():
    r = UnifiedSample(sample_id="x_0_0", source="cord", task=Task.KIE, image_path="/tmp/x.jpg",
                      fields=[], answers=["{\"total\": \"4.50\"}"], metric="anls")
    s = r.to_sample()
    assert s is not None and s.answer_type == Task.KIE and s.meta["task"] == Task.KIE
    # no image -> no trainable sample
    r2 = UnifiedSample(sample_id="y", source="cord", task=Task.KIE, answers=["x"])
    assert r2.to_sample() is None


def test_to_sample_derives_target_from_payload():
    # recognition with full_text but empty answers still yields a trainable sample
    r = UnifiedSample(sample_id="z_0_0", source="iam", task=Task.RECOGNITION,
                      image_path="/tmp/z.jpg", full_text="the quick brown fox")
    s = r.to_sample()
    assert s and s.answers == ["the quick brown fox"]


def test_to_training_samples_filters_unusable():
    rows = [UnifiedSample(sample_id="a", source="x", task=Task.VQA),          # no image/answer
            UnifiedSample(sample_id="b_0_0", source="x", task=Task.VQA,
                          image_path="/tmp/b.jpg", answers=["yes"])]
    assert len(to_training_samples(rows)) == 1


def test_unmappable_returns_empty():
    assert extract_unified("pubtables1m", {"image": object(), "boxes": [[1, 2, 3, 4]]},
                           {"key": "pubtables1m"}) == []
