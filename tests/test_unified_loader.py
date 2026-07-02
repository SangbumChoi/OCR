"""Offline tests for the unified task-typed loader (docvlm_eval.unified).

No network: feed hand-built raw records mimicking each benchmark's real schema and assert the
UnifiedSample carries the right task + structured payload (fields/regions/boxes/table), and that
to_sample() collapses back to a trainable flat Sample."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.unified import (QA, Task, UnifiedSample, extract_unified,
                                            merge_by_image, to_training_samples)


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


def test_doclaynet_localization_boxes():
    ex = {"bboxes": [[72.0, 55.0, 300.0, 20.0]], "category_id": [6],
          "metadata": {"coco_width": 1000, "coco_height": 1000}}
    r = _one("doclaynet", ex)[0]
    assert r.task == Task.LOCALIZATION
    rg = r.regions[0]
    assert rg.label == "Page-header" and rg.bbox.normalized is True
    assert abs(rg.bbox.x2 - 0.372) < 1e-6      # (72+300)/1000, xywh -> normalized corner


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


def test_merge_by_image_groups_qa_list():
    # three OCR-VQA-style records: same image, different questions -> one record with a qas list
    rows = [
        UnifiedSample(sample_id="ocrvqa_0000_0", source="ocrvqa", task=Task.VQA,
                      image_path="/tmp/a.jpg", instruction="Who wrote this?", answers=["Smith"]),
        UnifiedSample(sample_id="ocrvqa_0000_1", source="ocrvqa", task=Task.VQA,
                      image_path="/tmp/a.jpg", instruction="What is the title?", answers=["Physics"]),
        UnifiedSample(sample_id="ocrvqa_0001_0", source="ocrvqa", task=Task.VQA,
                      image_path="/tmp/b.jpg", instruction="What genre?", answers=["Science"]),
    ]
    merged = merge_by_image(rows)
    assert len(merged) == 2                              # two distinct images
    first = merged[0]
    assert [qa.question for qa in first.qas] == ["Who wrote this?", "What is the title?"]
    assert not first.instruction and not first.answers   # collapsed into qas
    # lossless: expands back to the SAME number of training samples
    assert len(to_training_samples(merged)) == 3
    ids = {s.sample_id for s in to_training_samples(merged)}
    assert "ocrvqa_0000_0_q0" in ids and "ocrvqa_0000_0_q1" in ids


def test_merge_by_image_dedupes_identical_questions():
    rows = [UnifiedSample(sample_id="x_0_0", source="docvqa", task=Task.VQA, image_path="/i.jpg",
                          instruction="Total?", answers=["$5"]),
            UnifiedSample(sample_id="x_0_1", source="docvqa", task=Task.VQA, image_path="/i.jpg",
                          instruction="Total?", answers=["$5"])]
    merged = merge_by_image(rows)
    assert len(merged) == 1 and len(merged[0].qas) == 1   # duplicate question merged


def test_to_samples_expands_qas():
    r = UnifiedSample(sample_id="i_0_0", source="ocrvqa", task=Task.VQA, image_path="/i.jpg",
                      qas=[QA("Q1", ["a1"]), QA("Q2", ["a2"]), QA("Q3", [])])
    s = r.to_samples()
    assert len(s) == 2 and s[0].question == "Q1" and s[0].meta["n_qas"] == 3   # empty-answer QA dropped


def test_cauldron_format_multiple_turns():
    ex = {"images": [object()],
          "texts": [{"user": "What is the title?", "assistant": "Physics", "source": "ST-VQA"},
                    {"user": "Who wrote it?", "assistant": "Smith", "source": "ST-VQA"}]}
    recs = _one("stvqa", ex)
    assert len(recs) == 2 and recs[0].task == Task.VQA and recs[0].answers == ["Physics"]
    assert _one("tatqa", {"texts": [{"user": "Total assets?", "assistant": "$5M"}]})[0].task \
        == Task.REASONING                                  # task typed per key, same adapter
    # PlotQA-style 90-turn records are capped so one source can't drown the corpus
    many = {"texts": [{"user": f"Q{i}?", "assistant": str(i)} for i in range(90)]}
    assert len(_one("plotqa", many)) == 5


def test_mtvqa_json_qa_pairs_and_language():
    ex = {"qa_pairs": json.dumps([{"question": "제목은?", "answer": "물리학"}]), "lang": "KO"}
    r = _one("mtvqa", ex)[0]
    assert r.task == Task.VQA and r.language == "ko" and r.answers == ["물리학"]
    # the real dataset ships PYTHON-REPR strings (single quotes), not JSON
    ex2 = {"qa_pairs": "[{'question': 'ما هو نوع المنتج؟', 'answer': 'جبنة'}]", "lang": "AR"}
    r2 = _one("mtvqa", ex2)[0]
    assert r2.language == "ar" and r2.answers == ["جبنة"]


def test_screenqa_regions_from_ui_elements():
    ex = {"question": "What is the default period?",
          "ground_truth": [{"full_answer": "The default is 5 days.",
                            "ui_elements": [{"bounds": [501.0, 291.0, 562.0, 363.0], "text": "5"}]}]}
    r = _one("screenqa", ex)[0]
    assert r.task == Task.VQA and r.answers == ["The default is 5 days."]
    assert r.regions and r.regions[0].bbox.normalized is False    # pixel screen bounds


def test_publaynet_normalizes_by_image_size():
    class _Img:  # minimal stand-in with .size
        size = (1000, 2000)
    ex = {"image": _Img(), "annotations": [{"bbox": [100, 200, 300, 400], "category_id": 4}]}
    r = _one("publaynet", ex)[0]
    rg = r.regions[0]
    assert r.task == Task.LOCALIZATION and rg.label == "Table" and rg.bbox.normalized
    assert abs(rg.bbox.x2 - 0.4) < 1e-6 and abs(rg.bbox.y2 - 0.3) < 1e-6   # (100+300)/1000, (200+400)/2000


def test_rvl_cdip_classification():
    r = _one("rvl_cdip", {"label": 11})[0]
    assert r.task == Task.CLASSIFICATION and r.answers == ["invoice"] and r.metric == "exact"


def test_synthdog_recognition_language():
    gt = json.dumps({"gt_parse": {"text_sequence": "3 위 에 올 랐 다"}})
    r = _one("synthdog_ko", {"ground_truth": gt})[0]
    assert r.task == Task.RECOGNITION and r.language == "ko" and r.full_text.startswith("3 위")


def test_detect_language_scripts_and_priors():
    from docvlm_eval.unified import detect_language
    assert detect_language("총 합계 60,000원 감사합니다") == "ko"
    assert detect_language("合計金額は五千円です、ありがとうございます") == "ja"     # kana -> ja, not zh
    assert detect_language("发票号码 12345 合计金额") == "zh"
    assert detect_language("The total amount is $5.00") == "en"
    assert detect_language("TOTAL HARGA 50.000 TUNAI", source="cord") == "id"       # Latin -> source prior
    assert detect_language("x^2 + y^2 = z^2", source="latexocr") == "und"
    assert detect_language("Invoice no 42 with stray 中 glyph") == "en"             # noise floor holds
    assert detect_language("") is None and detect_language("", source="cord") == "id"


def test_extract_unified_fills_language():
    r = _one("cord", {"ground_truth": json.dumps({"gt_parse": {"total": "50.000"}})})[0]
    assert r.language == "id"                       # source prior applied at extraction time
    r2 = _one("docvqa", {"question": "Total?", "answers": ["$5 in total please"]})[0]
    assert r2.language == "en"


def test_enrich_record_counts_and_dims():
    from docvlm_eval.unified import enrich_record
    row = {"language": "", "source": "funsd", "instruction": "Transcribe.",
           "answers": ["Name Bob"], "full_text": "Name Bob",
           "fields_json": json.dumps([{"key": "B-QUESTION", "value": "Name"}]),
           "regions_json": json.dumps([{"label": "word", "text": ""}]), "image": None,
           "hf_id": "naver-clova-ix/cord-v2"}
    out = enrich_record(row)
    assert out["language"] == "en" and out["n_fields"] == 1 and out["n_regions"] == 1
    assert out["image_width"] == 0                  # no image object -> dims default to 0
    assert out["phash"] == "" and out["license"] == "cc-by-4.0"


def test_to_grounding_samples_format(tmp_path):
    from PIL import Image
    from docvlm_eval.unified import Box, Region
    ip = tmp_path / "page.jpg"
    Image.new("RGB", (1000, 500), "white").save(ip)
    r = UnifiedSample(sample_id="doclaynet_0000_0", source="doclaynet", task=Task.LOCALIZATION,
                      image_path=str(ip),
                      regions=[Region("Table", Box(0.1, 0.2, 0.5, 0.6, True)),
                               Region("Table", Box(0.6, 0.2, 0.9, 0.6, True)),
                               Region("Title", Box(0.0, 0.0, 1.0, 0.1, True))])
    ss = r.to_grounding_samples()
    assert len(ss) == 2                                   # grouped by LABEL, not per region
    table = next(s for s in ss if "Table" in s.question)
    assert all(s.metric == "grounding" for s in ss)
    assert len(table.answers) == 2                                   # 2 golds, best-IoU semantics
    assert table.answers[0] == "100,100,500,300;1000,500"            # normalized -> stored pixels
    # no regions / no image -> nothing
    assert UnifiedSample(sample_id="x", source="s", task=Task.LOCALIZATION).to_grounding_samples() == []


def test_dhash_stability_and_hamming():
    from PIL import Image, ImageDraw
    from docvlm_eval.unified import dhash, hamming
    img = Image.new("RGB", (200, 100), "white")
    d = ImageDraw.Draw(img); d.rectangle([20, 20, 120, 80], fill="black")
    h1 = dhash(img)
    # resize + JPEG-style re-render -> same/near hash (that's the point vs byte md5)
    h2 = dhash(img.resize((150, 75)))
    assert hamming(h1, h2) <= 6
    other = Image.new("RGB", (200, 100), "white")
    d2 = ImageDraw.Draw(other); d2.ellipse([100, 10, 190, 90], fill="black")
    assert hamming(h1, dhash(other)) > 6            # different content -> far hash


def test_unmappable_returns_empty():
    assert extract_unified("pubtables1m", {"image": object(), "boxes": [[1, 2, 3, 4]]},
                           {"key": "pubtables1m"}) == []
