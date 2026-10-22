"""Per-benchmark rationale extraction + format-bound instruction/answer diversification (offline)."""

from __future__ import annotations

import json

import pytest

from docvlm_eval.unified import canon_answers
from docvlm_eval.unified.core import Box, Field, Region, Task, UnifiedSample, normalize_pixel_boxes
from docvlm_eval.unified.diversify import (
    DiversityConfig,
    answer_variants,
    conditions,
    diversify_qa,
    diversify_row,
    parse_options,
    strip_format_clause,
    strip_period,
)
from docvlm_eval.unified.rationale import (
    Rationale,
    attach,
    derive,
    parse_number,
    parse_table,
    quality,
    safe_eval,
)


def _row(source, qs, ans, **kw):
    r = {"sample_id": f"{source}_0000_0", "source": source, "task": kw.pop("task", "vqa"),
         "instructions": list(qs), "answers": list(ans), "answer_variants": [[a] for a in ans],
         "reasoning": [""] * len(qs), "task_detail": ["x"] * len(qs), "qa_tasks": ["vqa"] * len(qs),
         "elements_json": "[]", "full_annotation": "", "image_width": 1000, "image_height": 1000,
         "metric": "anls"}
    r.update(kw)
    return r


# --------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("s,v", [("24,000", 24000), ("60.000", 60000), ("Rp 12.000", 12000),
                                 ("$1,298.00", 1298.0), ("(230)", -230), ("48.5.", 48.5),
                                 ("-34.12%", -34.12), ("0.039", 0.039), ("abc", None)])
def test_parse_number(s, v):
    assert parse_number(s) == v


def test_safe_eval_handles_accounting_and_rejects_names():
    assert safe_eval("$8,452-$10,322") == -1870
    assert safe_eval("7,572 - (6,919)") == 14491          # (6,919) is an accounting negative
    assert safe_eval("25%-24%") == 1
    assert safe_eval("__import__('os')") is None
    assert safe_eval("1/0") is None


# --------------------------------------------------------------------------- derivers
def test_tatqa_derivation_is_split_verified_and_repairs_the_gold():
    r = _row("tatqa", ["What was the change?", "What is the percentage change?\nAnswer scale should be: percent.",
                       "How many years exceeded 5?", "Which year is higher?", "Broken?"],
             ["74.5-26.0\nAnswer: 48.5.", "(104,858-159,154)/159,154\nAnswer: -34.12.",
              "2019\nAnswer: 1.", "Find the higher year\nAnswer: 2019.", "1+1\nAnswer: 5."])
    out, rep = attach(r)
    assert out["answers"] == ["48.5", "-34.12", "1", "2019", "5"]     # every gold repaired
    assert out["reasoning"][0] == "Compute 74.5-26.0 = 48.5."
    assert "as a percentage: -34.12%" in out["reasoning"][1]
    assert out["reasoning_provenance"][2].endswith("tatqa_count")
    assert out["reasoning"][3] == "" and out["reasoning"][4] == ""    # plan text / wrong arithmetic
    assert {w for _, w in rep["rejected"]} == {"derivation_without_values", "derivation_mismatch"}


def test_cord_arithmetic_only_when_the_receipt_adds_up():
    gt = {"menu": [{"nm": "A", "price": "24,000"}, {"nm": "B", "price": "4,000"}],
          "sub_total": {"subtotal_price": "28,000", "tax_price": "2,800"},
          "total": {"total_price": "30,800", "cashprice": "50,000", "changeprice": "19,200"}}
    out, _ = attach(_row("cord", ["Extract the receipt's key fields as JSON."], [json.dumps(gt)]))
    derived = dict(zip(out["task_detail"][1:], out["answers"][1:]))
    assert derived == {"derived:cord_subtotal": "28,000", "derived:cord_total": "30,800",
                       "derived:cord_change": "19,200", "derived:cord_max_item": "A"}
    gt["total"]["changeprice"] = "10,000"                               # does not add up -> no QA
    out, _ = attach(_row("cord", ["Extract."], [json.dumps(gt)]))
    assert "derived:cord_change" not in out["task_detail"]


def test_funsd_link_is_grounded_in_word_boxes():
    els = [{"bbox": [0.1, 0.2, 0.15, 0.22, True], "information": "DATE:", "class": "B-QUESTION"},
           {"bbox": [0.2, 0.2, 0.25, 0.22, True], "information": "2-", "class": "B-ANSWER"},
           {"bbox": [0.26, 0.2, 0.3, 0.22, True], "information": "Dec-", "class": "I-ANSWER"},
           {"bbox": [0.31, 0.2, 0.33, 0.22, True], "information": "97", "class": "I-ANSWER"}]
    r = _row("funsd", ["Find the values."], [str({"DATE:": ["2- Dec- 97"]})], elements_json=json.dumps(els))
    (x,) = derive(r)
    assert x.answers == ["2- Dec- 97"] and "to the right of the label" in x.reasoning
    assert x.box == [0.2, 0.2, 0.33, 0.22]


def test_ocrvqa_yes_no_uses_the_rows_own_genre_answer():
    r = _row("ocrvqa", ["What type of book is this?", "Is this book related to Travel?",
                        "Is this book related to History?"], ["History", "No", "Yes"])
    texts = {x.qa_index: x.reasoning for x in derive(r)}
    assert "different category" in texts[1] and "exactly the topic" in texts[2]
    r["answers"][1] = "Yes"                                             # inconsistent pair -> skip
    assert 1 not in {x.qa_index for x in derive(r)}


def test_pixel_boxes_are_not_trusted():
    els = [{"bbox": [174.0, 433.0, 590.0, 552.0, False], "information": "Fri, Jun 15", "class": "ui_element"}]
    r = _row("screenqa", ["What is the date?"], ["The date is Friday, Jun 15."], elements_json=json.dumps(els))
    (x,) = derive(r)
    assert x.box is None and "Fri, Jun 15" in x.reasoning


def test_parse_table_refuses_spans():
    grid, n_head = parse_table("<table><thead><tr><td>k</td><td>v</td></tr></thead>"
                               "<tbody><tr><td>a</td><td>1</td></tr></tbody></table>")
    assert grid == [["k", "v"], ["a", "1"]] and n_head == 1
    assert parse_table('<table><tr><td colspan="2">x</td></tr><tr><td>a</td><td>b</td></tr></table>') is None


# --------------------------------------------------------------------------- gate
@pytest.mark.parametrize("r,why", [
    (Rationale("The answer is Paris.", "derived_exact", "lookup", ["Paris"], 0), "tautological"),
    (Rationale("The ledger row reads 'Lyon' in column two.", "derived_exact", "lookup", ["Lyon"], 0),
     "evidence_not_in_row"),
    (Rationale("Cash 50 minus total 20 equals 25 left.", "derived_exact", "arithmetic", ["50"], 0),
     "arithmetic_mismatch"),
    (Rationale("Yes, Paris is shown on the map legend.", "source", "consistency", ["Paris"], 0),
     "contradicts_answer"),
])
def test_quality_gate_rejections(r, why):
    row = _row("x", ["Is the capital Paris? 50"], ["no" if why == "contradicts_answer" else
                                                   ("30" if why == "arithmetic_mismatch" else "Paris")])
    ok, reason = quality(r, row)
    assert not ok and reason == why


def test_rejected_source_rationale_is_moved_aside():
    r = _row("hallusionbench", ["Is the bar for 2010 the highest in the chart?"], ["no"],
             reasoning=["Yes, the bar for 2010 is the highest in the chart."])
    out, _ = attach(r)
    # the paired-question verdict is stripped, and what is left only restates the question
    assert out["reasoning"] == [""] and out["reasoning_provenance"] == ["rejected:tautological"]
    assert out["reasoning_rejected"] == ["Yes, the bar for 2010 is the highest in the chart."]


def test_hallusion_verdict_of_the_paired_question_is_stripped():
    r = _row("hallusionbench", ["In 2013, was the United States the largest recipient?"], ["no"],
             reasoning=["Yes. In 2013, Hong Kong was the largest recipient, followed by Germany."])
    out, _ = attach(r)
    assert out["reasoning"] == ["In 2013, Hong Kong was the largest recipient, followed by Germany."]
    assert out["reasoning_provenance"] == ["source:hallusionbench_details_verdict_stripped"]


def test_ocrvqa_words_anchor_on_the_title_not_a_repeated_tagline():
    els = [{"bbox": [0.1, 0.10, 0.9, 0.30, True], "information": "killer", "class": "word"},
           {"bbox": [0.1, 0.30, 0.9, 0.45, True], "information": "doku", "class": "word"},
           {"bbox": [0.3, 0.92, 0.4, 0.95, True], "information": "killer", "class": "word"},
           {"bbox": [0.5, 0.92, 0.6, 0.95, True], "information": "doku", "class": "word"}]
    r = _row("ocrvqa", ["What is the title of this book?"], ["Killer Doku"], elements_json=json.dumps(els))
    (x,) = derive(r)
    assert x.box == [0.1, 0.1, 0.9, 0.45] and "top-center" in x.reasoning
    # a key word found ONLY far away (tagline) is unmatched, not pulled into the evidence box
    els[0]["information"] = "noise"
    (x,) = derive(_row("ocrvqa", ["What is the title of this book?"], ["Killer Doku Grid"],
                       elements_json=json.dumps(els + [{"bbox": [0.2, 0.46, 0.8, 0.5, True],
                                                        "information": "grid", "class": "word"}])))
    assert x.box[3] <= 0.5 and "2 of the 3" in x.reasoning


def test_publaynet_does_not_count_titles_and_omnidoc_title_drops_markdown():
    els = [{"bbox": [0.1, 0.1, 0.4, 0.12, True], "information": "", "class": "Title"},
           {"bbox": [0.1, 0.2, 0.4, 0.5, True], "information": "", "class": "Figure"}]
    qs = [x.question for x in derive(_row("publaynet", ["Localize."], [""], elements_json=json.dumps(els)))]
    assert qs == ["How many figures are on this page?"]
    els = [{"bbox": [0.1, 0.1, 0.5, 0.15, True], "information": "# Cubic Splines", "class": "title"}]
    (t,) = [x for x in derive(_row("omnidocbench", ["Parse."], ["x"], elements_json=json.dumps(els)))
            if x.deriver == "omnidoc_title"]
    assert t.answers == ["Cubic Splines"]


def test_attach_keeps_columns_aligned():
    gt = {"menu": [{"nm": "A", "price": "10"}, {"nm": "B", "price": "5"}], "sub_total": {"subtotal_price": "15"},
          "total": {"total_price": "15"}}
    out, _ = attach(_row("cord", ["Extract."], [json.dumps(gt)]))
    n = len(out["instructions"])
    for k in ("answers", "answer_variants", "reasoning", "task_detail", "qa_tasks",
              "reasoning_provenance", "evidence_box"):
        assert len(out[k]) == n, k


# --------------------------------------------------------------------------- diversify
@pytest.fixture(scope="module")
def cfg():
    return DiversityConfig.from_yaml()


def test_cauldron_suffixes_and_trailing_periods(cfg):
    assert strip_format_clause("Which bar is largest?\nOffer a terse response.") == "Which bar is largest?"
    # semantic clauses survive
    q = "What is the change?\nAnswer scale should be: percent."
    assert strip_format_clause(q) == q
    assert strip_period("Mean.") == "Mean" and strip_period("83.99.") == "83.99"
    assert strip_period("The answer is in the U.S.") == "The answer is in the U.S."
    assert answer_variants("Mean.", cfg)[0] == "Mean"
    assert {"2305", "2,305", "2305.00"} <= set(answer_variants("2,305.", cfg))


def test_conditions_gate_formats():
    assert "transcription" in conditions("Transcribe all the text in the image.", "TAN", "", None)
    assert "abstain" in conditions("When?", "<no answer>", "", None)
    assert "lang_en" not in conditions("診療時間は何時からですか", "午前9時", "", None)
    q = "Larvae turn into what form?\nOptions: 1) Ovums; 2) Eggs; 3) Adults"
    assert parse_options(q) == ["Ovums", "Eggs", "Adults"] and "has_options" in conditions(q, "Adults", "", None)


def test_variants_bind_instruction_and_answer(cfg):
    vs = diversify_qa("s", "ai2d", 0, "Larvae turn into what form?\nOptions: 1) Ovums; 2) Eggs; 3) Adults",
                      "Adults", ["Adults"], "", "exact", "", cfg)
    by = {v.format: v for v in vs}
    assert by["original"].answers[0] == "Adults"
    for v in vs:
        if v.format == "mcq_index":
            assert v.answers == ["3"] and "option" in v.instruction.lower()
        if v.format == "sentence":
            assert v.answers[0] == "The answer is Adults."
    # transcription / abstain / non-English: no re-rendered formats
    assert [v.format for v in diversify_qa("s", "sroie", 0, "Transcribe all the text in the image.",
                                           "TAN WOON", [], "", "ned", "", cfg)] == ["original"]
    assert [v.format for v in diversify_qa("s", "screenqa", 0, "When?", "<no answer>", [], "", "anls", "", cfg)] == ["original"]
    assert all(v.format != "sentence" for v in diversify_qa("s", "mtvqa", 0, "診療時間は何時からですか",
                                                            "9時", [], "", "anls", "", cfg))


def test_reasoned_format_only_with_a_rationale_and_is_deterministic(cfg):
    row = {"sample_id": "t", "source": "tatqa", "instructions": ["What was the change?"], "answers": ["48.5"],
           "answer_variants": [["48.5"]], "reasoning": ["Compute 74.5-26.0 = 48.5."], "metric": "anls"}
    a, b = diversify_row(row, cfg), diversify_row(row, cfg)
    assert [(v.instruction, v.answers) for v in a] == [(v.instruction, v.answers) for v in b]
    rs = [v for v in a if v.format == "reasoned"]
    assert rs and rs[0].answers == ["Reasoning: Compute 74.5-26.0 = 48.5.\nAnswer: 48.5"]
    row["reasoning"] = [""]
    assert not any(v.format == "reasoned" for v in diversify_row(row, cfg))


# --------------------------------------------------------------------------- loader fixes
def test_canon_answers_puts_the_majority_first():
    tv = ["nous les gosses", "dakota", "clos culombu", "dakota digital", "dakota", "dakota",
          "dakota digital", "dakota digital", "dakota", "dakota"]
    assert canon_answers(tv)[0] == "dakota"
    assert canon_answers(["5 days", "five days"]) == ["5 days", "five days"]


def test_normalize_pixel_boxes_before_resize():
    r = UnifiedSample("s", "cord", Task.KIE, fields=[Field("total", "9", Box(100, 200, 300, 400))],
                      regions=[Region("w", Box(0.1, 0.1, 0.2, 0.2, normalized=True))])
    assert normalize_pixel_boxes([r], 1000, 2000) == 1
    b = r.fields[0].bbox
    assert b.normalized and (b.x1, b.y1, b.x2, b.y2) == (0.1, 0.1, 0.3, 0.2)


def test_script_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("d", root / "scripts" / "derive_udd_reasoning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [_row("tatqa", ["What was the change?"], ["74.5-26.0\nAnswer: 48.5."]),
            _row("dvqa", ["Which bar is largest?\nOffer a terse response."], ["Soil."])]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    assert mod.main(["--rows", str(tmp_path / "rows.json"), "--out", str(tmp_path / "o"),
                     "--stats", str(tmp_path / "s.json"), "--preview", str(tmp_path / "p.md")]) == 0
    stats = json.loads((tmp_path / "s.json").read_text())
    assert stats["per_source"]["tatqa"]["answers_repaired"] == 1
    vs = [json.loads(x) for x in (tmp_path / "o" / "variants.jsonl").read_text().splitlines()]
    assert all("terse" not in v["question"] for v in vs if v["meta"]["format"] != "original")
    assert "## tatqa" in (tmp_path / "p.md").read_text()
