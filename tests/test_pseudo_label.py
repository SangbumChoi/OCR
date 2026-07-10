"""Offline tests for the pseudo-labeling pipeline (docvlm_eval.unified.pseudo_label).

Uses a FAKE labeler — the design under test is the pipeline semantics (plan without models,
fill-only-empty, provenance), not any OCR model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.unified.pseudo_label import FILLERS, apply, plan


def _mini():
    from datasets import Dataset
    return Dataset.from_list([
        {"sample_id": "iam_0000_0", "source": "iam", "task": "recognition",
         "full_text": "the quick brown fox", "table_html": "",
         "elements_json": "[]"},                                    # has gold transcript
        {"sample_id": "docvqa_0000_0", "source": "docvqa", "task": "vqa",
         "full_text": "", "table_html": "",
         "elements_json": "[]"},                                    # missing transcript
        {"sample_id": "doclaynet_0000_0", "source": "doclaynet", "task": "localization",
         "full_text": "", "table_html": "",
         "elements_json": json.dumps([{"key": "Table", "value": "", "kind": "region",
                                       "bbox": [0.1, 0.1, 0.5, 0.5, True]}])},  # textless region
        {"sample_id": "pubtabnet_0000_0", "source": "pubtabnet", "task": "table",
         "full_text": "", "table_html": "",
         "elements_json": "[]"},                                    # table row missing html
    ])


def test_plan_counts_without_models():
    rep = plan(_mini())
    assert rep["total_rows"] == 4
    assert rep["full_text"]["rows_needing_fill"] == 3           # all but the IAM gold
    assert rep["region_text"]["rows_needing_fill"] == 1         # the doclaynet textless box
    assert rep["table_html"]["rows_needing_fill"] == 1          # the table row
    assert "got-ocr2" in rep["full_text"]["suggested_models"]
    assert rep["full_text"]["by_source_top"]["docvqa"] == 1


def test_apply_fills_only_empty_and_records_provenance():
    ds = apply(_mini(), "full_text", labeler=lambda row: f"OCR({row['sample_id']})",
               name="fake-ocr")
    rows = {r["sample_id"]: r for r in ds}
    # gold untouched, empty filled, provenance recorded per row
    assert rows["iam_0000_0"]["full_text"] == "the quick brown fox"
    assert json.loads(rows["iam_0000_0"]["pseudo_json"]) == {}
    assert rows["docvqa_0000_0"]["full_text"] == "OCR(docvqa_0000_0)"
    assert json.loads(rows["docvqa_0000_0"]["pseudo_json"]) == {"full_text": "fake-ocr"}


def test_apply_labeler_can_skip():
    ds = apply(_mini(), "full_text", labeler=lambda row: None, name="noop")
    for r in ds:
        assert json.loads(r["pseudo_json"]) == {}               # nothing filled, nothing claimed


def test_normalization_standardizes_and_rejects():
    from docvlm_eval.unified.pseudo_label import normalize_table_html, normalize_text
    # chat wrappers stripped, whitespace runs collapsed, quotes dropped, NFC applied
    assert normalize_text('The text in the image reads: "TOTAL   25,000  KRW"') == "TOTAL 25,000 KRW"
    assert normalize_text("Here is the text:  Invoice\n\n\n\nNo. 42 ") == "Invoice\n\nNo. 42"
    # degenerate outputs are REJECTED (row stays unfilled), not written
    assert normalize_text("") is None
    assert normalize_text("I cannot read this image.") is None
    assert normalize_text("x" * 9000) is None
    # table filler keeps exactly the <table> block
    assert normalize_table_html("Sure! <table><tr><td>a</td></tr></table> enjoy") == \
        "<table><tr><td>a</td></tr></table>"
    assert normalize_table_html("there is no table here") is None


def test_apply_normalizes_fills_and_skips_rejections():
    ds = _mini()
    # a chatty model: row 2 gets a wrapped answer (normalized), row 4 a refusal (rejected)
    outputs = {"docvqa_0000_0": 'The text in the image says: "ACME  Corp."',
               "doclaynet_0000_0": "I'm sorry, I cannot help with that.",
               "pubtabnet_0000_0": "I cannot read this."}
    filled = apply(ds, "full_text", labeler=lambda r: outputs.get(r["sample_id"]),
                   name="mock-vlm")
    by_id = {filled["sample_id"][i]: i for i in range(len(filled))}
    ok = by_id["docvqa_0000_0"]
    assert filled["full_text"][ok] == "ACME Corp."                       # standardized, not raw
    assert json.loads(filled["pseudo_json"][ok]) == {"full_text": "mock-vlm"}
    rej = by_id["doclaynet_0000_0"]
    assert filled["full_text"][rej] == ""                                # refusal -> left empty
    assert json.loads(filled["pseudo_json"][rej]) == {}                  # and NO provenance
    gold = by_id["iam_0000_0"]
    assert filled["full_text"][gold] == "the quick brown fox"            # gold never touched
