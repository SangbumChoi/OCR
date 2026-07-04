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
