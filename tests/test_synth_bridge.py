"""Offline tests for the synthetic-DTO -> UDD bridge (docvlm_eval.unified.synth_bridge).

Builds DocSample-shaped gt dicts (and a real typed DocSample) in memory — no rendering, no
network — and asserts the UnifiedSample carries EVERY annotation in UDD-compatible form, then that
the record passes the same to_hf_dataset/safety_check the public sources go through."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.unified import Task, docsample_to_unified, to_training_samples


def _gt(**over):
    """A representative persisted gt.json dict (DocSample.to_dict superset shape)."""
    gt = {
        "doc_id": "cheque-0001", "doc_type": "cheque", "type": "cheque",
        "domain": "finance", "acquisition": "scan", "stressors": ["handwriting"],
        "anchor_metric": "anls", "languages": ["en"],
        "fields_detailed": [
            {"key": "payee", "value": "John Smith", "bbox": [100, 50, 300, 80]},
            {"key": "amount", "value": "1,200.00", "bbox": [500, 50, 620, 80]},
            {"key": "memo", "value": "rent", "bbox": None},
        ],
        "qa_detailed": [
            {"question": "Who is the payee?", "answers": ["John Smith"], "metric": "anls",
             "key": "payee", "answer_bbox": [100, 50, 300, 80],
             "rationale": "The payee line is at the top-left; it reads 'John Smith'."},
            {"question": "What is the amount?", "answers": ["1,200.00", "1200.00"],
             "metric": "relaxed_acc", "key": "amount", "answer_bbox": None},
        ],
    }
    gt.update(over)
    return gt


def test_bridge_carries_every_annotation():
    r = docsample_to_unified(_gt(), image_path="/tmp/cheque.png", image_size=(1000, 500))
    # QAs: 2 real + 1 rationale-derived reasoning QA (A2 kept as data)
    assert len(r.qas) == 3 and not r.instruction              # grouped (XOR invariant holds)
    assert r.qas[1].question == "Who is the payee? Explain your reasoning."
    assert r.qas[1].answers[0].startswith("The payee line is at the top-left")
    assert r.qas[1].answers[0].endswith("So the answer is John Smith.")
    assert r.qas[2].answers == ["1,200.00", "1200.00"]        # variants preserved
    # fields: pixel boxes normalized by (1000, 500)
    payee = next(f for f in r.fields if f.key == "payee")
    assert payee.bbox.normalized and abs(payee.bbox.x2 - 0.3) < 1e-6 \
        and abs(payee.bbox.y2 - 0.16) < 1e-6
    assert next(f for f in r.fields if f.key == "memo").bbox is None
    # answer_bbox -> grounding region (A1)
    assert len(r.regions) == 1 and r.regions[0].label == "payee"
    assert r.source == "synthetic" and r.meta["synthetic"] is True
    assert r.meta["doc_type"] == "cheque" and r.split == "synthetic"


def test_bridge_single_qa_stays_flat_and_legacy_mirror_works():
    gt = _gt(qa_detailed=None)
    gt.pop("qa_detailed")
    gt["qa"] = [{"question": "Total?", "answers": ["$5"], "metric": "exact"}]
    gt.pop("fields_detailed")
    gt["fields"] = {"total": "$5"}
    gt["spotting"] = {"total": [10, 10, 50, 30]}
    r = docsample_to_unified(gt, image_path="/tmp/x.png", image_size=(100, 100))
    assert r.instruction == "Total?" and not r.qas            # single QA -> flat state
    assert r.fields[0].key == "total" and r.fields[0].bbox.normalized
    assert r.metric == "exact"


def test_bridge_typed_docsample_and_task_typing():
    from docvlm_eval.synth.dto import BBox, DocSample, Field, QAItem
    doc = DocSample(doc_id="t1", doc_type="table_doc", stressors=[], anchor_metric="teds",
                    table_html="<table><tr><td>a</td></tr></table>")
    r = docsample_to_unified(doc, image_path="/tmp/t.png", image_size=(200, 200))
    assert r.task == Task.TABLE and r.table_html.startswith("<table")
    doc2 = DocSample(doc_id="k1", doc_type="form", stressors=[], anchor_metric="anls",
                     fields=[Field(key="name", value="Bob", bbox=BBox(1, 2, 3, 4))])
    r2 = docsample_to_unified(doc2, image_path="/tmp/k.png", image_size=(10, 10))
    assert r2.task == Task.KIE and r2.fields[0].bbox.normalized
    doc3 = DocSample(doc_id="q1", doc_type="page", stressors=[], anchor_metric="anls",
                     qa=[QAItem(question="Q?", answers=["a"])])
    assert docsample_to_unified(doc3, "/tmp/q.png", (10, 10)).task == Task.VQA


def test_bridge_rejects_empty_and_passes_udd_safety_check(tmp_path):
    with pytest.raises(ValueError, match="no trainable annotation"):
        docsample_to_unified({"doc_id": "empty"}, "/tmp/e.png", (10, 10))
    # the bridged record must survive the SAME build/validation path as public sources
    from PIL import Image
    from docvlm_eval.unified import safety_check
    img = tmp_path / "cheque.png"
    Image.new("RGB", (1000, 500), "white").save(img)
    r = docsample_to_unified(_gt(), image_path=str(img), image_size=(1000, 500))
    rep = safety_check([r], str(tmp_path / "ds"))
    assert rep["rows"] == 1 and rep["fields"] == 3 and rep["regions"] == 1
    # and expand to trainable samples: 3 QAs sharing one image
    assert len(to_training_samples([r])) == 3
