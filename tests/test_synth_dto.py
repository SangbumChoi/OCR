"""Synthetic-data DTO + GenConfig: ablation factors as GT, config-driven control.

Pure (schema/yaml only), so it runs without the [synth] render extra.
"""

import textwrap

import pytest

from docvlm_eval.synth.dto import BBox, DocSample, Field, GenConfig, QAItem, RenderSpec, script_for
from docvlm_eval.synth.to_samples import case_to_samples


def _doc(**kw) -> DocSample:
    base = dict(
        doc_id="inv0", doc_type="invoice", stressors=["table"], anchor_metric="KIE F1",
        fields=[
            Field("invoice_no", "INV-1", role="kie-value", bbox=BBox(1, 2, 3, 4), language="en"),
            Field("mrz", "P<USA...", role="mrz", language="en", font_px=10, is_small=True),
        ],
        qa=[QAItem("Total?", ["$10"], answer_type="H1", rationale="2+8=10",
                   answer_bbox=BBox(5, 6, 7, 8))],
        table_html="<table><tr><td>a</td></tr></table>",
        render=RenderSpec(size_px=[800, 600], dpi=150),
        languages=["en"],
    )
    base.update(kw)
    return DocSample(**base)


def test_to_dict_is_backward_compatible_superset():
    d = _doc().to_dict()
    # legacy flat keys the existing readers/tests rely on
    assert d["type"] == "invoice"
    assert d["fields"]["invoice_no"] == "INV-1"
    assert d["spotting"]["invoice_no"] == [1, 2, 3, 4]
    assert d["table_html"].startswith("<table>")
    assert d["qa"][0]["answers"] == ["$10"] and d["qa"][0]["answer_type"] == "H1"
    assert d["render"]["size_px"] == [800, 600]
    # structured view present alongside
    assert d["fields_detailed"][0]["bbox"] == [1, 2, 3, 4]
    assert "ablation_support" in d and d["render"]["aspect_ratio"] == round(800 / 600, 4)


def test_to_dict_feeds_existing_to_samples():
    samples = {s.sample_id: s for s in case_to_samples(_doc().to_dict(), "img.png", "inv")}
    assert "inv:qa0" in samples and samples["inv:qa0"].answer_type == "H1"
    assert samples["inv:spot:invoice_no"].metric == "grounding"   # A1 box -> grounding sample
    assert samples["inv:table"].metric == "teds"


def test_ablation_support_flags_track_content():
    sup = _doc().support()
    assert sup.spotting and sup.rationale and sup.small_text and sup.table
    assert not sup.multilingual
    # turning the supervision OFF (control arm) clears the flags
    bare = _doc(fields=[Field("x", "v")], qa=[QAItem("q", ["a"])])
    s2 = bare.support()
    assert not s2.spotting and not s2.rationale and not s2.small_text


def test_multilingual_flag_and_script_mapping():
    doc = _doc(fields=[Field("a", "v", language="en"), Field("b", "v2", language="ko",
                                                             script=script_for("ko"))],
               languages=["en", "ko"])
    assert doc.support().multilingual
    assert script_for("ko") == "hangul" and script_for("ar") == "arabic" and script_for("x") == "latin"


def test_from_builder_gt_upgrades_flat_gt():
    flat = {
        "type": "invoice", "stressors": ["t"], "anchor_metric": "F1",
        "fields": {"invoice_no": "INV-1", "_task": "ignore me"},
        "spotting": {"invoice_no": [1, 2, 3, 4], "total_region": [9, 9, 19, 19]},
        "qa": [{"key": "invoice_no", "question": "No?", "answers": ["INV-1"],
                "answer_type": "T2", "metric": "ned", "rationale": "read it"}],
        "render": {"dpi": 150, "size_px": [800, 600], "page_count": 1},
    }
    doc = DocSample.from_builder_gt(flat)
    by_key = {f.key: f for f in doc.fields}
    assert by_key["invoice_no"].bbox.to_list() == [1, 2, 3, 4]
    assert "_task" not in by_key                       # private fields dropped
    assert by_key["total_region"].role == "region"     # box without a field still carried
    assert doc.qa[0].answer_bbox.to_list() == [1, 2, 3, 4] and doc.qa[0].rationale == "read it"


YAML = textwrap.dedent("""
    base:
      name: base
      dpi: 150
      emit_spotting: true
      emit_rationale: true
      languages: [en]
    ablation_overrides:
      A1_spotting_off: {emit_spotting: false, emit_rationale: false}
      A7_highres: {target_long_side: 1536, dpi: 200}
      A4_ko_en: {languages: [ko, en]}
""")


def test_genconfig_from_yaml_base_and_overrides(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(YAML)
    base = GenConfig.from_yaml(str(p))
    assert base.name == "base" and base.emit_spotting and base.dpi == 150

    a1 = GenConfig.from_yaml(str(p), ablation="A1_spotting_off")
    assert a1.ablation == "A1_spotting_off" and not a1.emit_spotting and not a1.emit_rationale
    assert a1.dpi == 150                       # untouched knobs inherit from base

    a7 = GenConfig.from_yaml(str(p), ablation="A7_highres")
    assert a7.target_long_side == 1536 and a7.dpi == 200 and a7.emit_spotting  # only A7 knobs change

    a4 = GenConfig.from_yaml(str(p), ablation="A4_ko_en")
    assert a4.languages == ["ko", "en"]

    with pytest.raises(KeyError):
        GenConfig.from_yaml(str(p), ablation="nope")


def test_graph_evidence_keys_resolve_to_multiple_boxes():
    gt = {
        "type": "hard table",
        "stressors": ["multi-cell"],
        "anchor_metric": "relaxed_acc",
        "fields": {"left": "10", "right": "20"},
        "spotting": {"left": [1, 2, 3, 4], "right": [5, 6, 7, 8]},
        "qa": [
            {
                "question": "What is the total?",
                "answers": ["30"],
                "metric": "relaxed_acc",
                "answer_type": "H-table",
                "evidence_keys": ["left", "right"],
            }
        ],
        "render": {"dpi": 150, "size_px": [100, 100], "page_count": 1},
    }
    doc = DocSample.from_builder_gt(gt)
    assert doc.qa[0].evidence_keys == ["left", "right"]
    assert [box.to_list() for box in doc.qa[0].evidence_bboxes] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    detailed = doc.to_dict()["qa_detailed"][0]
    assert detailed["evidence_bboxes"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
