"""GT (gt.json) -> Sample conversion. Pure (schema only), so it runs without the [synth] extra."""

import json

from docvlm_eval.synth.to_samples import ABSTAIN_OK, case_to_samples, load_realistic_samples

GT = {
    "type": "invoice/receipt",
    "stressors": ["table"],
    "anchor_metric": "KIE F1",
    "fields": {"invoice_no": "INV-1"},
    "qa": [{"key": "invoice_no", "question": "What is the invoice number? Answer concisely.",
            "answers": ["INV-1"], "metric": "ned", "answer_type": "kie"}],
    "spotting": {"total": [10, 20, 30, 40]},
    "table_html": "<table><tr><td>a</td></tr></table>",
    "probes": [
        {"kind": "abstain", "question": "Tracking number?", "expected": "not present"},
        {"kind": "direction", "question": "RTL or LTR?", "expected": "right-to-left"},
        {"kind": "consistency", "question": "Agree?", "expected": "yes (1==1)"},
    ],
    "render": {"dpi": 150, "size_px": [800, 600], "page_count": 1},
}


def test_qa_spotting_table_and_probes_become_samples():
    s = case_to_samples(GT, "img.png", "invoice")
    kinds = {x.sample_id: x for x in s}
    # qa
    assert "invoice:qa0" in kinds
    assert kinds["invoice:qa0"].answers == ["INV-1"] and kinds["invoice:qa0"].metric == "ned"
    # spotting -> grounding, answer carries box + image size
    sp = kinds["invoice:spot:total"]
    assert sp.metric == "grounding" and sp.answers == ["10,20,30,40;800,600"]
    assert "800x600" in sp.question
    # table -> teds
    assert kinds["invoice:table"].metric == "teds"
    assert kinds["invoice:table"].answers == ["<table><tr><td>a</td></tr></table>"]


def test_probe_answer_mapping():
    s = {x.sample_id: x for x in case_to_samples(GT, "img.png", "inv")}
    assert s["inv:probe:abstain0"].answers == ABSTAIN_OK
    assert "rtl" in s["inv:probe:direction1"].answers
    assert "yes" in s["inv:probe:consistency2"].answers
    assert all(x.answer_type.startswith("probe:") for k, x in s.items() if ":probe:" in k)


def test_include_probes_toggle():
    with_p = case_to_samples(GT, "i.png", "p", include_probes=True)
    without_p = case_to_samples(GT, "i.png", "p", include_probes=False)
    assert len(with_p) - len(without_p) == 3  # the three probes


def test_meta_carries_case_context():
    s = case_to_samples(GT, "img.png", "invoice")
    assert all(x.meta["case"] == "invoice" for x in s)
    assert s[0].meta["doc_type"] == "invoice/receipt"


def test_handles_minimal_gt_without_optional_sections():
    minimal = {"type": "t", "qa": [{"question": "q", "answers": ["a"]}],
               "render": {"size_px": [100, 100]}}
    s = case_to_samples(minimal, "i.png", "m")
    assert len(s) == 1 and s[0].answer_type == "kie" and s[0].metric == "anls"


def _write_case(d, qa_q="q"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "gt.json").write_text(json.dumps(
        {"type": "t", "qa": [{"question": qa_q, "answers": ["a"]}],
         "render": {"size_px": [10, 10]}}))


def test_loader_handles_flat_and_variant_layouts(tmp_path):
    _write_case(tmp_path / "invoice")                 # flat (--count 1)
    _write_case(tmp_path / "id_card" / "0000")        # fanned out (--count N)
    _write_case(tmp_path / "id_card" / "0001")
    s = load_realistic_samples(tmp_path)
    ids = {x.sample_id for x in s}
    assert "invoice:qa0" in ids
    assert "id_card_0000:qa0" in ids and "id_card_0001:qa0" in ids  # prefixed by relative path
    assert len(s) == 3
