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
    assert s["inv:probe:abstain0"].meta["abstain_expected"] is True
    assert s["inv:probe:direction1"].meta["abstain_expected"] is False
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
    assert all(x.meta["generator_case"] == "invoice" for x in s)
    assert s[0].meta["doc_type"] == "invoice/receipt"


def test_meta_preserves_exact_generator_and_layout_identity():
    gt = {
        **GT,
        "generator_case": "hard_table",
        "render": {
            **GT["render"],
            "layout_family": "compact-v1",
        },
    }
    sample = case_to_samples(gt, "img.png", "hard_table_0007")[0]

    assert sample.meta["case"] == "hard_table_0007"
    assert sample.meta["generator_case"] == "hard_table"
    assert sample.meta["layout_family"] == "compact-v1"


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


def test_loader_labels_the_rendered_fallback_as_clean(tmp_path):
    case = tmp_path / "invoice"
    _write_case(case)
    (case / "clean.png").write_bytes(b"png")

    samples = load_realistic_samples(tmp_path, variant="degraded")

    assert samples[0].meta["render_variant"] == "clean"
    assert samples[0].meta["degradation"] == "clean"


def test_multiple_evidence_boxes_reach_posttraining_metadata():
    trace = {
        "schema_version": 1,
        "operation": "sum",
        "inputs": [],
        "parameters": {},
        "answer_value": 30,
        "answer": "30",
        "required_numeric_facts": [],
        "trace_fingerprint": "f" * 64,
    }
    gt = {
        "type": "hard table",
        "languages": ["en"],
        "render": {"size_px": [100, 200]},
        "qa": [
            {
                "question": "Total?",
                "answers": ["30"],
                "answer_type": "H-table",
                "metric": "relaxed_acc",
                "rationale": "10 + 20 = 30.",
                "reasoning_trace": trace,
                "evidence_bboxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
            }
        ],
    }
    sample = case_to_samples(
        gt,
        "/tmp/hard.png",
        "hard",
        render_variant="degraded",
    )[0]
    assert sample.meta["boxes"] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert sample.meta["rationale"] == "10 + 20 = 30."
    assert sample.meta["reasoning_trace"] == trace
    assert sample.meta["document_family"] == "hard table"
    assert sample.meta["evidence_count"] == 2
    assert sample.meta["degradation"] == "degraded"


def _write_counterfactual_case(
    path,
    *,
    pair_id,
    role,
    answer,
):
    path.mkdir(parents=True)
    gt = {
        "type": "hard chart",
        "languages": ["en"],
        "counterfactual": {
            "pair_id": pair_id,
            "role": role,
            "edit_scope": "latent_values",
        },
        "qa": [
            {
                "question": "What is the change?",
                "answers": [answer],
                "answer_type": "H-chart-change",
                "metric": "relaxed_acc",
                "graph_query_id": "change",
            }
        ],
        "render": {"size_px": [100, 100]},
    }
    (path / "gt.json").write_text(json.dumps(gt), encoding="utf-8")


def test_loader_keeps_only_answer_changing_counterfactual_pairs(tmp_path):
    _write_counterfactual_case(
        tmp_path / "hard_chart" / "0000",
        pair_id="hard_chart:0000",
        role="factual",
        answer="10",
    )
    _write_counterfactual_case(
        tmp_path / "hard_chart" / "0001",
        pair_id="hard_chart:0000",
        role="edited",
        answer="20",
    )
    _write_counterfactual_case(
        tmp_path / "hard_chart" / "0002",
        pair_id="hard_chart:0001",
        role="factual",
        answer="30",
    )
    _write_counterfactual_case(
        tmp_path / "hard_chart" / "0003",
        pair_id="hard_chart:0001",
        role="edited",
        answer="30",
    )

    samples = load_realistic_samples(tmp_path)
    by_id = {sample.sample_id: sample for sample in samples}
    changed = by_id["hard_chart_0000:qa0"]
    unchanged = by_id["hard_chart_0002:qa0"]

    assert changed.meta["counterfactual_group"] == "hard_chart:0000:change"
    assert changed.meta["counterfactual_eligible"] is True
    assert by_id["hard_chart_0001:qa0"].meta["control"] is True
    assert unchanged.meta["counterfactual_eligible"] is False
    assert "counterfactual_group" not in unchanged.meta
