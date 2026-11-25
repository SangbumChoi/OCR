"""Synth pattern library: ground truth is produced by the render, consistently.

These need the [synth] extra (weasyprint + pymupdf); skipped cleanly when absent so the core
suite still runs. The contract under test: every declared value lands in GT, spotting boxes are
real and inside the image, tables emit valid TEDS-gold, checkboxes record the checked set, and
redacted values are GT-only (never drawn) so they can serve as abstain targets.
"""

import pytest

pytest.importorskip("weasyprint")
pytest.importorskip("fitz")

from docvlm_eval.synth import DocBuilder, render_html  # noqa: E402


def _inside(box, size):
    x1, y1, x2, y2 = box
    w, h = size
    return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_field_registers_value_and_spotbox_is_inside_image():
    b = DocBuilder("t", ["s"], "m")
    b.field("Invoice No", "INV-2025-0042", key="invoice_no", spot=True)
    img, gt = b.build(dpi=120)
    assert gt["fields"]["invoice_no"] == "INV-2025-0042"
    assert "invoice_no" in gt["spotting"]
    assert _inside(gt["spotting"]["invoice_no"], img.size)


def test_uniform_schema_keys():
    b = DocBuilder("t", ["s"], "m")
    b.field("A", "x", key="a")
    _, gt = b.build(dpi=100)
    for k in ("type", "stressors", "anchor_metric", "fields", "source", "render"):
        assert k in gt
    assert gt["render"]["size_px"] and gt["render"]["page_count"] >= 1


def test_table_emits_valid_teds_gold():
    b = DocBuilder("t", ["s"], "m")
    b.table(["Item", "Qty"], [["Widget", "2"], ["Cable", "5"]], key="lines")
    _, gt = b.build(dpi=100)
    assert gt["fields"]["lines_rows"] == 2
    h = gt["table_html"]
    assert h.startswith("<table>") and h.endswith("</table>")
    assert h.count("<tr>") == 3  # header + 2 body
    assert "Widget" in h and "Cable" in h


def test_checkboxes_record_checked_and_box_per_option():
    b = DocBuilder("t", ["s"], "m")
    b.checkboxes("contact", [("Email", True), ("SMS", False), ("Phone", True)])
    img, gt = b.build(dpi=120)
    assert gt["selection"]["contact"] == ["Email", "Phone"]
    for opt in ("Email", "SMS", "Phone"):
        assert f"contact:{opt}" in gt["spotting"]
        assert _inside(gt["spotting"][f"contact:{opt}"], img.size)


def test_redacted_value_is_gt_only_and_adds_abstain_probe():
    secret = "57975432319"
    b = DocBuilder("t", ["s"], "m")
    b.redaction("Account: ", secret, key="account_number")
    _, gt = b.build(dpi=120)
    # stored as an abstain target...
    assert gt["redacted"]["account_number"] == secret
    # ...but never written into the visible fields
    import json
    assert secret not in json.dumps(gt["fields"])
    # ...and an abstain probe was auto-added
    assert any(p["kind"] == "abstain" for p in gt["probes"])


def test_redacted_text_is_not_rendered_so_search_finds_nothing():
    secret = "ZZ-UNIQUE-SECRET-9931"
    rr = render_html(f"<p>visible <span style='background:#111;color:#111'>████</span></p>"
                     f"<!-- {secret} is only in a comment, never drawn -->", dpi=120)
    try:
        assert rr.search_boxes(secret) == []
        assert rr.search_boxes("visible")  # control: real text is found
    finally:
        rr.close()


def test_repeated_text_boxes_are_distinct_by_occurrence():
    b = DocBuilder("t", ["s"], "m")
    b.field(None, "DUP", key="first", spot=True)
    b.field(None, "DUP", key="second", spot=True)
    img, gt = b.build(dpi=120)
    assert "first" in gt["spotting"] and "second" in gt["spotting"]
    # different vertical positions -> the two occurrences resolved to different boxes
    assert gt["spotting"]["first"] != gt["spotting"]["second"]


def test_color_probe_recovers_repeated_spots_and_count_derivation(
    monkeypatch,
):
    from docvlm_eval.synth.render import RenderResult

    monkeypatch.setattr(
        RenderResult,
        "native_search_boxes",
        lambda _self, _text: [
            [1, 1, 2, 2],
            [3, 1, 4, 2],
            [5, 1, 6, 2],
        ],
    )
    target = "A&B <C>"
    b = DocBuilder("t", ["complex-text-layer"], "grounding")
    b.field(None, target, key="first", spot=True)
    b.field(None, target, key="second", spot=True)
    b.ask_count(target)

    img, gt = b.build(dpi=120, color_probe_fallback=True)

    assert _inside(gt["spotting"]["first"], img.size)
    assert _inside(gt["spotting"]["second"], img.size)
    assert gt["spotting"]["first"] != gt["spotting"]["second"]
    assert next(
        qa for qa in gt["qa"] if qa["answer_type"] == "H-count"
    )["answers"] == ["2"]
    assert gt["render"]["box_resolver"] == (
        "pdf_text_then_color_probe"
    )
    assert gt["render"]["color_probe_fallback_count"] == 1


def test_color_probe_never_exposes_comment_text(monkeypatch):
    from docvlm_eval.synth.render import RenderResult

    monkeypatch.setattr(
        RenderResult,
        "native_search_boxes",
        lambda _self, _text: [],
    )
    secret = "GT-ONLY-SECRET"
    b = DocBuilder("t", ["abstain"], "grounding")
    b.raw(f"<p>visible</p><!-- {secret} -->")
    b.spot("secret", secret)

    _, gt = b.build(dpi=100, color_probe_fallback=True)

    assert "spotting" not in gt
    assert gt["render"]["color_probe_fallback_count"] == 0


def test_color_probe_supports_locate_and_region_derivations(
    monkeypatch,
):
    from docvlm_eval.synth.render import RenderResult

    monkeypatch.setattr(
        RenderResult,
        "native_search_boxes",
        lambda _self, _text: [],
    )
    b = DocBuilder("t", ["regions"], "grounding")
    b.field(None, "Left anchor", key="left")
    b.field(None, "Right anchor", key="right")
    b.ask_where("Left anchor")
    b.ask_region("the anchor region", ["Left anchor", "Right anchor"])

    img, gt = b.build(dpi=100, color_probe_fallback=True)

    locate = next(
        qa for qa in gt["qa"] if qa["answer_type"] == "L1-locate"
    )
    region = next(
        qa for qa in gt["qa"] if qa["answer_type"] == "L1-region"
    )
    assert _inside(locate["box"], img.size)
    assert _inside(region["box"], img.size)
    assert region["box"][0] <= locate["box"][0]
    assert region["box"][1] <= locate["box"][1]
    assert region["box"][2] >= locate["box"][2]
    assert region["box"][3] >= locate["box"][3]
    assert gt["render"]["color_probe_fallback_count"] == 2


def test_color_probe_can_be_disabled(monkeypatch):
    from docvlm_eval.synth.render import RenderResult

    monkeypatch.setattr(
        RenderResult,
        "native_search_boxes",
        lambda _self, _text: [],
    )
    b = DocBuilder("t", ["control"], "grounding")
    b.field(None, "unresolved", key="target", spot=True)

    _, gt = b.build(dpi=100, color_probe_fallback=False)

    assert "spotting" not in gt
    assert gt["render"]["box_resolver"] == "pdf_text"
    assert gt["render"]["color_probe_fallback_count"] == 0


def test_qa_records_answerable_pairs():
    b = DocBuilder("t", ["s"], "m")
    b.field("Total", "$145.50", key="total")
    b.qa("What is the total?", ["$145.50", "145.50"], answer_type="kie")
    _, gt = b.build(dpi=100)
    assert len(gt["qa"]) == 1
    qa = gt["qa"][0]
    assert qa["answers"] == ["$145.50", "145.50"] and qa["answer_type"] == "kie"
    assert qa["question"].startswith("What is the total?")  # concise suffix appended


def test_bubble_and_panel_build_reading_order():
    b = DocBuilder("t", ["s"], "m")
    b.bubble("hello", side="in")
    b.bubble("hi there", side="out")
    _, gt = b.build(dpi=100)
    assert gt["reading_order"] == ["hello", "hi there"]

    b2 = DocBuilder("t", ["s"], "m")
    b2.panel("p1", index=1)
    b2.panel("p2", index=2, side="right")
    _, gt2 = b2.build(dpi=100)
    assert gt2["reading_order"] == ["p1", "p2"]


def test_grounding_metric_question_has_no_concise_suffix():
    b = DocBuilder("t", ["s"], "m")
    b.qa("Return the bbox", "1,2,3,4", metric="grounding", answer_type="grounding")
    _, gt = b.build(dpi=100)
    assert gt["qa"][0]["question"] == "Return the bbox"  # no suffix for grounding/teds


def test_degrade_keeps_size_when_available():
    aug = pytest.importorskip("augraphy")  # noqa: F841
    from docvlm_eval.synth import degrade
    b = DocBuilder("t", ["s"], "m")
    b.field("A", "hello world", key="a")
    img, _ = b.build(dpi=100)
    out = degrade(img, "scan", seed=0)
    assert out is not None and out.size == img.size
