"""Multi-page synthesis preserves every page, text span, and spatial offset."""

from __future__ import annotations

import pytest

from docvlm_eval.synth.dto import DocSample
from docvlm_eval.synth.patterns import DocBuilder
from docvlm_eval.synth.render import render_html
from docvlm_eval.synth.to_samples import case_to_samples


def test_vertical_render_composes_all_pages_and_offsets_text_boxes():
    pytest.importorskip("weasyprint")
    pytest.importorskip("fitz")
    html = (
        "<section class=sheet>FIRST-PAGE-TARGET</section>"
        "<section>SECOND-PAGE-TARGET</section>"
    )
    css = (
        "@page{size:A6;margin:10mm}"
        ".sheet{break-after:page;min-height:100mm}"
    )
    result = render_html(html, css, dpi=96, page_mode="vertical")
    try:
        first = result.search_boxes("FIRST-PAGE-TARGET")
        second = result.search_boxes("SECOND-PAGE-TARGET")

        assert result.page_count == 2
        assert len(result.page_origins_px) == 2
        assert result.image.height == (
            result.page_sizes_px[0][1]
            + result.page_gap_px
            + result.page_sizes_px[1][1]
        )
        assert first and second
        assert first[0][1] < result.page_sizes_px[0][1]
        assert second[0][1] >= result.page_origins_px[1][1]
        assert result.full_text().splitlines() == [
            "FIRST-PAGE-TARGET",
            "SECOND-PAGE-TARGET",
        ]
    finally:
        result.close()


def test_grid_render_packs_three_pages_without_losing_page_identity():
    pytest.importorskip("weasyprint")
    pytest.importorskip("fitz")
    html = "".join(
        (
            f"<section class={'sheet' if index < 2 else 'last'}>"
            f"PAGE-{index + 1}-TARGET</section>"
        )
        for index in range(3)
    )
    css = (
        "@page{size:A6;margin:10mm}"
        ".sheet{break-after:page;min-height:100mm}"
    )
    result = render_html(html, css, dpi=96, page_mode="grid")
    try:
        assert result.page_count == 3
        assert result.image.width > result.page_sizes_px[0][0]
        assert result.image.height < 3 * result.page_sizes_px[0][1]
        assert result.page_origins_px[0][1] == result.page_origins_px[1][1]
        assert result.page_origins_px[2][1] > result.page_origins_px[0][1]
        for index in range(3):
            box = result.search_boxes(f"PAGE-{index + 1}-TARGET")[0]
            origin = result.page_origins_px[index]
            size = result.page_sizes_px[index]
            assert origin[0] <= box[0] < box[2] <= origin[0] + size[0]
            assert origin[1] <= box[1] < box[3] <= origin[1] + size[1]
    finally:
        result.close()


def test_complex_table_overflow_is_promoted_to_an_audited_all_page_canvas():
    pytest.importorskip("weasyprint")
    pytest.importorskip("fitz")
    pytest.importorskip("bs4")
    builder = DocBuilder(
        "overflow table fixture",
        ["complex-table", "full-page"],
        "teds",
        page="A6",
        margin="8mm",
    )
    rows = [
        [f"ROW-{index:02d}", f"VALUE-{index:02d}", f"NOTE-{index:02d}"]
        for index in range(48)
    ]
    builder.table(["Item", "Value", "Note"], rows)
    builder.want_fulltext()

    image, ground_truth = builder.build(dpi=72)

    render = ground_truth["render"]
    assert render["page_count"] > 1
    assert render["rendered_page_count"] == render["page_count"]
    assert render["page_mode"] == "vertical"
    assert render["auto_expanded_from_first"] is True
    assert render["layout_audit"]["status"] == "pass"
    assert render["layout_audit"]["missing_table_cells"] == []
    assert image.height > render["page_sizes_px"][0][1]


def test_builder_emits_cross_page_evidence_as_one_tall_document():
    pytest.importorskip("weasyprint")
    pytest.importorskip("fitz")
    builder = DocBuilder(
        "two-page fixture",
        ["multi-page", "cross-page-reasoning"],
        "exact",
        page="A6",
        page_mode="vertical",
        css=".sheet{break-after:page;min-height:100mm}",
    )
    builder.raw("<section class=sheet>")
    builder.field("First value", "ALPHA-17", key="first", spot=True)
    builder.raw("</section><section>")
    builder.field("Second value", "BETA-29", key="second", spot=True)
    builder.raw("</section>")
    builder.qa(
        "Combine the two values.",
        "ALPHA-17 BETA-29",
        metric="exact",
        answer_type="H-cross-page",
        evidence_keys=["first", "second"],
    )
    builder.want_fulltext()

    image, ground_truth = builder.build(dpi=96)
    structured = DocSample.from_builder_gt(
        ground_truth,
        builder=builder,
    ).to_dict()
    record = case_to_samples(
        structured,
        "fixture.png",
        "fixture",
        render_variant="clean",
    )
    cross_page = next(
        sample for sample in record if sample.answer_type == "H-cross-page"
    )

    assert structured["render"]["page_count"] == 2
    assert structured["render"]["rendered_page_count"] == 2
    assert structured["render"]["page_mode"] == "vertical"
    assert image.height > 2 * image.width
    assert cross_page.meta["evidence_pages"] == [0, 1]
    assert cross_page.meta["cross_page_evidence"] is True
    assert cross_page.meta["page_count"] == 2
