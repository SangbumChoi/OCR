from pathlib import Path

from PIL import Image

from docvlm_eval.unified import Task, UnifiedSample, render_detail_report


def test_detail_report_sanitizes_tables_and_preserves_full_page(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (600, 1400), "white").save(image_path)
    table_html = (
        "<table><tr><th>Item</th><th>Value</th></tr>"
        "<tr><td rowspan='2' onclick='bad()'>A</td><td>1</td></tr>"
        "<tr><td><script>alert(1)</script>2</td></tr></table>"
    )
    row = UnifiedSample(
        sample_id="complex-table",
        source="fixture",
        task=Task.TABLE,
        table_html=table_html,
        image_path=str(image_path),
    )
    output = tmp_path / "details.html"

    assert render_detail_report([row], str(output), max_long_side=800) == str(output)

    report = output.read_text(encoding="utf-8")
    assert "600x1400 px | 3 rows | 5 cells" in report
    assert "rowspan=\"2\"" in report
    assert "onclick" not in report
    assert "<script>" not in report
    assert "alert(1)" not in report
    assert ">2</td>" in report
    assert table_html not in report
    assert report.count("data:image/jpeg;base64,") == 1


def test_detail_report_skips_non_table_landscape_thumbnail(tmp_path: Path):
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (800, 300), "white").save(image_path)
    row = UnifiedSample(
        sample_id="crop",
        source="fixture",
        task=Task.RECOGNITION,
        full_text="short",
        image_path=str(image_path),
    )
    output = tmp_path / "details.html"

    assert render_detail_report([row], str(output)) == str(output)
    assert not output.exists()


def test_detail_report_validates_resource_limits(tmp_path: Path):
    output = tmp_path / "details.html"

    try:
        render_detail_report([], str(output), max_samples=0)
    except ValueError as exc:
        assert str(exc) == "max_samples must be positive"
    else:
        raise AssertionError("expected invalid max_samples to fail")
