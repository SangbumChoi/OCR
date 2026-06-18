"""Markdown table prettifier."""

from docvlm_eval.report_md import prettify_tables


def test_columns_aligned():
    raw = "| a | bbbb |\n|---|---|\n| 1 | 2 |\n"
    out = prettify_tables(raw)
    lines = out.strip().split("\n")
    # every row has the same length once padded
    assert len({len(l) for l in lines}) == 1
    assert "bbbb" in out


def test_non_table_untouched():
    raw = "# Title\n\nsome text\n\n- bullet\n"
    assert prettify_tables(raw) == raw


def test_ragged_rows_padded():
    raw = "| a | b | c |\n|---|---|---|\n| 1 |\n| x | y | z |\n"
    out = prettify_tables(raw)
    lines = [l for l in out.split("\n") if l.strip()]
    assert len({len(l) for l in lines}) == 1  # short row padded to full width


def test_preserves_table_content():
    raw = "| model | score |\n|---|---|\n| internvl3-1b | 0.81 |\n"
    out = prettify_tables(raw)
    assert "internvl3-1b" in out and "0.81" in out and out.count("|") == raw.count("|")
