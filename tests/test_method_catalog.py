from copy import deepcopy
from pathlib import Path

from docvlm_eval.method_catalog import (
    EXPECTED_CATEGORY_COUNTS,
    load_method_catalog,
    render_method_survey,
    validate_method_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "frontier_method_catalog.jsonl"
REPORT = ROOT / "docs" / "report" / "frontier_method_survey.md"


def test_frontier_catalog_has_100_critical_method_records():
    rows = load_method_catalog(CATALOG)

    assert len(rows) == 100
    assert validate_method_catalog(rows) == []
    assert {row["category"] for row in rows} == set(EXPECTED_CATEGORY_COUNTS)
    assert all(row["benefit"] != row["limitation"] for row in rows)


def test_frontier_catalog_rejects_duplicate_and_shallow_records():
    rows = deepcopy(load_method_catalog(CATALOG))
    rows[1]["id"] = rows[0]["id"]
    rows[2]["benefit"] = "Too short"

    errors = validate_method_catalog(rows)

    assert any("duplicate method ids" in error for error in errors)
    assert any("benefit is too short" in error for error in errors)


def test_rendered_survey_contains_every_method_and_decision():
    rows = load_method_catalog(CATALOG)
    report = render_method_survey(rows)

    assert report.count("\n| V") >= 10
    assert "Recommended end-to-end stack" in report
    assert "Distilling Step-by-Step rationale supervision" in report
    assert "Financial programs" in report
    for row in rows:
        assert row["name"] in report


def test_checked_in_survey_matches_the_catalog():
    rows = load_method_catalog(CATALOG)

    assert REPORT.read_text(encoding="utf-8") == render_method_survey(rows)
