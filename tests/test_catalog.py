"""Catalog integrity: required fields, unique keys, valid metric names, category coverage."""

from docvlm_eval.benchmarks.catalog import load_catalog

VALID_METRIC_HINTS = ("anls", "exact", "relaxed", "ocrbench", "f1", "teds", "edit",
                      "cer", "wer", "ned", "acc", "bleu", "cdm", "map", "ece", "grits",
                      "h-mean", "retention", "consistency", "rate")


def test_catalog_loads():
    cat = load_catalog()
    assert len(cat) >= 30


def test_required_fields_present():
    for e in load_catalog():
        for field in ("key", "name", "category", "metric", "purpose", "source"):
            assert field in e and e[field], f"{e.get('key')} missing {field}"


def test_keys_unique():
    keys = [e["key"] for e in load_catalog()]
    assert len(keys) == len(set(keys))


def test_purpose_is_descriptive():
    for e in load_catalog():
        assert len(e["purpose"]) > 15, f"{e['key']} purpose too short"


def test_all_families_covered():
    codes = {str(e["category"]).split(".")[0].strip() for e in load_catalog()}
    # the ten standard taxonomy task-codes must all be present (plus optional custom family F)
    required = {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "E1"}
    assert required.issubset(codes), f"codes present: {sorted(codes)}"


def test_metric_names_recognisable():
    for e in load_catalog():
        m = e["metric"].lower()
        assert any(h in m for h in VALID_METRIC_HINTS), f"{e['key']} odd metric '{e['metric']}'"


def test_fetchable_entries_have_split():
    for e in load_catalog():
        if e.get("hf_id"):
            assert e.get("split"), f"{e['key']} has hf_id but no split"
