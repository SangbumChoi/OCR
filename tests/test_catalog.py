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


def test_all_ten_categories_covered():
    nums = set()
    for e in load_catalog():
        nums.add(int(str(e["category"]).split(".")[0]))
    assert nums == set(range(1, 11)), f"categories present: {sorted(nums)}"


def test_metric_names_recognisable():
    for e in load_catalog():
        m = e["metric"].lower()
        assert any(h in m for h in VALID_METRIC_HINTS), f"{e['key']} odd metric '{e['metric']}'"


def test_fetchable_entries_have_split():
    for e in load_catalog():
        if e.get("hf_id"):
            assert e.get("split"), f"{e['key']} has hf_id but no split"
