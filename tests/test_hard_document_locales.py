import copy
import random
from functools import lru_cache
from pathlib import Path

import pytest

from docvlm_eval.synth.dto import DocSample, GenConfig
from docvlm_eval.synth.hard_cases import HARD_CASE_FACTORIES
from docvlm_eval.synth.hard_locale import (
    HARD_DOCUMENT_LANGUAGES,
    hard_text,
    validate_hard_document_language,
    validate_hard_locale_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def _cached_render_record(name: str, language: str) -> dict:
    pytest.importorskip("weasyprint")
    case = HARD_CASE_FACTORIES[name](random.Random(101), 5, language)
    _, gt = case.builder.build(dpi=96)
    doc = DocSample.from_builder_gt(
        gt,
        builder=case.builder,
        gen_config=GenConfig(languages=[language]),
        domain=case.domain,
        acquisition=case.acquisition,
    )
    doc.languages = [language]
    return doc.to_dict()


def _render_record(name: str, language: str) -> dict:
    return copy.deepcopy(_cached_render_record(name, language))


def test_hard_locale_catalog_is_complete_and_rejects_unknown_languages():
    validate_hard_locale_catalog()
    assert set(HARD_DOCUMENT_LANGUAGES) == {"en", "es", "ko", "ja", "zh"}
    with pytest.raises(ValueError, match="must be one of"):
        hard_text("ar", "table_title")


def test_default_multilingual_mix_and_a4_uniform_override():
    config = ROOT / "configs" / "synth_data.yaml"
    base = GenConfig.from_yaml(str(config))
    a4 = GenConfig.from_yaml(str(config), ablation="A4_all")

    assert base.languages == list(HARD_DOCUMENT_LANGUAGES)
    assert sum((base.language_weights or {}).values()) == pytest.approx(1.0)
    assert a4.languages == list(HARD_DOCUMENT_LANGUAGES)
    assert a4.language_weights is None


@pytest.mark.parametrize("language", HARD_DOCUMENT_LANGUAGES)
@pytest.mark.parametrize("name", sorted(HARD_CASE_FACTORIES))
def test_hard_document_render_and_labels_share_one_locale(name, language):
    record = _render_record(name, language)

    validate_hard_document_language(record, language)

    assert record["languages"] == [language]
    assert {
        field["language"] for field in record["fields_detailed"]
    } == {language}
    assert {
        tuple(qa["languages"]) for qa in record["qa_detailed"]
    } == {(language,)}
    assert record["semantic_graph"]["language"] == language
    assert record["fields"]["full_text"]
    assert all(
        qa["graph_query_id"]
        for qa in record["qa"]
        if qa["answer_type"].startswith(("T-", "H-"))
    )
    assert any(probe["kind"] == "abstain" for probe in record["probes"])


def test_hard_document_numeric_programs_are_locale_invariant():
    query_answers = {}
    template_fingerprints = set()
    for language in HARD_DOCUMENT_LANGUAGES:
        record = _render_record("hard_chart", language)
        template_fingerprints.add(
            record["semantic_graph"]["template_fingerprint"]
        )
        query_answers[language] = {
            query["query_id"]: query["resolved"]["answer"]
            for query in record["semantic_graph"]["queries"]
        }

    assert len({_answers_key(value) for value in query_answers.values()}) == 1
    assert len(template_fingerprints) == 1


def test_hard_document_text_answers_are_localized():
    expected = {
        "en": "East",
        "es": "Este",
        "ko": "동부",
        "ja": "東部",
        "zh": "东部",
    }

    for language, answer in expected.items():
        record = _render_record("hard_table", language)
        answers = {
            query["query_id"]: query["resolved"]["answer"]
            for query in record["semantic_graph"]["queries"]
        }
        assert answers["largest_revenue"] == answer


def test_hard_document_template_fingerprints_ignore_locale():
    for name in HARD_CASE_FACTORIES:
        fingerprints = {
            _render_record(name, language)["semantic_graph"][
                "template_fingerprint"
            ]
            for language in HARD_DOCUMENT_LANGUAGES
        }
        assert len(fingerprints) == 1, name


def test_hard_document_language_validator_rejects_false_labels():
    record = _render_record("hard_chart", "en")
    record["languages"] = ["ko"]

    with pytest.raises(ValueError, match="field language mismatch"):
        validate_hard_document_language(record, "ko")


def _answers_key(answers: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(answers.items()))
