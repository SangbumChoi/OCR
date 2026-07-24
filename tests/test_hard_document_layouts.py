"""Hard-document layouts vary pixels without changing authored semantics."""

import hashlib
import random
from functools import lru_cache

import pytest

from docvlm_eval.synth.dto import DocSample, GenConfig
from docvlm_eval.synth.hard_cases import HARD_CASE_FACTORIES
from docvlm_eval.synth.hard_layout import (
    HARD_LAYOUT_FAMILIES,
    hard_layout_spec,
    layout_fingerprint,
)
from docvlm_eval.synth.quality import audit_render_evidence


def _answers(graph: dict) -> dict[str, str]:
    return {
        query["query_id"]: query["resolved"]["answer"]
        for query in graph["queries"]
    }


@pytest.mark.parametrize("name", sorted(HARD_CASE_FACTORIES))
def test_layout_families_preserve_semantic_program_and_answers(name):
    cases = [
        HARD_CASE_FACTORIES[name](
            random.Random(101),
            5,
            "en",
            layout_family=layout,
        )
        for layout in HARD_LAYOUT_FAMILIES
    ]
    graphs = [case.builder.semantic_graph for case in cases]

    assert {case.layout_family for case in cases} == set(HARD_LAYOUT_FAMILIES)
    assert len({graph["content_fingerprint"] for graph in graphs}) == 1
    assert len({graph["template_fingerprint"] for graph in graphs}) == 1
    assert len({_answers_key(_answers(graph)) for graph in graphs}) == 1
    assert {
        graph["metadata"]["layout_family"] for graph in graphs
    } == set(HARD_LAYOUT_FAMILIES)
    assert len({tuple(case.builder._html) for case in cases}) == 3
    assert len({case.builder.page for case in cases}) >= 2


@pytest.mark.parametrize("layout", HARD_LAYOUT_FAMILIES)
def test_layout_registry_rejects_unknown_families_and_fingerprints(layout):
    spec = hard_layout_spec("hard_chart", layout)

    assert spec.family == layout
    assert len(layout_fingerprint("labelled temporal bar chart", layout)) == 64

    with pytest.raises(ValueError, match="unknown hard layout"):
        hard_layout_spec("hard_chart", "unknown")


@lru_cache(maxsize=None)
def _render(name: str, layout: str):
    pytest.importorskip("weasyprint")
    case = HARD_CASE_FACTORIES[name](
        random.Random(101),
        5,
        "en",
        layout_family=layout,
    )
    image, gt = case.builder.build(dpi=96)
    doc = DocSample.from_builder_gt(
        gt,
        builder=case.builder,
        gen_config=GenConfig(
            languages=["en"],
            hard_layout_families=[layout],
        ),
        domain=case.domain,
        acquisition=case.acquisition,
    )
    doc.languages = ["en"]
    return image, doc.to_dict(), tuple(key for key, _, _ in case.builder._spots)


@pytest.mark.parametrize("layout", HARD_LAYOUT_FAMILIES)
@pytest.mark.parametrize("name", sorted(HARD_CASE_FACTORIES))
def test_each_layout_renders_one_page_with_visible_spatial_targets(name, layout):
    image, record, required_spots = _render(name, layout)

    assert record["render"]["page_count"] == 1
    assert record["render"]["layout_family"] == layout
    assert record["render"]["layout_fingerprint"] == layout_fingerprint(
        record["type"],
        layout,
    )
    assert record["semantic_graph"]["metadata"]["layout_family"] == layout
    report = audit_render_evidence(
        image,
        record,
        required_spotting_keys=required_spots,
    )
    assert report["status"] == "passed"


@pytest.mark.parametrize("name", sorted(HARD_CASE_FACTORIES))
def test_three_layouts_produce_distinct_rasters(name):
    digests = {
        hashlib.sha256(
            f"{image.size}".encode("utf-8") + image.tobytes()
        ).hexdigest()
        for layout in HARD_LAYOUT_FAMILIES
        for image, _, _ in [_render(name, layout)]
    }

    assert len(digests) == len(HARD_LAYOUT_FAMILIES)


def test_compact_chart_normalizes_tall_random_walk_inside_page():
    pytest.importorskip("weasyprint")
    case = HARD_CASE_FACTORIES["hard_chart"](
        random.Random(1_131_227_940),
        5,
        "en",
        layout_family="compact-v1",
    )
    image, gt = case.builder.build(dpi=150)
    record = DocSample.from_builder_gt(
        gt,
        builder=case.builder,
        gen_config=GenConfig(
            languages=["en"],
            hard_layout_families=["compact-v1"],
        ),
        domain=case.domain,
        acquisition=case.acquisition,
    ).to_dict()

    report = audit_render_evidence(
        image,
        record,
        required_spotting_keys=tuple(key for key, _, _ in case.builder._spots),
    )
    assert report["status"] == "passed"


def _answers_key(answers: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(answers.items()))
