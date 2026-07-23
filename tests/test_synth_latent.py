import random

import pytest

from docvlm_eval.synth.hard_cases import HARD_CASE_FACTORIES
from docvlm_eval.synth.latent import (
    GraphEdge,
    GraphNode,
    GraphQuery,
    LatentDocumentGraph,
)
from docvlm_eval.synth.splits import SplitPolicy, validate_split_leakage


def _simple_graph(value: float, *, expected: str | None = None) -> LatentDocumentGraph:
    return LatentDocumentGraph(
        graph_id=f"g-{value}",
        template_family="simple-v1",
        nodes=[
            GraphNode("a", "cell", value, "A", attributes={"field_key": "a_cell"}),
            GraphNode("b", "cell", 2, "B", attributes={"field_key": "b_cell"}),
        ],
        queries=[
            GraphQuery(
                "total",
                "Total?",
                "sum",
                ("a", "b"),
                "H-sum",
                evidence=("a", "b"),
                answer_format="decimal",
                expected=expected,
            )
        ],
    )


def test_graph_recomputes_answer_rationale_and_evidence():
    graph = _simple_graph(3, expected="5")
    resolved = graph.resolve("total")
    assert resolved.answer == "5"
    assert "3.0, 2.0" in resolved.rationale
    assert resolved.evidence_keys == ("a_cell", "b_cell")


def test_graph_rejects_stale_expected_answer():
    with pytest.raises(ValueError, match="recomputed"):
        _simple_graph(3, expected="6")


def test_template_fingerprint_excludes_values_but_content_fingerprint_does_not():
    first = _simple_graph(3)
    second = _simple_graph(9)
    assert first.template_fingerprint == second.template_fingerprint
    assert first.content_fingerprint != second.content_fingerprint


def test_weighted_paths_are_executable():
    graph = LatentDocumentGraph(
        graph_id="ownership",
        template_family="ownership-v1",
        nodes=[
            GraphNode("a", "entity", "A"),
            GraphNode("b", "entity", "B"),
            GraphNode("c", "entity", "C"),
        ],
        edges=[
            GraphEdge("ab", "a", "owns", "b", 0.5),
            GraphEdge("bc", "b", "owns", "c", 0.4),
        ],
        queries=[
            GraphQuery(
                "indirect",
                "Indirect ownership?",
                "path_product",
                ("ab", "bc"),
                "H-finance-path",
                answer_format="fraction_percent",
            )
        ],
    )
    assert graph.resolve("indirect").answer == "20.00%"


@pytest.mark.parametrize("level", [1, 3, 5])
def test_hard_case_programs_validate_at_every_curriculum_level(level):
    for name, factory in HARD_CASE_FACTORIES.items():
        case = factory(random.Random(101), level)
        graph = case.builder.semantic_graph
        assert graph["queries"], name
        assert graph["difficulty"]["level"] == level
        assert len(graph["content_fingerprint"]) == 64
        for query in graph["queries"]:
            assert query["resolved"]["answer"]
            assert query["resolved"]["rationale"]


def test_split_policy_is_deterministic_and_content_sensitive():
    policy = SplitPolicy(seed=11)
    first = {"doc_id": "a", "semantic_graph": _simple_graph(3).to_dict()}
    second = {"doc_id": "b", "semantic_graph": _simple_graph(9).to_dict()}
    assert policy.assign(first) == policy.assign(first)
    assert (
        first["semantic_graph"]["content_fingerprint"]
        != second["semantic_graph"]["content_fingerprint"]
    )


def test_split_validator_rejects_cross_split_content_leakage():
    graph = _simple_graph(3).to_dict()
    records = [
        {"split": "train", "semantic_graph": graph},
        {"split": "heldout", "semantic_graph": graph},
    ]
    with pytest.raises(ValueError, match="semantic content leakage"):
        validate_split_leakage(records)


def test_split_validator_reports_template_overlap_separately():
    records = [
        {"split": "train", "semantic_graph": _simple_graph(3).to_dict()},
        {"split": "heldout", "semantic_graph": _simple_graph(9).to_dict()},
    ]
    report = validate_split_leakage(records)
    assert report["unique_content"] == 2
    assert report["template_overlap_count"] == 1
    with pytest.raises(ValueError, match="template leakage"):
        validate_split_leakage(records, require_template_isolation=True)
