import random

import pytest

from docvlm_eval.synth.hard_cases import HARD_CASE_FACTORIES
from docvlm_eval.synth.latent import (
    GraphEdge,
    GraphNode,
    GraphQuery,
    LatentDocumentGraph,
)
from docvlm_eval.synth.splits import (
    SplitPolicy,
    build_record_semantic_identity,
    validate_split_leakage,
)


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
    assert resolved.reasoning_trace["operation"] == "sum"
    assert [
        item["value"] for item in resolved.reasoning_trace["inputs"]
    ] == [3, 2]
    assert resolved.reasoning_trace["answer_value"] == 5
    assert len(resolved.reasoning_trace["trace_fingerprint"]) == 64


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
    resolved = graph.resolve("indirect")
    assert resolved.answer == "20.00%"
    assert [item["input_type"] for item in resolved.reasoning_trace["inputs"]] == [
        "edge",
        "edge",
    ]
    assert resolved.reasoning_trace["answer_value"] == pytest.approx(0.2)


def test_scientific_interval_and_significance_programs_are_executable():
    graph = LatentDocumentGraph(
        graph_id="scientific-inference",
        template_family="scientific-inference-v1",
        nodes=[
            GraphNode("treatment", "mean", 80, "Treatment mean"),
            GraphNode("control", "mean", 100, "Control mean"),
            GraphNode("treatment_se", "uncertainty", 3, "Treatment SE"),
            GraphNode("control_se", "uncertainty", 2, "Control SE"),
        ],
        queries=[
            GraphQuery(
                "interval",
                "95% interval?",
                "confidence_interval",
                ("treatment", "treatment_se"),
                "H-science-confidence-interval",
                metric="anls",
                answer_format="text",
                parameters={
                    "critical_value": 1.96,
                    "decimal_places": 1,
                    "separator": " to ",
                },
            ),
            GraphQuery(
                "decision",
                "Supported difference?",
                "significance_decision",
                ("treatment", "control", "treatment_se", "control_se"),
                "H-science-inference",
                metric="anls",
                answer_format="text",
                parameters={
                    "threshold": 1.96,
                    "outputs": ["not supported", "supported"],
                },
            ),
        ],
    )

    interval = graph.resolve("interval")
    decision = graph.resolve("decision")

    assert interval.answer == "74.1 to 85.9"
    assert interval.reasoning_trace["answer_value"] == "74.1 to 85.9"
    assert "5.9" in interval.rationale
    assert decision.answer == "supported"
    assert decision.reasoning_trace["answer_value"] == "supported"
    assert "-5.55" in decision.rationale


def test_scientific_claim_consistency_program_checks_the_reported_claim():
    graph = LatentDocumentGraph(
        graph_id="scientific-claim-consistency",
        template_family="scientific-claim-consistency-v1",
        nodes=[
            GraphNode("treatment", "mean", 80, "Treatment mean"),
            GraphNode("control", "mean", 100, "Control mean"),
            GraphNode("treatment_se", "uncertainty", 3, "Treatment SE"),
            GraphNode("control_se", "uncertainty", 2, "Control SE"),
            GraphNode("claim_supported", "claim", 1, "Supported claim"),
            GraphNode("claim_unsupported", "claim", 0, "Unsupported claim"),
        ],
        queries=[
            GraphQuery(
                "consistent",
                "Is the claim consistent?",
                "significance_claim_consistency",
                (
                    "treatment",
                    "control",
                    "treatment_se",
                    "control_se",
                    "claim_supported",
                ),
                "H-science-claim-verification",
                metric="anls",
                answer_format="text",
                parameters={
                    "threshold": 1.96,
                    "claim_labels": ["not supported", "supported"],
                    "outputs": ["inconsistent", "consistent"],
                },
            ),
            GraphQuery(
                "inconsistent",
                "Is the claim consistent?",
                "significance_claim_consistency",
                (
                    "treatment",
                    "control",
                    "treatment_se",
                    "control_se",
                    "claim_unsupported",
                ),
                "H-science-claim-verification",
                metric="anls",
                answer_format="text",
                parameters={
                    "threshold": 1.96,
                    "claim_labels": ["not supported", "supported"],
                    "outputs": ["inconsistent", "consistent"],
                },
            ),
        ],
    )

    consistent = graph.resolve("consistent")
    inconsistent = graph.resolve("inconsistent")

    assert consistent.answer == "consistent"
    assert inconsistent.answer == "inconsistent"
    assert consistent.reasoning_trace["operation"] == (
        "significance_claim_consistency"
    )
    assert "data imply supported" in inconsistent.rationale
    assert "Results claim states not supported" in inconsistent.rationale


def test_scientific_generator_authors_both_inference_decisions():
    decisions = set()
    claim_decisions = set()
    for seed in range(32):
        case = HARD_CASE_FACTORIES["hard_science"](
            random.Random(seed),
            5,
            "en",
        )
        graph = case.builder.semantic_graph
        standard_errors = [
            row
            for row in graph["nodes"]
            if row["node_id"].startswith("se_")
        ]
        assert len({row["value"] for row in standard_errors}) == len(
            standard_errors
        )
        precise = next(
            row
            for row in graph["queries"]
            if row["query_id"] == "most_precise_condition"
        )
        minimum = min(standard_errors, key=lambda row: row["value"])
        expected_condition = minimum["label"].removesuffix(" standard error")
        assert precise["resolved"]["answer"] == expected_condition
        query = next(
            row
            for row in graph["queries"]
            if row["query_id"] == "compound_b_significance"
        )
        decision = query["resolved"]["answer"]
        decisions.add(decision)
        assert (
            decision == "supported"
        ) is graph["metadata"]["authored_compound_b_supported"]
        figure_query = next(
            row
            for row in graph["queries"]
            if row["query_id"] == "figure_b_reduction"
        )
        reduction = next(
            row
            for row in graph["nodes"]
            if row["node_id"] == "reduction_2"
        )
        assert figure_query["resolved"]["answer"] == f"{reduction['value']:.1f}"
        assert figure_query["resolved"]["evidence_keys"] == (
            "figure_effect_2",
        )
        claim_query = next(
            row
            for row in graph["queries"]
            if row["query_id"] == "compound_b_claim_consistency"
        )
        claim_decision = claim_query["resolved"]["answer"]
        claim_decisions.add(claim_decision)
        assert (
            claim_decision == "consistent"
        ) is graph["metadata"]["authored_claim_consistent"]
        assert len(claim_query["resolved"]["evidence_keys"]) == 5

    assert decisions == {"supported", "not supported"}
    assert claim_decisions == {"consistent", "inconsistent"}


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
            trace = query["resolved"]["reasoning_trace"]
            assert trace["operation"] == query["operation"]
            assert len(trace["trace_fingerprint"]) == 64


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


def test_graph_free_exact_records_receive_semantic_leakage_identity():
    record = {
        "generator_case": "audit_packet",
        "type": "three-page audit packet",
        "fields": {"purchase_order": "PO-123", "full_text": "rendered text"},
        "qa": [
            {
                "question": "What is the purchase order?",
                "answers": ["PO-123"],
                "metric": "exact",
                "answer_type": "kie",
            }
        ],
    }
    identity = build_record_semantic_identity(record)
    duplicate = {
        **record,
        "semantic_identity": identity,
        "split": "heldout",
    }

    assert identity["source"] == "exact_record_contract"
    assert len(identity["content_fingerprint"]) == 64
    with pytest.raises(ValueError, match="semantic content leakage"):
        validate_split_leakage(
            [
                {
                    **record,
                    "semantic_identity": identity,
                    "split": "train",
                },
                duplicate,
            ]
        )


def test_layout_split_policy_and_leakage_gate():
    fingerprint = "layout-a"
    first = {
        "doc_id": "a",
        "semantic_graph": _simple_graph(3).to_dict(),
        "render": {"layout_fingerprint": fingerprint},
    }
    second = {
        "doc_id": "b",
        "semantic_graph": _simple_graph(9).to_dict(),
        "render": {"layout_fingerprint": fingerprint},
    }
    policy = SplitPolicy(seed=11, group_by="layout")

    assert policy.assign(first) == policy.assign(second)

    records = [
        {**first, "split": "train"},
        {**second, "split": "heldout"},
    ]
    report = validate_split_leakage(records)
    assert report["unique_layouts"] == 1
    assert report["layout_overlap_count"] == 1
    with pytest.raises(ValueError, match="layout leakage"):
        validate_split_leakage(records, require_layout_isolation=True)


def test_layout_isolation_requires_layout_provenance():
    record = {
        "split": "train",
        "semantic_graph": _simple_graph(3).to_dict(),
    }

    with pytest.raises(ValueError, match="missing"):
        validate_split_leakage([record], require_layout_isolation=True)
