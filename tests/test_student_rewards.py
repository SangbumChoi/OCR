import pytest


def _config(rationale_verifier="evidence_semantic"):
    from docvlm_eval.student.rewards import RewardConfig

    return RewardConfig(
        weights={
            "answer_correctness": 0.25,
            "normalized_text_similarity": 0.15,
            "box_iou": 0.15,
            "table_tree_similarity": 0.10,
            "chart_numeric_tolerance": 0.10,
            "formula_equivalence": 0.10,
            "grounded_rationale_consistency": 0.10,
            "calibrated_abstention": 0.05,
        },
        rationale_verifier=rationale_verifier,
    )


def _program_context():
    from docvlm_eval.student.rewards import RewardContext
    from docvlm_eval.synth.latent import (
        GraphNode,
        GraphQuery,
        LatentDocumentGraph,
    )

    graph = LatentDocumentGraph(
        graph_id="reward-trace",
        template_family="reward-trace-v1",
        nodes=[
            GraphNode("left", "cell", 10, "left"),
            GraphNode("right", "cell", 20, "right"),
        ],
        queries=[
            GraphQuery(
                "total",
                "What is the total?",
                "sum",
                ("left", "right"),
                "H-sum",
                answer_format="integer",
            )
        ],
    )
    resolved = graph.resolve("total")
    return RewardContext(
        sample_id="program-trace",
        answers=(resolved.answer,),
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
        gold_rationale=resolved.rationale,
        reasoning_trace=resolved.reasoning_trace,
    )


def test_structured_response_parser_enforces_the_grounded_contract():
    from docvlm_eval.student.rewards import (
        StructuredResponse,
        parse_structured_response,
    )

    response = StructuredResponse(
        answer="Revenue is 42",
        evidence=((0.1, 0.2, 0.5, 0.6),),
        rationale="The cited cell contains 42.",
    )

    assert parse_structured_response(response.to_json()) == response
    with pytest.raises(ValueError, match="one JSON object"):
        parse_structured_response("Revenue is 42")
    with pytest.raises(ValueError, match="normalized"):
        parse_structured_response(
            '{"answer":"42","evidence":[[10,20,30,40]],"rationale":""}'
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_structured_response('{"answer":"42","confidence":1}')
    with pytest.raises(ValueError, match="must contain"):
        parse_structured_response('{"answer":"42","evidence":[]}')


def test_malformed_recovery_is_dense_but_cannot_cross_the_structure_gate():
    from docvlm_eval.student.rewards import (
        RewardConfig,
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    context = RewardContext(sample_id="recovery", answers=("42",))
    target = build_structured_target("42")
    config = RewardConfig(
        weights={"answer_correctness": 1.0},
        malformed_recovery_max=0.1,
    )

    close = score_structured_response(target[:-1], context, config)
    far = score_structured_response("not-json", context, config)
    valid_wrong = score_structured_response(
        build_structured_target("17"),
        context,
        config,
    )

    assert not close.structurally_valid
    assert not far.structurally_valid
    assert close.total > far.total
    assert close.components["malformed_recovery_similarity"] > far.components[
        "malformed_recovery_similarity"
    ]
    assert close.total < 0.1
    assert valid_wrong.structurally_valid
    assert valid_wrong.total == pytest.approx(0.1)
    assert valid_wrong.total > close.total


def test_reward_config_rejects_an_unbounded_malformed_recovery():
    from docvlm_eval.student.rewards import RewardConfig

    with pytest.raises(ValueError, match="malformed_recovery_max"):
        RewardConfig(
            weights={"answer_correctness": 1.0},
            malformed_recovery_max=0.3,
        )
    with pytest.raises(ValueError, match="recovery ceiling"):
        RewardConfig(
            weights={"answer_correctness": 1.0},
            malformed_reward=0.9,
            malformed_recovery_max=0.1,
        )


def test_chart_grounding_and_rationale_rewards_are_independently_reported():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    context = RewardContext(
        sample_id="chart-1",
        answers=("42",),
        metric="relaxed_acc",
        answer_type="chart-numeric",
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
        gold_rationale="Read the value from the highlighted bar.",
        chart_expected=True,
    )
    response = build_structured_target(
        "42.1",
        evidence=((0.1, 0.2, 0.5, 0.6),),
        rationale="Read the value from the highlighted bar.",
    )

    result = score_structured_response(response, context, _config())

    assert result.structurally_valid
    assert result.components["answer_correctness"] == 0.0
    assert result.components["chart_numeric_tolerance"] == 1.0
    assert result.components["box_iou"] == pytest.approx(1.0)
    assert result.components["rationale_text_similarity"] == pytest.approx(1.0)
    assert result.components["grounded_rationale_consistency"] == pytest.approx(1.0)
    assert result.components["calibrated_abstention"] == 1.0
    assert 0 < result.total < 1


def test_grounded_rationale_rejects_arbitrary_nonempty_text():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    context = RewardContext(
        sample_id="chart-rationale-hack",
        answers=("42",),
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
        gold_rationale="Read the value from the highlighted bar.",
    )
    grounded = score_structured_response(
        build_structured_target(
            "42",
            evidence=context.gold_boxes,
            rationale=context.gold_rationale,
        ),
        context,
        _config(),
    )
    arbitrary = score_structured_response(
        build_structured_target(
            "42",
            evidence=context.gold_boxes,
            rationale="This sentence is intentionally unrelated.",
        ),
        context,
        _config(),
    )

    assert grounded.components["grounded_rationale_consistency"] == 1.0
    assert arbitrary.components["rationale_text_similarity"] == 0.0
    assert arbitrary.components["grounded_rationale_consistency"] == 0.0
    assert arbitrary.total < grounded.total


def test_reward_config_rejects_an_unverified_rationale_method():
    from docvlm_eval.student.rewards import RewardConfig

    with pytest.raises(ValueError, match="rationale_verifier"):
        RewardConfig(
            weights={"answer_correctness": 1.0},
            rationale_verifier="nonempty",
        )


def test_program_trace_verifier_rewards_exact_intermediate_facts():
    from docvlm_eval.student.rewards import (
        build_structured_target,
        score_structured_response,
    )

    context = _program_context()
    result = score_structured_response(
        build_structured_target(
            "30",
            evidence=context.gold_boxes,
            rationale=context.gold_rationale,
        ),
        context,
        _config("evidence_program_trace"),
    )

    assert result.components["rationale_program_fact_score"] == 1.0
    assert result.components["program_trace_consistency"] == 1.0
    assert result.components["grounded_rationale_consistency"] == 1.0


def test_program_trace_verifier_penalizes_wrong_or_hallucinated_numbers():
    from docvlm_eval.student.rewards import (
        build_structured_target,
        score_structured_response,
    )

    context = _program_context()
    exact = score_structured_response(
        build_structured_target(
            "30",
            evidence=context.gold_boxes,
            rationale=context.gold_rationale,
        ),
        context,
        _config("evidence_program_trace"),
    )
    wrong = score_structured_response(
        build_structured_target(
            "30",
            evidence=context.gold_boxes,
            rationale="Add 10 and 20 to obtain 31 after adjustment by 7.",
        ),
        context,
        _config("evidence_program_trace"),
    )

    assert wrong.components["rationale_program_fact_score"] < 1.0
    assert (
        wrong.components["grounded_rationale_consistency"]
        < exact.components["grounded_rationale_consistency"]
    )
    assert wrong.total < exact.total


def test_program_trace_verifier_accepts_fraction_percent_equivalence():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )
    from docvlm_eval.synth.latent import (
        GraphEdge,
        GraphNode,
        GraphQuery,
        LatentDocumentGraph,
    )

    graph = LatentDocumentGraph(
        graph_id="percent-trace",
        template_family="percent-trace-v1",
        nodes=[
            GraphNode("a", "stage", "A"),
            GraphNode("b", "stage", "B"),
            GraphNode("c", "stage", "C"),
        ],
        edges=[
            GraphEdge("ab", "a", "passes", "b", 0.5),
            GraphEdge("bc", "b", "passes", "c", 0.4),
        ],
        queries=[
            GraphQuery(
                "yield",
                "Yield?",
                "path_product",
                ("ab", "bc"),
                "H-path",
                answer_format="fraction_percent",
            )
        ],
    )
    resolved = graph.resolve("yield")
    context = RewardContext(
        sample_id="percent-trace",
        answers=(resolved.answer,),
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
        gold_rationale=resolved.rationale,
        reasoning_trace=resolved.reasoning_trace,
    )
    result = score_structured_response(
        build_structured_target(
            resolved.answer,
            evidence=context.gold_boxes,
            rationale="50% times 40% gives 20%.",
        ),
        context,
        _config("evidence_program_trace"),
    )

    assert result.components["rationale_program_fact_score"] == 1.0

    wrong_scale = score_structured_response(
        build_structured_target(
            resolved.answer,
            evidence=context.gold_boxes,
            rationale="0.5% times 0.4% gives 0.2%.",
        ),
        context,
        _config("evidence_program_trace"),
    )
    assert wrong_scale.components["rationale_program_fact_score"] < 1.0


def test_program_trace_verifier_reexecutes_scientific_inference():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )
    from docvlm_eval.synth.latent import (
        GraphNode,
        GraphQuery,
        LatentDocumentGraph,
    )

    graph = LatentDocumentGraph(
        graph_id="scientific-reward-trace",
        template_family="scientific-reward-trace-v1",
        nodes=[
            GraphNode("treatment", "mean", 80, "Treatment mean"),
            GraphNode("control", "mean", 100, "Control mean"),
            GraphNode("treatment_se", "uncertainty", 3, "Treatment SE"),
            GraphNode("control_se", "uncertainty", 2, "Control SE"),
        ],
        queries=[
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
            )
        ],
    )
    resolved = graph.resolve("decision")
    context = RewardContext(
        sample_id="scientific-program-trace",
        answers=(resolved.answer,),
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
        gold_rationale=resolved.rationale,
        reasoning_trace=resolved.reasoning_trace,
    )

    result = score_structured_response(
        build_structured_target(
            resolved.answer,
            evidence=context.gold_boxes,
            rationale=resolved.rationale,
        ),
        context,
        _config("evidence_program_trace"),
    )

    assert result.components["rationale_program_fact_score"] == 1.0
    assert result.components["program_trace_consistency"] == 1.0
    assert result.components["grounded_rationale_consistency"] == 1.0


def test_program_trace_verifier_reexecutes_scientific_claim_consistency():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )
    from docvlm_eval.synth.latent import (
        GraphNode,
        GraphQuery,
        LatentDocumentGraph,
    )

    graph = LatentDocumentGraph(
        graph_id="scientific-claim-reward-trace",
        template_family="scientific-claim-reward-trace-v1",
        nodes=[
            GraphNode("treatment", "mean", 80, "Treatment mean"),
            GraphNode("control", "mean", 100, "Control mean"),
            GraphNode("treatment_se", "uncertainty", 3, "Treatment SE"),
            GraphNode("control_se", "uncertainty", 2, "Control SE"),
            GraphNode("reported_claim", "claim", 0, "Unsupported claim"),
        ],
        queries=[
            GraphQuery(
                "claim_consistency",
                "Is the claim consistent?",
                "significance_claim_consistency",
                (
                    "treatment",
                    "control",
                    "treatment_se",
                    "control_se",
                    "reported_claim",
                ),
                "H-science-claim-verification",
                metric="anls",
                answer_format="text",
                parameters={
                    "threshold": 1.96,
                    "claim_labels": ["not supported", "supported"],
                    "outputs": ["inconsistent", "consistent"],
                },
            )
        ],
    )
    resolved = graph.resolve("claim_consistency")
    context = RewardContext(
        sample_id="scientific-claim-program-trace",
        answers=(resolved.answer,),
        gold_boxes=(
            (0.05, 0.1, 0.2, 0.2),
            (0.25, 0.1, 0.4, 0.2),
            (0.45, 0.1, 0.6, 0.2),
            (0.65, 0.1, 0.8, 0.2),
            (0.1, 0.7, 0.9, 0.8),
        ),
        gold_rationale=resolved.rationale,
        reasoning_trace=resolved.reasoning_trace,
    )

    result = score_structured_response(
        build_structured_target(
            resolved.answer,
            evidence=context.gold_boxes,
            rationale=resolved.rationale,
        ),
        context,
        _config("evidence_program_trace"),
    )

    assert resolved.answer == "inconsistent"
    assert result.components["rationale_program_fact_score"] == 1.0
    assert result.components["program_trace_consistency"] == 1.0
    assert result.components["grounded_rationale_consistency"] == 1.0


def test_reward_context_rejects_a_tampered_program_trace():
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.rewards import RewardContext

    context = _program_context()
    tampered = dict(context.reasoning_trace)
    tampered["answer_value"] = 31
    sample = Sample(
        "tampered",
        "image.png",
        "Total?",
        ["30"],
        meta={"reasoning_trace": tampered},
    )

    with pytest.raises(ValueError, match="fingerprint"):
        RewardContext.from_sample(sample)


def test_box_reward_penalizes_extra_box_spraying():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    context = RewardContext(
        sample_id="grounding-1",
        answers=("42",),
        gold_boxes=((0.1, 0.2, 0.5, 0.6),),
    )
    exact = score_structured_response(
        build_structured_target(
            "42",
            evidence=((0.1, 0.2, 0.5, 0.6),),
        ),
        context,
        _config(),
    )
    sprayed = score_structured_response(
        build_structured_target(
            "42",
            evidence=(
                (0.1, 0.2, 0.5, 0.6),
                (0.0, 0.0, 0.1, 0.1),
                (0.8, 0.8, 1.0, 1.0),
            ),
        ),
        context,
        _config(),
    )

    assert exact.components["box_iou"] == pytest.approx(1.0)
    assert sprayed.components["box_iou"] < exact.components["box_iou"]
    assert sprayed.total < exact.total


def test_table_formula_and_abstention_rewards_use_task_specific_masks():
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    table = RewardContext(
        sample_id="table-1",
        answers=("<table><tr><td>A</td></tr></table>",),
        metric="teds",
        answer_type="table",
        table_expected=True,
    )
    table_result = score_structured_response(
        build_structured_target("<table><tr><td>A</td></tr></table>"),
        table,
        _config(),
    )
    assert table_result.components["table_tree_similarity"] == 1.0
    assert "formula_equivalence" not in table_result.applicable

    formula = RewardContext(
        sample_id="formula-1",
        answers=(r"\frac{a}{b}",),
        answer_type="formula",
        formula_expected=True,
    )
    formula_result = score_structured_response(
        build_structured_target(r"$\dfrac{a}{b}$"),
        formula,
        _config(),
    )
    assert formula_result.components["formula_equivalence"] == 1.0

    abstain = RewardContext(
        sample_id="absent-1",
        answers=("not present",),
        answer_type="probe:abstain",
        abstain_expected=True,
    )
    abstain_result = score_structured_response(
        build_structured_target("not present"),
        abstain,
        _config(),
    )
    assert abstain_result.components["calibrated_abstention"] == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "no aparece",
        "기재되어 있지 않음",
        "記載なし",
        "未提供",
    ],
)
def test_calibrated_abstention_accepts_supported_locales(answer):
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    result = score_structured_response(
        build_structured_target(answer),
        RewardContext(
            sample_id="localized-absence",
            answers=(answer,),
            answer_type="probe:abstain",
            abstain_expected=True,
        ),
        _config(),
    )

    assert result.components["calibrated_abstention"] == 1.0


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("x+x", "2x"),
        ("(a+b)^2", "a^2+2ab+b^2"),
        (r"\sin^2(x)+\cos^2(x)", "1"),
        ("x^2=1", "(x-1)(x+1)=0"),
        (r"$\dfrac{a}{b}$", r"\frac{a}{b}"),
    ],
)
def test_formula_equivalence_accepts_bounded_symbolic_rewrites(predicted, gold):
    pytest.importorskip("sympy")
    pytest.importorskip("antlr4")
    from docvlm_eval.student.rewards import formula_equivalent

    assert formula_equivalent(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("x^2+y^2", "(x+y)^2"),
        ("x=1", "x=2"),
        (r"\input{foo}", "foo"),
        (r"\int_0^1 x dx", r"\frac{1}{2}"),
        (r"\frac{", "x"),
        ("x" * 513, "x"),
    ],
)
def test_formula_equivalence_rejects_wrong_or_unbounded_expressions(
    predicted,
    gold,
):
    pytest.importorskip("sympy")
    pytest.importorskip("antlr4")
    from docvlm_eval.student.rewards import formula_equivalent

    assert not formula_equivalent(predicted, gold)


def test_symbolic_formula_equivalence_contributes_to_reward():
    pytest.importorskip("sympy")
    pytest.importorskip("antlr4")
    from docvlm_eval.student.rewards import (
        RewardContext,
        build_structured_target,
        score_structured_response,
    )

    result = score_structured_response(
        build_structured_target("(a+b)^2"),
        RewardContext(
            sample_id="formula-symbolic",
            answers=("a^2+2ab+b^2",),
            answer_type="formula",
            formula_expected=True,
        ),
        _config(),
    )

    assert result.components["answer_correctness"] == 0.0
    assert result.components["formula_equivalence"] == 1.0
    assert "formula_equivalence" in result.applicable


def test_reward_context_normalizes_authored_sample_evidence():
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.rewards import RewardContext

    sample = Sample(
        sample_id="qa-1",
        image_path="page.png",
        question="What is the total?",
        answers=["25"],
        answer_type="numeric",
        metric="relaxed_acc",
        meta={
            "box": [10, 20, 50, 60],
            "size": [100, 200],
            "rationale": "Read the total cell.",
        },
    )

    context = RewardContext.from_sample(sample)

    assert context.gold_boxes == ((0.1, 0.1, 0.5, 0.3),)
    assert context.chart_expected
    assert context.gold_rationale == "Read the total cell."


def test_reward_context_accepts_multiple_pixel_evidence_boxes():
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.rewards import RewardContext

    sample = Sample(
        "multi",
        "/tmp/image.png",
        "Question?",
        ["answer"],
        "H-table",
        "anls",
        {
            "size": [100, 200],
            "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        },
    )
    context = RewardContext.from_sample(sample)
    assert context.gold_boxes == (
        (0.1, 0.1, 0.3, 0.2),
        (0.5, 0.3, 0.7, 0.4),
    )


def test_malformed_output_is_gated_before_task_rewards():
    from docvlm_eval.student.rewards import RewardContext, score_structured_response

    result = score_structured_response(
        "The answer is 42.",
        RewardContext("sample", ("42",)),
        _config(),
    )

    assert result.total == 0.0
    assert not result.structurally_valid
    assert result.components == {}
