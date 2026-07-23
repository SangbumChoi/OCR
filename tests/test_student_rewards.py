import pytest


def _config():
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
        }
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
        rationale="The cited bar is labeled 42.1.",
    )

    result = score_structured_response(response, context, _config())

    assert result.structurally_valid
    assert result.components["answer_correctness"] == 0.0
    assert result.components["chart_numeric_tolerance"] == 1.0
    assert result.components["box_iou"] == pytest.approx(1.0)
    assert result.components["grounded_rationale_consistency"] == pytest.approx(1.0)
    assert result.components["calibrated_abstention"] == 1.0
    assert 0 < result.total < 1


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
