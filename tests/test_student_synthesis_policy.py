import json

import pytest

from docvlm_eval.student.synthesis_policy import (
    file_fingerprint,
    payload_fingerprint,
    plan_synthesis_batch,
    validate_generation_plan,
    validate_generation_plan_source,
)


def _config():
    return {
        "schema_version": 1,
        "budget": 20,
        "seed": 7,
        "failure_weights": {
            "score_deficit": 0.5,
            "reward_deficit": 0.3,
            "structure_failure": 0.2,
        },
        "prior_strength": 1.0,
        "uncertainty_coefficient": 0.0,
        "temperature": 0.1,
        "exploration_fraction": 0.1,
        "factor_weights": {
            "generator_case": 0.0,
            "language": 1.0,
            "difficulty_level": 0.0,
            "layout_family": 0.0,
            "composition_tier": 0.0,
        },
        "candidate_space": {
            "languages": ["en", "ko"],
            "difficulty_levels": [5],
            "cases": [
                {
                    "generator_case": "hard_table",
                    "composition_tier": "single_document",
                    "layout_families": ["compact-v1"],
                }
            ],
        },
        "generation": {"no_degrade": True},
    }


def _progress_config():
    config = _config()
    config.update(
        {
            "schema_version": 2,
            "require_matched_baseline": True,
            "learning_progress_coefficient": 0.5,
            "learning_progress_weights": {
                "score_gain": 0.5,
                "reward_gain": 0.3,
                "structure_gain": 0.2,
            },
        }
    )
    return config


def _reward_routed_config():
    config = _progress_config()
    config["schema_version"] = 3
    config["factor_weights"] = {
        "generator_case": 0.0,
        "language": 0.0,
        "difficulty_level": 0.0,
        "layout_family": 0.0,
        "composition_tier": 1.0,
    }
    config["candidate_space"]["languages"] = ["en"]
    config["candidate_space"]["cases"] = [
        {
            "generator_case": "hard_table",
            "composition_tier": "single_document",
            "layout_families": ["compact-v1"],
        },
        {
            "generator_case": "hard_chart",
            "composition_tier": "single_document",
            "layout_families": ["compact-v1"],
        },
    ]
    config["reward_routing"] = {
        "coefficient": 1.0,
        "prior_strength": 1.0,
        "components": {
            "structural_validity": {
                "weight": 1.0,
                "cases": ["*"],
            },
            "table_tree_similarity": {
                "weight": 1.0,
                "cases": ["hard_table"],
            },
            "chart_numeric_tolerance": {
                "weight": 1.0,
                "cases": ["hard_chart"],
            },
        },
    }
    return config


def _row(language, score, reward, valid, *, split="validation"):
    return {
        "sample_id": f"{split}-{language}",
        "split": split,
        "language": language,
        "score": score,
        "reward": reward,
        "structurally_valid": valid,
        "meta": {
            "generator_case": "hard_table",
            "language": language,
            "difficulty": {"level": 5},
            "layout_family": "compact-v1",
            "page_count": 1,
            "document_count": 1,
        },
    }


def _plan_with_baseline(rows, baseline_rows, config=None):
    return plan_synthesis_batch(
        rows,
        config or _progress_config(),
        source_fingerprint="sha256:current",
        source_path="/tmp/current/validation/per_sample.jsonl",
        baseline_rows=baseline_rows,
        baseline_source_fingerprint="sha256:baseline",
        baseline_source_path="/tmp/baseline/validation/per_sample.jsonl",
    )


def _reward_row(case, component, value):
    row = _row("en", 0.5, 0.5, True)
    row["sample_id"] = f"validation-{case}"
    row["meta"]["generator_case"] = case
    row["reward_components"] = {component: value}
    row["applicable_rewards"] = [component]
    return row


def test_validation_failures_deterministically_shift_the_next_batch():
    rows = [
        _row("en", 1.0, 1.0, True),
        _row("ko", 0.0, 0.0, False),
    ]
    first = plan_synthesis_batch(
        rows,
        _config(),
        source_fingerprint="sha256:evaluation",
        source_path="/tmp/validation/per_sample.jsonl",
    )
    second = plan_synthesis_batch(
        rows,
        _config(),
        source_fingerprint="sha256:evaluation",
        source_path="/tmp/validation/per_sample.jsonl",
    )

    assert first == second
    assert first["training_authorized"]
    assert sum(job["count"] for job in first["jobs"]) == 20
    counts = {job["language"]: job["count"] for job in first["jobs"]}
    assert counts["ko"] > counts["en"]
    assert first["factor_statistics"]["language"]["ko"]["mean_failure"] == 1.0
    validate_generation_plan(first, require_training_authorized=True)


def test_heldout_can_only_emit_non_executable_analysis():
    rows = [_row("ko", 0.0, 0.0, False, split="heldout")]
    with pytest.raises(ValueError, match="requires split='validation'"):
        plan_synthesis_batch(
            rows,
            _config(),
            source_fingerprint="sha256:heldout",
            source_path="/tmp/heldout/per_sample.jsonl",
        )

    analysis = plan_synthesis_batch(
        rows,
        _config(),
        source_fingerprint="sha256:heldout",
        source_path="/tmp/heldout/per_sample.jsonl",
        allow_heldout_analysis=True,
    )
    assert not analysis["training_authorized"]
    assert analysis["claim_scope"] == "heldout_analysis_only"
    with pytest.raises(ValueError, match="not authorized"):
        validate_generation_plan(
            analysis,
            require_training_authorized=True,
        )


def test_matched_learning_progress_shifts_equal_failure_arms():
    rows = [
        _row("en", 0.5, 0.5, True),
        _row("ko", 0.5, 0.5, True),
    ]
    baseline_rows = [
        _row("en", 0.0, 0.0, False),
        _row("ko", 0.5, 0.5, True),
    ]

    plan = _plan_with_baseline(rows, baseline_rows)

    counts = {job["language"]: job["count"] for job in plan["jobs"]}
    assert counts["en"] > counts["ko"]
    assert plan["schema_version"] == 2
    assert plan["policy"] == (
        "factor_shrinkage_learning_progress_curriculum"
    )
    assert plan["global_learning_progress"] > 0
    assert plan["baseline_source"]["rows"] == 2
    assert plan["matched_sample_ids_fingerprint"].startswith("sha256:")
    assert (
        plan["factor_statistics"]["language"]["en"][
            "mean_learning_progress"
        ]
        > 0
    )
    validate_generation_plan(plan, require_training_authorized=True)


def test_matched_learning_progress_requires_exact_sample_ids():
    rows = [_row("en", 0.5, 0.5, True)]
    baseline_rows = [_row("ko", 0.0, 0.0, False)]

    with pytest.raises(ValueError, match="sample_id set differs"):
        _plan_with_baseline(rows, baseline_rows)


def test_matched_learning_progress_requires_immutable_sample_identity():
    row = _row("en", 0.5, 0.5, True)
    baseline = _row("en", 0.0, 0.0, False)
    baseline["question"] = "A different benchmark question"

    with pytest.raises(ValueError, match="sample identity differs"):
        _plan_with_baseline([row], [baseline])


def test_decomposed_reward_deficit_routes_the_next_generator_family():
    rows = [
        _reward_row(
            "hard_table",
            "table_tree_similarity",
            0.0,
        ),
        _reward_row(
            "hard_chart",
            "chart_numeric_tolerance",
            1.0,
        ),
    ]
    baseline_rows = json.loads(json.dumps(rows))

    plan = _plan_with_baseline(
        rows,
        baseline_rows,
        _reward_routed_config(),
    )

    counts = {
        job["generator_case"]: job["count"] for job in plan["jobs"]
    }
    assert counts["hard_table"] > counts["hard_chart"]
    assert plan["schema_version"] == 3
    assert plan["policy"] == (
        "reward_routed_learning_progress_curriculum"
    )
    assert (
        plan["reward_component_statistics"]["table_tree_similarity"][
            "routed_utility"
        ]
        > plan["reward_component_statistics"]["chart_numeric_tolerance"][
            "routed_utility"
        ]
    )
    table_job = next(
        job
        for job in plan["jobs"]
        if job["generator_case"] == "hard_table"
    )
    assert table_job["dominant_reward_route"] == (
        "table_tree_similarity"
    )
    validate_generation_plan(plan, require_training_authorized=True)


def test_reward_routing_rejects_changed_baseline_applicability():
    row = _reward_row(
        "hard_table",
        "table_tree_similarity",
        0.0,
    )
    baseline = json.loads(json.dumps(row))
    baseline["applicable_rewards"] = ["chart_numeric_tolerance"]
    baseline["reward_components"] = {"chart_numeric_tolerance": 0.0}

    with pytest.raises(ValueError, match="applicability differs"):
        _plan_with_baseline(
            [row],
            [baseline],
            _reward_routed_config(),
        )


def test_structural_failure_routes_when_task_rewards_are_inapplicable():
    row = _row("en", 0.0, 0.0, False)
    row["applicable_rewards"] = []
    row["reward_components"] = {}
    baseline = json.loads(json.dumps(row))

    plan = _plan_with_baseline(
        [row],
        [baseline],
        _reward_routed_config(),
    )

    structural = plan["reward_component_statistics"][
        "structural_validity"
    ]
    assert structural["n"] == 1
    assert structural["mean_deficit"] == 1.0
    assert structural["routed_utility"] == 1.0
    assert all(
        job["reward_route_evidence_count"] >= 1
        for job in plan["jobs"]
    )


def test_reward_routing_rejects_resigned_job_tampering():
    rows = [
        _reward_row(
            "hard_table",
            "table_tree_similarity",
            0.0,
        ),
        _reward_row(
            "hard_chart",
            "chart_numeric_tolerance",
            1.0,
        ),
    ]
    plan = _plan_with_baseline(
        rows,
        json.loads(json.dumps(rows)),
        _reward_routed_config(),
    )
    tampered = json.loads(json.dumps(plan))
    tampered["jobs"][0]["dominant_reward_route"] = (
        "chart_numeric_tolerance"
    )
    unsigned = dict(tampered)
    unsigned.pop("plan_fingerprint")
    tampered["plan_fingerprint"] = payload_fingerprint(unsigned)

    with pytest.raises(ValueError, match="component statistics"):
        validate_generation_plan(tampered)


def test_schema_two_requires_a_matched_baseline():
    with pytest.raises(ValueError, match="requires matched baseline"):
        plan_synthesis_batch(
            [_row("en", 0.5, 0.5, True)],
            _progress_config(),
            source_fingerprint="sha256:current",
            source_path="/tmp/current/validation/per_sample.jsonl",
        )


def test_plan_fingerprint_rejects_content_tampering():
    plan = plan_synthesis_batch(
        [_row("en", 0.5, 0.5, True)],
        _config(),
        source_fingerprint="sha256:evaluation",
        source_path="/tmp/validation/per_sample.jsonl",
    )
    serialized = json.loads(json.dumps(plan))
    serialized["jobs"][0]["count"] += 1
    serialized["budget"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        validate_generation_plan(serialized)

    unsigned = dict(plan)
    unsigned.pop("plan_fingerprint")
    assert plan["plan_fingerprint"] == payload_fingerprint(unsigned)


def test_executable_plan_pins_the_validation_source_bytes(tmp_path):
    source = tmp_path / "validation.jsonl"
    source.write_text(
        '{"split":"validation"}\n{"split":"validation"}\n',
        encoding="utf-8",
    )
    plan = plan_synthesis_batch(
        [
            _row("en", 0.5, 0.5, True),
            _row("ko", 0.0, 0.0, False),
        ],
        _config(),
        source_fingerprint=file_fingerprint(source),
        source_path=str(source),
    )

    assert validate_generation_plan_source(plan) == source
    source.write_text('{"split":"validation","changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint"):
        validate_generation_plan_source(plan)


def test_executable_progress_plan_pins_both_validation_sources(tmp_path):
    current = tmp_path / "current.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    current_rows = [_row("en", 0.5, 0.5, True)]
    baseline_rows = [_row("en", 0.0, 0.0, False)]
    current.write_text(
        json.dumps(current_rows[0]) + "\n",
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(baseline_rows[0]) + "\n",
        encoding="utf-8",
    )
    plan = plan_synthesis_batch(
        current_rows,
        _progress_config(),
        source_fingerprint=file_fingerprint(current),
        source_path=str(current),
        baseline_rows=baseline_rows,
        baseline_source_fingerprint=file_fingerprint(baseline),
        baseline_source_path=str(baseline),
    )

    assert validate_generation_plan_source(plan) == current
    baseline.write_text(
        json.dumps({**baseline_rows[0], "score": 0.1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline source fingerprint"):
        validate_generation_plan_source(plan)
