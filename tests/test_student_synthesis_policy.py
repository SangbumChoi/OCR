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
