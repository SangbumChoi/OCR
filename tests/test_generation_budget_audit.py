from docvlm_eval.generation_budget_audit import (
    GenerationBudgetPolicy,
    audit_generation_budget_coverage,
)
from docvlm_eval.schema import Sample


class _CharacterTokenizer:
    fingerprint = "sha256:character-tokenizer"

    def encode(self, text, add_special_tokens=False):
        tokens = list(range(len(str(text))))
        return [1, *tokens, 2] if add_special_tokens else tokens


def _sample(sample_id, answer_type, answer):
    return Sample(
        sample_id=sample_id,
        image_path="",
        question="Read the document.",
        answers=[answer],
        answer_type=answer_type,
        metric="exact",
    )


def test_generation_budget_audit_uses_only_calibration_splits_for_recommendations():
    policy = GenerationBudgetPolicy(
        name="evaluation",
        base_tokens=128,
        hard_cap=512,
        by_answer_type=(("table*", 512),),
    )
    report = audit_generation_budget_coverage(
        {
            "train": [
                _sample("train-kie", "kie", "short"),
                _sample("train-table", "table-html", "x" * 180),
            ],
            "validation": [
                _sample("validation-table", "table-html", "x" * 220),
            ],
            "heldout": [
                _sample("heldout-table", "table-html", "x" * 600),
            ],
        },
        _CharacterTokenizer(),
        (policy,),
        minimum_coverage=1.0,
    )

    policy_group = report["policy_groups"][
        report["policy_index"]["evaluation"]
    ]
    train = policy_group["splits"]["train"]
    heldout = policy_group["splits"]["heldout"]
    recommendation = report["recommendations"]["table-html"]
    assert train["overall"]["coverage"] == 1.0
    assert heldout["overall"]["coverage"] == 0.0
    assert heldout["overall"]["overflow_examples"][0]["sample_id"] == (
        "heldout-table"
    )
    assert recommendation["derived_from_splits"] == [
        "train",
        "validation",
    ]
    assert recommendation["max"] < heldout["overall"]["max"]
    assert heldout["overall"]["near_cap_count"] == 1
    assert report["heldout_used_for_recommendations"] is False
    assert report["gate"]["status"] == "fail"


def test_generation_budget_audit_fails_mismatched_stage_policies():
    samples = {
        "train": [_sample("train", "kie", "short")],
        "heldout": [_sample("heldout", "kie", "short")],
    }
    report = audit_generation_budget_coverage(
        samples,
        _CharacterTokenizer(),
        (
            GenerationBudgetPolicy(
                name="evaluation",
                base_tokens=128,
                hard_cap=512,
            ),
            GenerationBudgetPolicy(
                name="rlvr",
                base_tokens=128,
                hard_cap=512,
                by_answer_type=(("kie", 256),),
            ),
        ),
        calibration_splits=("train",),
    )

    assert report["policies_consistent"] is False
    assert report["gate"] == {
        "status": "fail",
        "failures": ["generation policies are not identical"],
    }


def test_generation_budget_audit_passes_complete_consistent_coverage():
    samples = {
        "train": [_sample("train", "kie", "short")],
        "heldout": [_sample("heldout", "kie", "also short")],
    }
    policies = tuple(
        GenerationBudgetPolicy(
            name=name,
            base_tokens=128,
            hard_cap=512,
        )
        for name in ("evaluation", "preference", "rlvr")
    )
    report = audit_generation_budget_coverage(
        samples,
        _CharacterTokenizer(),
        policies,
        calibration_splits=("train",),
    )

    assert report["policies_consistent"] is True
    assert len(report["policy_groups"]) == 1
    only_group = next(iter(report["policy_groups"].values()))
    assert only_group["stages"] == ["evaluation", "preference", "rlvr"]
    assert report["gate"] == {"status": "pass", "failures": []}
    assert report["fingerprint"].startswith("sha256:")
