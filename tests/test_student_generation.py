import pytest


torch = pytest.importorskip("torch")


def test_task_label_generation_budgets_are_bounded_and_specific():
    from docvlm_eval.generation_policy import (
        resolve_generation_token_budget,
        validate_generation_token_budget_policy,
    )

    policy = validate_generation_token_budget_policy(
        base_tokens=128,
        hard_cap=512,
        by_answer_type={
            "table*": 384,
            "table-html": 512,
            "OCR-FULL": 512,
        },
    )

    assert resolve_generation_token_budget(
        "table-html",
        base_tokens=128,
        hard_cap=512,
        by_answer_type=policy,
    ) == (512, "table-html")
    assert resolve_generation_token_budget(
        "table-structure",
        base_tokens=128,
        hard_cap=512,
        by_answer_type=policy,
    ) == (384, "table*")
    assert resolve_generation_token_budget(
        "ocr-full",
        base_tokens=128,
        hard_cap=512,
        by_answer_type=policy,
    ) == (512, "ocr-full")
    assert resolve_generation_token_budget(
        "kie",
        base_tokens=128,
        hard_cap=512,
        by_answer_type=policy,
    ) == (128, "default")


@pytest.mark.parametrize(
    ("hard_cap", "overrides"),
    [
        (64, {}),
        (512, {"*": 256}),
        (512, {"table*middle": 256}),
        (512, {"table": 127}),
        (512, {"table": 513}),
        (512, {"table": 1.5}),
    ],
)
def test_task_label_generation_budget_rejects_unsafe_policies(
    hard_cap,
    overrides,
):
    from docvlm_eval.generation_policy import (
        validate_generation_token_budget_policy,
    )

    with pytest.raises(ValueError):
        validate_generation_token_budget_policy(
            base_tokens=128,
            hard_cap=hard_cap,
            by_answer_type=overrides,
        )


def test_suffix_cycle_detector_ignores_nonconsecutive_table_structure():
    from docvlm_eval.student.generation import (
        has_repeated_suffix_cycle,
        repeated_suffix_cycle_mask,
    )

    table_like = [10, 20, 30, 10, 21, 31, 10, 22, 32]
    loop = [4, 5, 4, 5, 4, 5]

    assert not has_repeated_suffix_cycle(
        table_like,
        min_tokens=6,
        max_period=3,
        repetitions=3,
    )
    assert has_repeated_suffix_cycle(
        loop,
        min_tokens=6,
        max_period=3,
        repetitions=3,
    )
    mask = repeated_suffix_cycle_mask(
        torch.tensor([[4, 5, 4, 5, 4], [10, 20, 10, 21, 10]]),
        torch.tensor([5, 22]),
        min_tokens=6,
        max_period=2,
        repetitions=3,
    )
    assert mask.tolist() == [True, False]
