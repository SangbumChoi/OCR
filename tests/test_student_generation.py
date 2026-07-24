import pytest


torch = pytest.importorskip("torch")


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
