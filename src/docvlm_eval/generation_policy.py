"""Target-independent generation budget policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_generation_token_budget_policy(
    *,
    base_tokens: int,
    hard_cap: int,
    by_answer_type: Any,
) -> tuple[tuple[str, int], ...]:
    """Normalize a bounded task-label policy without using target content."""

    if (
        not isinstance(base_tokens, int)
        or isinstance(base_tokens, bool)
        or base_tokens <= 0
    ):
        raise ValueError("base generation token budget must be a positive integer")
    if (
        not isinstance(hard_cap, int)
        or isinstance(hard_cap, bool)
        or hard_cap < base_tokens
    ):
        raise ValueError(
            "max_new_tokens_hard_cap must be an integer at least as large "
            "as the base generation token budget"
        )
    if not isinstance(by_answer_type, Mapping):
        raise ValueError("max_new_tokens_by_answer_type must be a mapping")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw_pattern, raw_budget in by_answer_type.items():
        pattern = str(raw_pattern).strip().lower()
        if (
            not pattern
            or pattern == "*"
            or "*" in pattern[:-1]
            or pattern.count("*") > 1
        ):
            raise ValueError(
                "max_new_tokens_by_answer_type patterns must be non-empty "
                "exact labels or trailing-wildcard prefixes"
            )
        if pattern in seen:
            raise ValueError(
                "max_new_tokens_by_answer_type patterns must be unique "
                "case-insensitively"
            )
        if not isinstance(raw_budget, int) or isinstance(raw_budget, bool):
            raise ValueError(
                "answer-type generation budgets must be integer values "
                "between the base budget and hard cap"
            )
        if not base_tokens <= raw_budget <= hard_cap:
            raise ValueError(
                "answer-type generation budgets must be integer values "
                "between the base budget and hard cap"
            )
        seen.add(pattern)
        normalized.append((pattern, raw_budget))
    return tuple(normalized)


def resolve_generation_token_budget(
    answer_type: str,
    *,
    base_tokens: int,
    hard_cap: int,
    by_answer_type: Mapping[str, int] | tuple[tuple[str, int], ...],
) -> tuple[int, str]:
    """Resolve one public task label to a bounded generation budget."""

    items = (
        tuple(by_answer_type.items())
        if isinstance(by_answer_type, Mapping)
        else tuple(by_answer_type)
    )
    policy = validate_generation_token_budget_policy(
        base_tokens=base_tokens,
        hard_cap=hard_cap,
        by_answer_type=dict(items),
    )
    label = str(answer_type).strip().lower()
    exact = {
        pattern: budget
        for pattern, budget in policy
        if not pattern.endswith("*")
    }
    if label in exact:
        return exact[label], label
    prefixes = sorted(
        (
            (pattern[:-1], budget, pattern)
            for pattern, budget in policy
            if pattern.endswith("*") and label.startswith(pattern[:-1])
        ),
        key=lambda item: (-len(item[0]), item[2]),
    )
    if prefixes:
        _, budget, pattern = prefixes[0]
        return budget, pattern
    return base_tokens, "default"
