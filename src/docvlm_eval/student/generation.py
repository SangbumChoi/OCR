"""Structure-preserving generation guards and diagnostics."""

from __future__ import annotations

import torch


def repeated_suffix_cycle_mask(
    completion_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    min_tokens: int,
    max_period: int,
    repetitions: int,
) -> torch.Tensor:
    """Detect exact consecutive suffix cycles after appending one candidate.

    This does not ban recurring table tags or common n-grams at distant
    positions. It only fires when the entire trailing period repeats
    consecutively, which is the characteristic failure mode of generation
    loops near a token limit.
    """

    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [batch, tokens]")
    if candidate_ids.shape != (completion_ids.shape[0],):
        raise ValueError("candidate_ids must have shape [batch]")
    if min_tokens < 1 or max_period < 1 or repetitions < 2:
        raise ValueError(
            "repetition guard requires positive token/period limits and "
            "at least two repetitions"
        )
    candidate_sequence = torch.cat(
        (completion_ids, candidate_ids[:, None]),
        dim=1,
    )
    length = int(candidate_sequence.shape[1])
    detected = torch.zeros(
        candidate_sequence.shape[0],
        dtype=torch.bool,
        device=candidate_sequence.device,
    )
    if length < min_tokens:
        return detected
    largest_period = min(max_period, length // repetitions)
    for period in range(1, largest_period + 1):
        suffix = candidate_sequence[:, -period:]
        repeated = torch.ones_like(detected)
        for offset in range(2, repetitions + 1):
            start = length - offset * period
            stop = start + period
            repeated &= torch.all(
                candidate_sequence[:, start:stop] == suffix,
                dim=1,
            )
        detected |= repeated
    return detected


def has_repeated_suffix_cycle(
    token_ids: list[int],
    *,
    min_tokens: int,
    max_period: int,
    repetitions: int,
) -> bool:
    if not token_ids:
        return False
    values = torch.tensor(token_ids, dtype=torch.long)[None, :-1]
    candidate = torch.tensor([token_ids[-1]], dtype=torch.long)
    return bool(
        repeated_suffix_cycle_mask(
            values,
            candidate,
            min_tokens=min_tokens,
            max_period=max_period,
            repetitions=repetitions,
        )[0]
    )
