"""Coverage audit for structured-generation token budgets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .generation_policy import (
    resolve_generation_token_budget,
    validate_generation_token_budget_policy,
)
from .schema import Sample
from .student.posttrain import StructuredPostTrainingDataset


@dataclass(frozen=True)
class GenerationBudgetPolicy:
    name: str
    base_tokens: int
    hard_cap: int
    by_answer_type: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("generation budget policy name cannot be empty")
        normalized = validate_generation_token_budget_policy(
            base_tokens=self.base_tokens,
            hard_cap=self.hard_cap,
            by_answer_type=dict(self.by_answer_type),
        )
        object.__setattr__(self, "by_answer_type", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_tokens": self.base_tokens,
            "hard_cap": self.hard_cap,
            "by_answer_type": dict(self.by_answer_type),
        }


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _length_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {
            "n": 0,
            "min": 0,
            "mean": 0.0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "n": len(values),
        "min": min(values),
        "mean": round(sum(values) / len(values), 6),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
    }


def _target_lengths(
    samples: Sequence[Sample],
    tokenizer: Any,
    *,
    target_mode: str,
) -> list[dict[str, Any]]:
    dataset = StructuredPostTrainingDataset(
        samples,
        target_mode=target_mode,
    )
    rows = []
    for index, sample in enumerate(dataset.samples):
        target = dataset.target(index)
        token_ids = tokenizer.encode(target, add_special_tokens=False)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "answer_type": sample.answer_type,
                "target_tokens": len(token_ids) + 1,
            }
        )
    return rows


def _coverage_summary(
    rows: Sequence[dict[str, Any]],
    policy: GenerationBudgetPolicy,
    *,
    max_examples: int,
) -> dict[str, Any]:
    evaluated = []
    for row in rows:
        budget, source = resolve_generation_token_budget(
            str(row["answer_type"]),
            base_tokens=policy.base_tokens,
            hard_cap=policy.hard_cap,
            by_answer_type=policy.by_answer_type,
        )
        target_tokens = int(row["target_tokens"])
        evaluated.append(
            {
                **row,
                "budget": budget,
                "budget_source": source,
                "covered": target_tokens <= budget,
                "within_hard_cap": target_tokens <= policy.hard_cap,
            }
        )

    def summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        covered = sum(bool(item["covered"]) for item in items)
        within_cap = sum(bool(item["within_hard_cap"]) for item in items)
        overflow = sorted(
            (
                {
                    "sample_id": str(item["sample_id"]),
                    "answer_type": str(item["answer_type"]),
                    "target_tokens": int(item["target_tokens"]),
                    "budget": int(item["budget"]),
                    "budget_source": str(item["budget_source"]),
                }
                for item in items
                if not item["covered"]
            ),
            key=lambda item: (
                -(item["target_tokens"] - item["budget"]),
                item["sample_id"],
            ),
        )
        return {
            **_length_summary(
                [int(item["target_tokens"]) for item in items]
            ),
            "covered": covered,
            "coverage": round(covered / count, 8) if count else 0.0,
            "within_hard_cap": within_cap,
            "hard_cap_coverage": (
                round(within_cap / count, 8) if count else 0.0
            ),
            "overflow_count": count - covered,
            "near_cap_count": sum(
                int(item["target_tokens"]) >= math.ceil(int(item["budget"]) * 0.9)
                for item in items
            ),
            "overflow_examples": overflow[:max_examples],
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evaluated:
        grouped.setdefault(str(item["answer_type"]), []).append(item)
    return {
        "overall": summarize(evaluated),
        "by_answer_type": {
            answer_type: summarize(items)
            for answer_type, items in sorted(grouped.items())
        },
    }


def audit_generation_budget_coverage(
    samples_by_split: Mapping[str, Sequence[Sample]],
    tokenizer: Any,
    policies: Sequence[GenerationBudgetPolicy],
    *,
    target_mode: str = "evidence_linked",
    calibration_splits: Sequence[str] = ("train", "validation"),
    minimum_coverage: float = 1.0,
    recommendation_multiple: int = 32,
    max_overflow_examples: int = 20,
    require_policy_consistency: bool = True,
) -> dict[str, Any]:
    """Audit target fit without deriving policy choices from heldout data."""

    if not samples_by_split or any(not rows for rows in samples_by_split.values()):
        raise ValueError("generation budget audit requires non-empty splits")
    if not policies:
        raise ValueError("generation budget audit requires at least one policy")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum coverage must be within (0, 1]")
    if recommendation_multiple <= 0 or max_overflow_examples <= 0:
        raise ValueError(
            "recommendation multiple and overflow example count must be positive"
        )
    missing_calibration = set(calibration_splits) - set(samples_by_split)
    if missing_calibration:
        raise ValueError(
            "generation budget calibration splits are missing: "
            f"{sorted(missing_calibration)}"
        )
    lengths_by_split = {
        split: _target_lengths(
            samples,
            tokenizer,
            target_mode=target_mode,
        )
        for split, samples in sorted(samples_by_split.items())
    }
    canonical_policies = [
        {
            "base_tokens": policy.base_tokens,
            "hard_cap": policy.hard_cap,
            "by_answer_type": dict(policy.by_answer_type),
        }
        for policy in policies
    ]
    policies_consistent = all(
        policy == canonical_policies[0]
        for policy in canonical_policies[1:]
    )
    policy_groups: dict[str, dict[str, Any]] = {}
    policy_index = {}
    for policy, canonical in zip(policies, canonical_policies, strict=True):
        group_id = _fingerprint(canonical)
        policy_index[policy.name] = group_id
        group = policy_groups.setdefault(
            group_id,
            {
                "stages": [],
                "policy": canonical,
                "splits": {
                    split: _coverage_summary(
                        rows,
                        policy,
                        max_examples=max_overflow_examples,
                    )
                    for split, rows in lengths_by_split.items()
                },
            },
        )
        group["stages"].append(policy.name)
    policy_groups = {
        group_id: policy_groups[group_id]
        for group_id in sorted(policy_groups)
    }
    gate_failures = []
    if require_policy_consistency and not policies_consistent:
        gate_failures.append("generation policies are not identical")
    for group in policy_groups.values():
        stage_label = ",".join(group["stages"])
        for split, coverage in group["splits"].items():
            observed = float(coverage["overall"]["coverage"])
            if observed < minimum_coverage:
                gate_failures.append(
                    f"{stage_label}:{split} coverage "
                    f"{observed:.8f} is below {minimum_coverage:.8f}"
                )

    calibration_rows = [
        row
        for split in calibration_splits
        for row in lengths_by_split[split]
    ]
    grouped_lengths: dict[str, list[int]] = {}
    for row in calibration_rows:
        grouped_lengths.setdefault(
            str(row["answer_type"]),
            [],
        ).append(int(row["target_tokens"]))
    recommendation_cap = min(policy.hard_cap for policy in policies)
    recommendations = {}
    for answer_type, lengths in sorted(grouped_lengths.items()):
        observed_max = max(lengths)
        recommended = int(
            math.ceil(observed_max / recommendation_multiple)
            * recommendation_multiple
        )
        recommendations[answer_type] = {
            **_length_summary(lengths),
            "recommended_tokens": min(recommended, recommendation_cap),
            "fits_shared_hard_cap": observed_max <= recommendation_cap,
            "derived_from_splits": list(calibration_splits),
        }
    report = {
        "schema_version": 1,
        "target_mode": target_mode,
        "tokenizer_fingerprint": str(
            getattr(tokenizer, "fingerprint", "unknown")
        ),
        "minimum_coverage": minimum_coverage,
        "require_policy_consistency": require_policy_consistency,
        "policies_consistent": policies_consistent,
        "calibration_splits": list(calibration_splits),
        "heldout_used_for_recommendations": False,
        "split_sample_counts": {
            split: len(rows) for split, rows in lengths_by_split.items()
        },
        "policy_index": policy_index,
        "policy_groups": policy_groups,
        "recommendations": recommendations,
        "gate": {
            "status": "pass" if not gate_failures else "fail",
            "failures": gate_failures,
        },
    }
    report["fingerprint"] = _fingerprint(report)
    return report
