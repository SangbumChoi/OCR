"""Failure-driven, leakage-safe synthetic document curriculum planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


POLICY_FACTORS = (
    "generator_case",
    "language",
    "difficulty_level",
    "layout_family",
    "composition_tier",
)
COMPOSITION_TIERS = {"single_document", "multi_page", "multi_document"}


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_fingerprint(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_evaluation_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"evaluation row {line_number} must be a JSON object"
                )
            rows.append(row)
    if not rows:
        raise ValueError("evaluation input contains no rows")
    return rows


def load_synthesis_policy_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("synthesis policy config must be a mapping")
    validate_synthesis_policy_config(payload)
    return payload


def _positive_number(mapping: Mapping[str, Any], key: str) -> float:
    value = float(mapping.get(key, 0.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return value


def validate_synthesis_policy_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("synthesis policy schema_version must be 1")
    if int(config.get("budget", 0)) <= 0:
        raise ValueError("synthesis policy budget must be positive")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("synthesis policy seed must be non-negative")
    for key in ("prior_strength", "uncertainty_coefficient", "temperature"):
        value = _positive_number(config, key)
        if key in {"prior_strength", "temperature"} and value <= 0:
            raise ValueError(f"{key} must be positive")
    exploration = float(config.get("exploration_fraction", -1.0))
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration_fraction must be within [0, 1]")
    reward_weights = config.get("failure_weights")
    if not isinstance(reward_weights, dict):
        raise ValueError("failure_weights must be a mapping")
    expected_rewards = {"score_deficit", "reward_deficit", "structure_failure"}
    if set(reward_weights) != expected_rewards:
        raise ValueError(
            "failure_weights must define score_deficit, reward_deficit, "
            "and structure_failure"
        )
    reward_total = sum(_positive_number(reward_weights, key) for key in reward_weights)
    if reward_total <= 0:
        raise ValueError("failure_weights must have positive total weight")
    factor_weights = config.get("factor_weights")
    if not isinstance(factor_weights, dict) or set(factor_weights) != set(
        POLICY_FACTORS
    ):
        raise ValueError(
            "factor_weights must define every synthesis policy factor"
        )
    factor_total = sum(_positive_number(factor_weights, key) for key in factor_weights)
    if factor_total <= 0:
        raise ValueError("factor_weights must have positive total weight")
    space = config.get("candidate_space")
    if not isinstance(space, dict):
        raise ValueError("candidate_space must be a mapping")
    languages = space.get("languages")
    difficulties = space.get("difficulty_levels")
    cases = space.get("cases")
    if not isinstance(languages, list) or not languages:
        raise ValueError("candidate_space.languages must be a non-empty list")
    if not isinstance(difficulties, list) or not difficulties:
        raise ValueError(
            "candidate_space.difficulty_levels must be a non-empty list"
        )
    if any(not str(language).strip() for language in languages):
        raise ValueError("candidate languages cannot be empty")
    if any(int(level) not in range(1, 6) for level in difficulties):
        raise ValueError("candidate difficulty levels must be within [1, 5]")
    if not isinstance(cases, list) or not cases:
        raise ValueError("candidate_space.cases must be a non-empty list")
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every candidate case must be a mapping")
        name = str(case.get("generator_case") or "").strip()
        if not name or name in seen_cases:
            raise ValueError("candidate generator_case values must be unique")
        seen_cases.add(name)
        if case.get("composition_tier") not in COMPOSITION_TIERS:
            raise ValueError(
                f"candidate case {name!r} has invalid composition_tier"
            )
        layouts = case.get("layout_families")
        if layouts is not None and (
            not isinstance(layouts, list)
            or not layouts
            or any(not str(layout).strip() for layout in layouts)
        ):
            raise ValueError(
                f"candidate case {name!r} layout_families must be null "
                "or a non-empty list"
            )
        case_languages = case.get("languages")
        if case_languages is not None and (
            not isinstance(case_languages, list)
            or not case_languages
            or any(not str(language).strip() for language in case_languages)
        ):
            raise ValueError(
                f"candidate case {name!r} languages must be null "
                "or a non-empty list"
            )
        case_difficulties = case.get("difficulty_levels")
        if case_difficulties is not None and (
            not isinstance(case_difficulties, list)
            or not case_difficulties
            or any(int(level) not in range(1, 6) for level in case_difficulties)
        ):
            raise ValueError(
                f"candidate case {name!r} difficulty_levels must be null "
                "or a non-empty list within [1, 5]"
            )
    generation = config.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping")
    if not isinstance(generation.get("no_degrade", False), bool):
        raise ValueError("generation.no_degrade must be a boolean")


def _difficulty_level(meta: Mapping[str, Any]) -> int:
    difficulty = meta.get("difficulty")
    if isinstance(difficulty, dict):
        difficulty = difficulty.get("level")
    try:
        level = int(difficulty)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation row is missing difficulty.level") from exc
    if level not in range(1, 6):
        raise ValueError("evaluation difficulty level must be within [1, 5]")
    return level


def _composition_tier(meta: Mapping[str, Any]) -> str:
    try:
        document_count = int(meta.get("document_count") or 1)
        page_count = int(meta.get("page_count") or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid page_count or document_count metadata") from exc
    if document_count > 1:
        return "multi_document"
    if page_count > 1:
        return "multi_page"
    return "single_document"


def arm_from_evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("evaluation row meta must be a mapping")
    generator_case = str(meta.get("generator_case") or "").strip()
    if not generator_case:
        raise ValueError(
            "evaluation metadata requires exact generator_case attribution"
        )
    language = str(row.get("language") or meta.get("language") or "").strip()
    if not language or language == "und":
        raise ValueError("evaluation metadata requires a known language")
    layout = meta.get("layout_family")
    return {
        "generator_case": generator_case,
        "language": language,
        "difficulty_level": _difficulty_level(meta),
        "layout_family": str(layout).strip() if layout else None,
        "composition_tier": _composition_tier(meta),
    }


def _clamped_metric(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"evaluation row requires numeric {key}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"evaluation {key} must be within [0, 1]")
    return value


def _failure(row: Mapping[str, Any], weights: Mapping[str, Any]) -> float:
    components = {
        "score_deficit": 1.0 - _clamped_metric(row, "score"),
        "reward_deficit": 1.0 - _clamped_metric(row, "reward"),
        "structure_failure": 0.0
        if row.get("structurally_valid") is True
        else 1.0,
    }
    total = sum(float(weights[name]) for name in components)
    return sum(
        float(weights[name]) * value for name, value in components.items()
    ) / total


def _candidate_arms(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    space = config["candidate_space"]
    arms: list[dict[str, Any]] = []
    for case in space["cases"]:
        layouts = case.get("layout_families") or [None]
        languages = case.get("languages") or space["languages"]
        difficulties = (
            case.get("difficulty_levels") or space["difficulty_levels"]
        )
        for language in languages:
            for difficulty in difficulties:
                for layout in layouts:
                    arm = {
                        "generator_case": str(case["generator_case"]),
                        "language": str(language),
                        "difficulty_level": int(difficulty),
                        "layout_family": (
                            str(layout) if layout is not None else None
                        ),
                        "composition_tier": str(case["composition_tier"]),
                    }
                    arm["arm_id"] = payload_fingerprint(arm)
                    arms.append(arm)
    return sorted(arms, key=lambda arm: arm["arm_id"])


def _largest_remainder(
    probabilities: Sequence[float],
    budget: int,
    tie_breakers: Sequence[str],
) -> list[int]:
    exact = [probability * budget for probability in probabilities]
    allocations = [math.floor(value) for value in exact]
    remaining = budget - sum(allocations)
    order = sorted(
        range(len(exact)),
        key=lambda index: (
            -(exact[index] - allocations[index]),
            tie_breakers[index],
        ),
    )
    for index in order[:remaining]:
        allocations[index] += 1
    return allocations


def _factor_key(value: Any) -> str:
    return "<none>" if value is None else str(value)


def _factor_statistics(
    observed: Sequence[tuple[dict[str, Any], float]],
    *,
    global_failure: float,
    prior_strength: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[str, float]] = {
        factor: defaultdict(float) for factor in POLICY_FACTORS
    }
    counts: dict[str, dict[str, int]] = {
        factor: defaultdict(int) for factor in POLICY_FACTORS
    }
    for arm, failure in observed:
        for factor in POLICY_FACTORS:
            value = _factor_key(arm[factor])
            totals[factor][value] += failure
            counts[factor][value] += 1
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for factor in POLICY_FACTORS:
        result[factor] = {}
        for value in sorted(counts[factor]):
            count = counts[factor][value]
            mean = totals[factor][value] / count
            posterior = (
                totals[factor][value] + prior_strength * global_failure
            ) / (count + prior_strength)
            result[factor][value] = {
                "n": count,
                "mean_failure": round(mean, 8),
                "posterior_failure": round(posterior, 8),
            }
    return result


def plan_synthesis_batch(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    source_fingerprint: str,
    source_path: str,
    budget: int | None = None,
    seed: int | None = None,
    allow_heldout_analysis: bool = False,
) -> dict[str, Any]:
    """Allocate an exact next batch from validation failures.

    Heldout rows can only produce an analysis artifact, never an executable
    training plan.
    """

    validate_synthesis_policy_config(config)
    if not rows:
        raise ValueError("cannot plan synthesis from empty evaluation rows")
    splits = {str(row.get("split") or "") for row in rows}
    if len(splits) != 1:
        raise ValueError("policy input must contain exactly one evaluation split")
    source_split = next(iter(splits))
    if source_split == "validation":
        training_authorized = True
    elif source_split == "heldout" and allow_heldout_analysis:
        training_authorized = False
    else:
        raise ValueError(
            "training policy learning requires split='validation'; heldout "
            "requires explicit analysis-only authorization"
        )
    effective_budget = int(config["budget"] if budget is None else budget)
    effective_seed = int(config["seed"] if seed is None else seed)
    if effective_budget <= 0 or effective_seed < 0:
        raise ValueError("budget must be positive and seed non-negative")
    failure_weights = config["failure_weights"]
    observed = [
        (arm_from_evaluation_row(row), _failure(row, failure_weights))
        for row in rows
    ]
    global_failure = sum(value for _, value in observed) / len(observed)
    prior_strength = float(config["prior_strength"])
    factor_stats = _factor_statistics(
        observed,
        global_failure=global_failure,
        prior_strength=prior_strength,
    )
    factor_weights = {
        factor: float(config["factor_weights"][factor])
        for factor in POLICY_FACTORS
    }
    factor_weight_total = sum(factor_weights.values())
    uncertainty_coefficient = float(config["uncertainty_coefficient"])
    candidates = _candidate_arms(config)
    logits: list[float] = []
    scored_candidates: list[dict[str, Any]] = []
    for arm in candidates:
        posterior_total = 0.0
        uncertainty_total = 0.0
        evidence_total = 0
        for factor in POLICY_FACTORS:
            stat = factor_stats[factor].get(_factor_key(arm[factor]))
            count = int(stat["n"]) if stat else 0
            posterior = (
                float(stat["posterior_failure"])
                if stat
                else global_failure
            )
            weight = factor_weights[factor]
            posterior_total += weight * posterior
            uncertainty_total += weight / math.sqrt(count + 1)
            evidence_total += count
        predicted_failure = posterior_total / factor_weight_total
        uncertainty = uncertainty_total / factor_weight_total
        priority = predicted_failure + uncertainty_coefficient * uncertainty
        logits.append(priority / float(config["temperature"]))
        scored_candidates.append(
            {
                **arm,
                "predicted_failure": round(predicted_failure, 8),
                "uncertainty": round(uncertainty, 8),
                "priority": round(priority, 8),
                "factor_evidence_count": evidence_total,
            }
        )
    max_logit = max(logits)
    exponentials = [math.exp(value - max_logit) for value in logits]
    softmax_total = sum(exponentials)
    exploitation = [value / softmax_total for value in exponentials]
    exploration = float(config["exploration_fraction"])
    uniform = 1.0 / len(candidates)
    probabilities = [
        (1.0 - exploration) * probability + exploration * uniform
        for probability in exploitation
    ]
    allocations = _largest_remainder(
        probabilities,
        effective_budget,
        [arm["arm_id"] for arm in candidates],
    )
    jobs: list[dict[str, Any]] = []
    for candidate, probability, count in zip(
        scored_candidates,
        probabilities,
        allocations,
    ):
        if count <= 0:
            continue
        job_index = len(jobs)
        arm_seed = (
            effective_seed
            + int(candidate["arm_id"].split(":", 1)[1][:8], 16)
        ) % (2**31 - 1)
        jobs.append(
            {
                **candidate,
                "allocation_probability": round(probability, 10),
                "count": count,
                "seed": arm_seed,
                "output_subdir": f"job-{job_index:04d}",
            }
        )
    jobs.sort(key=lambda job: (-float(job["priority"]), job["arm_id"]))
    for index, job in enumerate(jobs):
        job["output_subdir"] = f"job-{index:04d}"
    plan = {
        "schema_version": 1,
        "policy": "factor_shrinkage_failure_curriculum",
        "training_authorized": training_authorized,
        "claim_scope": (
            "next_training_batch"
            if training_authorized
            else "heldout_analysis_only"
        ),
        "source": {
            "split": source_split,
            "path": source_path,
            "fingerprint": source_fingerprint,
            "rows": len(rows),
        },
        "budget": effective_budget,
        "seed": effective_seed,
        "global_failure": round(global_failure, 8),
        "failure_weights": dict(failure_weights),
        "factor_weights": dict(factor_weights),
        "prior_strength": prior_strength,
        "uncertainty_coefficient": uncertainty_coefficient,
        "temperature": float(config["temperature"]),
        "exploration_fraction": exploration,
        "generation": dict(config.get("generation") or {}),
        "candidate_count": len(candidates),
        "factor_statistics": factor_stats,
        "jobs": jobs,
    }
    plan["plan_fingerprint"] = payload_fingerprint(plan)
    validate_generation_plan(
        plan,
        require_training_authorized=training_authorized,
    )
    return plan


def validate_generation_plan(
    plan: Mapping[str, Any],
    *,
    require_training_authorized: bool = False,
) -> None:
    if int(plan.get("schema_version", 0)) != 1:
        raise ValueError("generation plan schema_version must be 1")
    if require_training_authorized and plan.get("training_authorized") is not True:
        raise ValueError("generation plan is not authorized for training")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError("generation plan source must be a mapping")
    if (
        not str(source.get("path") or "").strip()
        or not str(source.get("fingerprint") or "").startswith("sha256:")
        or int(source.get("rows", 0)) <= 0
    ):
        raise ValueError(
            "generation plan source requires path, SHA-256 fingerprint, "
            "and positive row count"
        )
    if plan.get("training_authorized") is True and source.get("split") != "validation":
        raise ValueError(
            "training-authorized generation plans must originate from validation"
        )
    budget = int(plan.get("budget", 0))
    jobs = plan.get("jobs")
    if budget <= 0 or not isinstance(jobs, list) or not jobs:
        raise ValueError("generation plan requires a positive budget and jobs")
    seen_ids: set[str] = set()
    allocated = 0
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("generation plan jobs must be mappings")
        for factor in POLICY_FACTORS:
            if factor not in job:
                raise ValueError(f"generation job is missing {factor}")
        if job["composition_tier"] not in COMPOSITION_TIERS:
            raise ValueError("generation job has invalid composition_tier")
        if int(job["difficulty_level"]) not in range(1, 6):
            raise ValueError("generation job difficulty must be within [1, 5]")
        count = int(job.get("count", 0))
        if count <= 0:
            raise ValueError("generation job counts must be positive")
        arm_id = str(job.get("arm_id") or "")
        if not arm_id or arm_id in seen_ids:
            raise ValueError("generation job arm_id values must be unique")
        seen_ids.add(arm_id)
        allocated += count
    if allocated != budget:
        raise ValueError(
            f"generation plan allocations sum to {allocated}, expected {budget}"
        )
    fingerprint = plan.get("plan_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith(
        "sha256:"
    ):
        raise ValueError("generation plan requires a SHA-256 plan fingerprint")
    unsigned = dict(plan)
    unsigned.pop("plan_fingerprint", None)
    if fingerprint != payload_fingerprint(unsigned):
        raise ValueError("generation plan fingerprint does not match content")


def validate_generation_plan_source(plan: Mapping[str, Any]) -> Path:
    """Verify that an executable plan still points to its exact validation rows."""

    validate_generation_plan(plan, require_training_authorized=True)
    source_path = Path(plan["source"]["path"])
    if not source_path.is_file():
        raise ValueError(
            f"generation plan source artifact does not exist: {source_path}"
        )
    if file_fingerprint(source_path) != plan["source"]["fingerprint"]:
        raise ValueError(
            "generation plan source fingerprint does not match its "
            "validation artifact"
        )
    rows = load_evaluation_rows(source_path)
    if len(rows) != int(plan["source"]["rows"]):
        raise ValueError(
            "generation plan source row count does not match its "
            "validation artifact"
        )
    if {str(row.get("split") or "") for row in rows} != {"validation"}:
        raise ValueError(
            "training-authorized generation source must contain only "
            "validation rows"
        )
    return source_path


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def iter_plan_documents(plan: Mapping[str, Any]) -> Iterable[tuple[dict[str, Any], int]]:
    validate_generation_plan(plan, require_training_authorized=True)
    for job in plan["jobs"]:
        for replica in range(int(job["count"])):
            yield job, replica
