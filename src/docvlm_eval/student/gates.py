"""Executable deployment gates for native-student evaluation artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import StudentConfig, student_config_fingerprint
from .optim import OptimizerSpec


_STATUSES = {"pass", "fail", "insufficient_evidence"}
_ABSTENTIONS = {"", "none", "n/a", "na", "not available", "unknown", "absent"}


def _round(value: float) -> float:
    return round(float(value), 6)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _result(
    gate: Mapping[str, Any],
    status: str,
    reason: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in _STATUSES:
        raise ValueError(f"invalid gate status {status!r}")
    return {
        "id": str(gate["id"]),
        "requirement": str(gate["requirement"]),
        "status": status,
        "reason": reason,
        "evidence": dict(evidence or {}),
    }


def _heldout_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> list[Mapping[str, Any]]:
    return list((rows or {}).get("heldout", ()))


def _matched_rows(
    current_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], str | None]:
    current = {str(row["sample_id"]): row for row in current_rows}
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    if not current or not baseline:
        return [], "held-out per-sample artifacts are missing"
    if set(current) != set(baseline):
        return [], (
            "current and baseline held-out sample IDs differ "
            f"({len(current)} current, {len(baseline)} baseline)"
        )
    return [(current[key], baseline[key]) for key in sorted(current)], None


def _component(row: Mapping[str, Any], name: str) -> float | None:
    if name not in row.get("applicable_rewards", ()):
        return None
    value = row.get("reward_components", {}).get(name)
    return float(value) if value is not None else None


def _parameter_budget(
    gate: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    maximum = int(gate["max_parameters"])
    total = int(parameter_counts.get("total", -1))
    if total < 0:
        return _result(
            gate,
            "insufficient_evidence",
            "actual model parameter count is unavailable",
        )
    status = "pass" if total < maximum else "fail"
    return _result(
        gate,
        status,
        f"actual deployment parameter count is {'below' if status == 'pass' else 'not below'} the limit",
        {
            "actual_parameters": total,
            "max_parameters_exclusive": maximum,
            "components": {
                key: int(value)
                for key, value in parameter_counts.items()
                if key != "total"
            },
        },
    )


def _generalization(
    gate: Mapping[str, Any],
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_error: str | None,
) -> dict[str, Any]:
    if baseline is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a reference-checkpoint evaluation is required",
        )
    if pair_error:
        return _result(gate, "insufficient_evidence", pair_error)
    try:
        current_heldout = float(current["splits"]["heldout"]["score"])
        baseline_heldout = float(baseline["splits"]["heldout"]["score"])
        current_gap = float(
            current["train_minus_heldout"]["headline"]["score"]
        )
        baseline_gap = float(
            baseline["train_minus_heldout"]["headline"]["score"]
        )
    except (KeyError, TypeError, ValueError):
        return _result(
            gate,
            "insufficient_evidence",
            "train, held-out, or generalization-gap summaries are missing",
        )
    heldout_delta = current_heldout - baseline_heldout
    gap_increase = current_gap - baseline_gap
    minimum_delta = float(gate["min_heldout_score_delta"])
    maximum_gap_increase = float(gate["max_gap_increase"])
    passed = heldout_delta >= minimum_delta and gap_increase <= maximum_gap_increase
    return _result(
        gate,
        "pass" if passed else "fail",
        "held-out improvement and gap-growth constraints were evaluated on matched samples",
        {
            "matched_samples": len(pairs),
            "current_heldout_score": _round(current_heldout),
            "baseline_heldout_score": _round(baseline_heldout),
            "heldout_score_delta": _round(heldout_delta),
            "min_heldout_score_delta": minimum_delta,
            "current_train_minus_heldout_gap": _round(current_gap),
            "baseline_train_minus_heldout_gap": _round(baseline_gap),
            "gap_increase": _round(gap_increase),
            "max_gap_increase": maximum_gap_increase,
        },
    )


def _grounding(
    gate: Mapping[str, Any],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_error: str | None,
    baseline_available: bool,
) -> dict[str, Any]:
    if not baseline_available:
        return _result(
            gate,
            "insufficient_evidence",
            "a reference-checkpoint evaluation is required",
        )
    if pair_error:
        return _result(gate, "insufficient_evidence", pair_error)
    box_pairs = [
        (current_box, baseline_box)
        for current, baseline in pairs
        if (current_box := _component(current, "box_iou")) is not None
        and (baseline_box := _component(baseline, "box_iou")) is not None
    ]
    extraction_patterns = tuple(
        str(value).lower()
        for value in gate.get(
            "extraction_answer_type_patterns",
            ("ocr", "extract", "kie", "transcription"),
        )
    )
    text_pairs = [
        (current_text, baseline_text)
        for current, baseline in pairs
        if any(
            pattern in str(current.get("answer_type", "")).lower()
            for pattern in extraction_patterns
        )
        and (current_text := _component(
            current, "normalized_text_similarity"
        )) is not None
        and (baseline_text := _component(
            baseline, "normalized_text_similarity"
        )) is not None
    ]
    if not box_pairs or not text_pairs:
        return _result(
            gate,
            "insufficient_evidence",
            "matched box-IoU and extraction-text slices are both required",
            {
                "box_samples": len(box_pairs),
                "extraction_samples": len(text_pairs),
            },
        )
    box_delta = _mean([pair[0] for pair in box_pairs]) - _mean(
        [pair[1] for pair in box_pairs]
    )
    text_drop = _mean([pair[1] for pair in text_pairs]) - _mean(
        [pair[0] for pair in text_pairs]
    )
    minimum_box_delta = float(gate["min_box_iou_delta"])
    maximum_text_drop = float(gate["max_extraction_similarity_drop"])
    passed = box_delta >= minimum_box_delta and text_drop <= maximum_text_drop
    return _result(
        gate,
        "pass" if passed else "fail",
        "grounding gain and extraction preservation were evaluated on matched slices",
        {
            "box_samples": len(box_pairs),
            "box_iou_delta": _round(box_delta),
            "min_box_iou_delta": minimum_box_delta,
            "extraction_samples": len(text_pairs),
            "extraction_similarity_drop": _round(text_drop),
            "max_extraction_similarity_drop": maximum_text_drop,
        },
    )


def _reasoning(
    gate: Mapping[str, Any],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_error: str | None,
    baseline_available: bool,
) -> dict[str, Any]:
    if not baseline_available:
        return _result(
            gate,
            "insufficient_evidence",
            "a reference-checkpoint evaluation is required",
        )
    if pair_error:
        return _result(gate, "insufficient_evidence", pair_error)
    groups: dict[str, dict[bool, list[float]]] = {}
    for current, baseline in pairs:
        meta = current.get("meta", {})
        hypothesis = meta.get("counterfactual_group") or meta.get("hypothesis")
        if not hypothesis or "control" not in meta:
            continue
        control = bool(meta["control"])
        groups.setdefault(str(hypothesis), {False: [], True: []})[control].append(
            float(current["score"]) - float(baseline["score"])
        )
    complete = {
        name: values
        for name, values in groups.items()
        if values[False] and values[True]
    }
    minimum_pairs = int(gate["min_counterfactual_pairs"])
    if len(complete) < minimum_pairs:
        return _result(
            gate,
            "insufficient_evidence",
            "too few matched reasoning groups contain both factual and counterfactual variants",
            {
                "complete_counterfactual_groups": len(complete),
                "min_counterfactual_pairs": minimum_pairs,
            },
        )
    factual_delta = _mean(
        [score for values in complete.values() for score in values[False]]
    )
    counterfactual_delta = _mean(
        [score for values in complete.values() for score in values[True]]
    )
    minimum_delta = float(gate["min_score_delta"])
    passed = factual_delta >= minimum_delta and counterfactual_delta >= minimum_delta
    return _result(
        gate,
        "pass" if passed else "fail",
        "reasoning deltas were measured separately on factual and counterfactual variants",
        {
            "complete_counterfactual_groups": len(complete),
            "factual_score_delta": _round(factual_delta),
            "counterfactual_score_delta": _round(counterfactual_delta),
            "min_score_delta": minimum_delta,
        },
    )


def _multilingual(
    gate: Mapping[str, Any],
    current: Mapping[str, Any],
    control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if control is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a monolingual-control evaluation is required",
        )
    try:
        current_languages = current["splits"]["heldout"]["by_language"]
        control_languages = control["splits"]["heldout"]["by_language"]
    except (KeyError, TypeError):
        return _result(
            gate,
            "insufficient_evidence",
            "held-out language summaries are missing",
        )
    languages = sorted(set(current_languages) & set(control_languages))
    missing = sorted(set(current_languages) - set(control_languages))
    minimum_languages = int(gate["min_languages"])
    if missing or len(languages) < minimum_languages:
        return _result(
            gate,
            "insufficient_evidence",
            "monolingual controls do not cover every required language",
            {
                "matched_languages": languages,
                "missing_control_languages": missing,
                "min_languages": minimum_languages,
            },
        )
    drops = {
        language: float(control_languages[language]["score"])
        - float(current_languages[language]["score"])
        for language in languages
    }
    maximum_drop = float(gate["max_language_drop"])
    passed = all(drop <= maximum_drop for drop in drops.values())
    return _result(
        gate,
        "pass" if passed else "fail",
        "each multilingual score was compared with its monolingual control",
        {
            "language_drops": {
                language: _round(drop) for language, drop in drops.items()
            },
            "max_language_drop": maximum_drop,
        },
    )


def _selective_risk(
    rows: Sequence[Mapping[str, Any]],
    coverage: float,
) -> float | None:
    if not rows:
        return None
    confidence_key = (
        "calibrated_confidence"
        if all(row.get("calibrated_confidence") is not None for row in rows)
        else "confidence"
    )
    if any(row.get(confidence_key) is None for row in rows):
        return None
    selected = sorted(
        rows,
        key=lambda row: float(row[confidence_key]),
        reverse=True,
    )[: max(1, math.ceil(len(rows) * coverage))]
    return 1.0 - _mean([float(row["score"]) for row in selected])


def _hallucination_rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    abstention_rows = [
        row
        for row in rows
        if bool(row.get("meta", {}).get("abstain_expected"))
        or "absence" in str(row.get("answer_type", "")).lower()
    ]
    if not abstention_rows:
        return None
    hallucinations = []
    for row in abstention_rows:
        answer = str(row.get("answer", "")).strip().lower()
        hallucinations.append(
            float(answer not in _ABSTENTIONS and float(row["score"]) < 1.0)
        )
    return _mean(hallucinations)


def _reliability(
    gate: Mapping[str, Any],
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_error: str | None,
) -> dict[str, Any]:
    if baseline is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a reference-checkpoint evaluation is required",
        )
    if pair_error:
        return _result(gate, "insufficient_evidence", pair_error)
    current_rows = [pair[0] for pair in pairs]
    baseline_rows = [pair[1] for pair in pairs]
    coverage = float(gate["coverage"])
    current_risk = _selective_risk(current_rows, coverage)
    baseline_risk = _selective_risk(baseline_rows, coverage)
    current_hallucination = _hallucination_rate(current_rows)
    baseline_hallucination = _hallucination_rate(baseline_rows)
    try:
        current_calibration = current["splits"]["heldout"]["calibration"]
        baseline_calibration = baseline["splits"]["heldout"]["calibration"]
        current_raw_ece = float(current_calibration["raw_ece"])
        current_calibrated_ece = float(
            current_calibration["calibrated_ece"]
        )
        baseline_calibrated_ece = float(
            baseline_calibration["calibrated_ece"]
        )
    except (KeyError, TypeError, ValueError):
        return _result(
            gate,
            "insufficient_evidence",
            "held-out raw and calibrated ECE are required",
        )
    if (
        current_risk is None
        or baseline_risk is None
        or current_hallucination is None
        or baseline_hallucination is None
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "per-sample confidence and an explicit abstention slice are required",
            {
                "matched_samples": len(pairs),
                "confidence_available": current_risk is not None
                and baseline_risk is not None,
                "abstention_slice_available": current_hallucination is not None
                and baseline_hallucination is not None,
            },
        )
    risk_reduction = baseline_risk - current_risk
    hallucination_increase = current_hallucination - baseline_hallucination
    minimum_reduction = float(gate["min_selective_risk_reduction"])
    maximum_hallucination = float(gate["max_hallucination_increase"])
    maximum_ece = float(gate["max_calibrated_ece"])
    maximum_ece_increase = float(gate["max_ece_increase_vs_raw"])
    minimum_ece_reduction = float(
        gate["min_calibrated_ece_reduction"]
    )
    ece_increase = current_calibrated_ece - current_raw_ece
    ece_reduction = baseline_calibrated_ece - current_calibrated_ece
    passed = (
        risk_reduction >= minimum_reduction
        and hallucination_increase <= maximum_hallucination
        and current_calibrated_ece <= maximum_ece
        and ece_increase <= maximum_ece_increase
        and ece_reduction >= minimum_ece_reduction
    )
    return _result(
        gate,
        "pass" if passed else "fail",
        "selective risk and hallucination were compared at fixed coverage",
        {
            "coverage": coverage,
            "current_selective_risk": _round(current_risk),
            "baseline_selective_risk": _round(baseline_risk),
            "selective_risk_reduction": _round(risk_reduction),
            "min_selective_risk_reduction": minimum_reduction,
            "current_hallucination_rate": _round(current_hallucination),
            "baseline_hallucination_rate": _round(baseline_hallucination),
            "hallucination_increase": _round(hallucination_increase),
            "max_hallucination_increase": maximum_hallucination,
            "current_raw_ece": _round(current_raw_ece),
            "current_calibrated_ece": _round(current_calibrated_ece),
            "baseline_calibrated_ece": _round(baseline_calibrated_ece),
            "calibrated_ece_reduction": _round(ece_reduction),
            "min_calibrated_ece_reduction": minimum_ece_reduction,
            "ece_increase_vs_raw": _round(ece_increase),
            "max_ece_increase_vs_raw": maximum_ece_increase,
            "max_calibrated_ece": maximum_ece,
        },
    )


def _generation_stability(
    gate: Mapping[str, Any],
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_error: str | None,
) -> dict[str, Any]:
    if baseline is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a reference-checkpoint evaluation is required",
        )
    if pair_error:
        return _result(gate, "insufficient_evidence", pair_error)
    try:
        current_policy = current["splits"]["heldout"][
            "generation_token_budget_policy"
        ]
        baseline_policy = baseline["splits"]["heldout"][
            "generation_token_budget_policy"
        ]
    except (KeyError, TypeError):
        return _result(
            gate,
            "insufficient_evidence",
            "heldout generation token policies are missing",
        )
    if (
        not isinstance(current_policy, Mapping)
        or not isinstance(baseline_policy, Mapping)
        or dict(current_policy) != dict(baseline_policy)
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "current and baseline generation token policies differ",
        )
    patterns = tuple(
        str(value).strip().lower()
        for value in gate["answer_type_patterns"]
    )
    target_pairs = [
        (current, baseline)
        for current, baseline in pairs
        if any(
            pattern in str(current.get("answer_type", "")).lower()
            for pattern in patterns
        )
    ]
    minimum_samples = int(gate["min_target_samples"])
    if len(target_pairs) < minimum_samples:
        return _result(
            gate,
            "insufficient_evidence",
            "too few matched long structured generation samples",
            {
                "matched_target_samples": len(target_pairs),
                "min_target_samples": minimum_samples,
                "answer_type_patterns": list(patterns),
            },
        )

    required_boolean_fields = (
        "degenerate_repetition",
        "reached_max_new_tokens",
        "structurally_valid",
    )
    for current, baseline in target_pairs:
        sample_id = str(current.get("sample_id") or "")
        if current.get("answer_type") != baseline.get("answer_type"):
            return _result(
                gate,
                "insufficient_evidence",
                "matched generation answer types differ",
                {"sample_id": sample_id},
            )
        for label, row in (("current", current), ("baseline", baseline)):
            if any(
                not isinstance(row.get(field), bool)
                for field in required_boolean_fields
            ):
                return _result(
                    gate,
                    "insufficient_evidence",
                    "generation diagnostics require boolean repetition, "
                    "limit, and structure fields",
                    {"sample_id": sample_id, "artifact": label},
                )
            try:
                generated_tokens = int(row["generated_tokens"])
                token_budget = int(row["generation_token_budget"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                return _result(
                    gate,
                    "insufficient_evidence",
                    "generation diagnostics require token counts, budget, "
                    "and score",
                    {"sample_id": sample_id, "artifact": label},
                )
            if (
                generated_tokens < 0
                or token_budget <= 0
                or generated_tokens > token_budget
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                return _result(
                    gate,
                    "insufficient_evidence",
                    "generation diagnostics contain invalid values",
                    {"sample_id": sample_id, "artifact": label},
                )
        if (
            int(current["generation_token_budget"])
            != int(baseline["generation_token_budget"])
            or current.get("generation_token_budget_source")
            != baseline.get("generation_token_budget_source")
            or not str(
                current.get("generation_token_budget_source") or ""
            ).strip()
            or not str(
                baseline.get("generation_token_budget_source") or ""
            ).strip()
        ):
            return _result(
                gate,
                "insufficient_evidence",
                "current and baseline generation token policies differ",
                {"sample_id": sample_id},
            )

    current_rows = [pair[0] for pair in target_pairs]
    baseline_rows = [pair[1] for pair in target_pairs]

    def rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        return _mean([float(row[field]) for row in rows])

    def mean_utilization(rows: Sequence[Mapping[str, Any]]) -> float:
        return _mean(
            [
                int(row["generated_tokens"])
                / int(row["generation_token_budget"])
                for row in rows
            ]
        )

    current_repetition = rate(current_rows, "degenerate_repetition")
    baseline_repetition = rate(baseline_rows, "degenerate_repetition")
    repetition_increase = current_repetition - baseline_repetition
    current_max_token = rate(current_rows, "reached_max_new_tokens")
    baseline_max_token = rate(baseline_rows, "reached_max_new_tokens")
    max_token_increase = current_max_token - baseline_max_token
    current_score = _mean([float(row["score"]) for row in current_rows])
    baseline_score = _mean([float(row["score"]) for row in baseline_rows])
    score_drop = baseline_score - current_score
    current_structure = rate(current_rows, "structurally_valid")
    baseline_structure = rate(baseline_rows, "structurally_valid")
    structure_drop = baseline_structure - current_structure
    violations = []
    if current_repetition > float(
        gate["max_degenerate_repetition_rate"]
    ):
        violations.append("degenerate_repetition_rate")
    if repetition_increase > float(
        gate["max_degenerate_repetition_increase"]
    ):
        violations.append("degenerate_repetition_increase")
    if current_max_token > float(gate["max_token_rate"]):
        violations.append("max_token_rate")
    if max_token_increase > float(gate["max_token_rate_increase"]):
        violations.append("max_token_rate_increase")
    if score_drop > float(gate["max_target_score_drop"]):
        violations.append("target_score_drop")
    if structure_drop > float(gate["max_structure_validity_drop"]):
        violations.append("structure_validity_drop")
    evidence = {
        "matched_target_samples": len(target_pairs),
        "answer_type_patterns": list(patterns),
        "current_degenerate_repetition_rate": _round(
            current_repetition
        ),
        "baseline_degenerate_repetition_rate": _round(
            baseline_repetition
        ),
        "degenerate_repetition_increase": _round(
            repetition_increase
        ),
        "max_degenerate_repetition_rate": float(
            gate["max_degenerate_repetition_rate"]
        ),
        "max_degenerate_repetition_increase": float(
            gate["max_degenerate_repetition_increase"]
        ),
        "current_max_token_rate": _round(current_max_token),
        "baseline_max_token_rate": _round(baseline_max_token),
        "max_token_rate_increase": _round(max_token_increase),
        "max_token_rate": float(gate["max_token_rate"]),
        "max_allowed_token_rate_increase": float(
            gate["max_token_rate_increase"]
        ),
        "current_target_score": _round(current_score),
        "baseline_target_score": _round(baseline_score),
        "target_score_drop": _round(score_drop),
        "max_target_score_drop": float(gate["max_target_score_drop"]),
        "current_structure_validity": _round(current_structure),
        "baseline_structure_validity": _round(baseline_structure),
        "structure_validity_drop": _round(structure_drop),
        "max_structure_validity_drop": float(
            gate["max_structure_validity_drop"]
        ),
        "current_mean_budget_utilization": _round(
            mean_utilization(current_rows)
        ),
        "baseline_mean_budget_utilization": _round(
            mean_utilization(baseline_rows)
        ),
        "violations": violations,
    }
    return _result(
        gate,
        "fail" if violations else "pass",
        (
            "long structured generation violates repetition, truncation, "
            "score, or structure thresholds"
            if violations
            else "long structured generation remains stable under a matched "
            "token policy"
        ),
        evidence,
    )


def _visual_efficiency(
    gate: Mapping[str, Any],
    blueprint: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sequence_mode = (
        blueprint.get("training", {})
        .get("pretraining", {})
        .get("input_pipeline", {})
        .get("visual_sequence_mode")
    )
    if sequence_mode != "packed":
        return _result(
            gate,
            "insufficient_evidence",
            "the packed visual backend gate does not evaluate dense execution",
            {"visual_sequence_mode": sequence_mode},
        )
    if report is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a target-device visual backend benchmark is required",
        )
    minimum_schema_version = int(
        gate.get("min_benchmark_schema_version", 2)
    )
    try:
        reported_schema_version = int(report.get("schema_version", 0))
    except (TypeError, ValueError):
        reported_schema_version = 0
    if (
        reported_schema_version < minimum_schema_version
        or report.get("scope")
        != "student_vision_tower_and_connector"
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "visual backend benchmark schema or scope is invalid",
        )
    expected_student = StudentConfig.from_blueprint(dict(blueprint))
    expected_config = expected_student.to_dict()
    expected_fingerprint = student_config_fingerprint(expected_student)
    if (
        report.get("student_config") != expected_config
        or report.get("student_config_fingerprint")
        != expected_fingerprint
        or report.get("connector_family")
        != expected_student.connector.family
    ):
        return _result(
            gate,
            "fail",
            "visual backend evidence was measured on a different student configuration",
            {
                "reported_fingerprint": report.get(
                    "student_config_fingerprint"
                ),
                "expected_fingerprint": expected_fingerprint,
                "reported_connector_family": report.get(
                    "connector_family"
                ),
                "expected_connector_family": (
                    expected_student.connector.family
                ),
            },
        )

    environment = report.get("environment")
    benchmark_config = report.get("benchmark_config")
    records = report.get("results")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(benchmark_config, Mapping)
        or not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "visual backend benchmark metadata is incomplete",
        )
    required_device = str(gate.get("required_device_type", "cuda"))
    if environment.get("device_type") != required_device:
        return _result(
            gate,
            "insufficient_evidence",
            f"benchmark must run on {required_device}",
            {
                "reported_device_type": environment.get("device_type"),
                "reported_device": environment.get("device"),
            },
        )

    requested = str(gate.get("candidate_requested_backend", "auto"))
    candidate = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("requested_backend") == requested
        ),
        None,
    )
    loop = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("requested_backend") == "loop"
        ),
        None,
    )
    dense_requested = str(
        gate.get("dense_control_requested_backend", "dense_adaptive")
    )
    dense = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("requested_backend") == dense_requested
        ),
        None,
    )
    if candidate is None or loop is None or dense is None:
        return _result(
            gate,
            "insufficient_evidence",
            "matched loop, dense-control, and candidate records are required",
            {
                "candidate_requested_backend": requested,
                "candidate_available": candidate is not None,
                "loop_available": loop is not None,
                "dense_control_requested_backend": dense_requested,
                "dense_control_available": dense is not None,
            },
        )
    if (
        candidate.get("status") != "ok"
        or loop.get("status") != "ok"
        or dense.get("status") != "ok"
    ):
        return _result(
            gate,
            "fail",
            "the loop, dense control, or candidate backend failed",
            {
                "candidate_status": candidate.get("status"),
                "candidate_error": candidate.get("error"),
                "loop_status": loop.get("status"),
                "loop_error": loop.get("error"),
                "dense_control_status": dense.get("status"),
                "dense_control_error": dense.get("error"),
            },
        )

    required_mode = str(gate.get("required_mode", "training"))
    minimum_tokens = int(gate.get("min_visual_tokens", 1))
    minimum_batch = int(gate.get("min_batch_size", 1))
    minimum_warmup = int(gate.get("min_warmup_iterations", 0))
    minimum_iterations = int(gate.get("min_measured_iterations", 1))
    minimum_rounds = int(gate.get("min_rounds", 1))
    evidence = {
        "benchmark_schema_version": report.get("schema_version"),
        "device": environment.get("device"),
        "device_name": environment.get("device_name"),
        "torch": environment.get("torch"),
        "cuda": environment.get("cuda"),
        "mode": benchmark_config.get("mode"),
        "precision": report.get("resolved_precision"),
        "visual_tokens": report.get("visual_tokens"),
        "batch_size": report.get("batch_size"),
        "warmup_iterations": benchmark_config.get("warmup_iterations"),
        "measured_iterations": benchmark_config.get("measured_iterations"),
        "rounds": benchmark_config.get("rounds"),
        "candidate_rounds": candidate.get("rounds"),
        "paired_rounds_vs_loop": candidate.get(
            "paired_rounds_vs_loop"
        ),
        "paired_rounds_vs_dense_adaptive": candidate.get(
            "paired_rounds_vs_dense_adaptive"
        ),
        "requested_backend": requested,
        "resolved_backend": candidate.get("resolved_backend"),
        "median_ms": candidate.get("median_ms"),
        "median_speedup_vs_loop": candidate.get(
            "median_speedup_vs_loop"
        ),
        "min_speedup_vs_loop": candidate.get("min_speedup_vs_loop"),
        "peak_memory_ratio_vs_loop": candidate.get(
            "peak_memory_ratio_vs_loop"
        ),
        "max_abs_delta_vs_loop": candidate.get(
            "max_abs_delta_vs_loop"
        ),
        "dense_control_requested_backend": dense_requested,
        "median_speedup_vs_dense_adaptive": candidate.get(
            "median_speedup_vs_dense_adaptive"
        ),
        "min_speedup_vs_dense_adaptive": candidate.get(
            "min_speedup_vs_dense_adaptive"
        ),
        "peak_memory_ratio_vs_dense_adaptive": candidate.get(
            "peak_memory_ratio_vs_dense_adaptive"
        ),
        "student_config_fingerprint": report.get(
            "student_config_fingerprint"
        ),
    }
    try:
        evidence_sufficient = (
            int(report["schema_version"]) >= minimum_schema_version
            and benchmark_config.get("mode") == required_mode
            and int(report["visual_tokens"]) >= minimum_tokens
            and int(report["batch_size"]) >= minimum_batch
            and int(benchmark_config["warmup_iterations"]) >= minimum_warmup
            and int(benchmark_config["measured_iterations"])
            >= minimum_iterations
            and int(benchmark_config["rounds"]) >= minimum_rounds
            and int(candidate["rounds"]) >= minimum_rounds
            and int(candidate["paired_rounds_vs_loop"])
            >= minimum_rounds
            and int(candidate["paired_rounds_vs_dense_adaptive"])
            >= minimum_rounds
            and candidate.get("median_speedup_vs_loop") is not None
            and candidate.get("min_speedup_vs_loop") is not None
            and candidate.get("peak_memory_ratio_vs_loop") is not None
            and candidate.get("max_abs_delta_vs_loop") is not None
            and candidate.get("median_speedup_vs_dense_adaptive")
            is not None
            and candidate.get("min_speedup_vs_dense_adaptive")
            is not None
            and candidate.get("peak_memory_ratio_vs_dense_adaptive")
            is not None
        )
    except (KeyError, TypeError, ValueError):
        evidence_sufficient = False
    if not evidence_sufficient:
        return _result(
            gate,
            "insufficient_evidence",
            "benchmark schema, dose, mode, or paired measurements are insufficient",
            evidence,
        )

    required_backend = str(gate.get("required_resolved_backend", "flex"))
    try:
        speedup = float(candidate["median_speedup_vs_loop"])
        minimum_observed_speedup = float(candidate["min_speedup_vs_loop"])
        memory_ratio = float(candidate["peak_memory_ratio_vs_loop"])
        numerical_delta = float(candidate["max_abs_delta_vs_loop"])
        dense_speedup = float(
            candidate["median_speedup_vs_dense_adaptive"]
        )
        dense_minimum_observed_speedup = float(
            candidate["min_speedup_vs_dense_adaptive"]
        )
        dense_memory_ratio = float(
            candidate["peak_memory_ratio_vs_dense_adaptive"]
        )
    except (TypeError, ValueError):
        return _result(
            gate,
            "insufficient_evidence",
            "loop-relative benchmark measurements must be numeric",
            evidence,
        )
    minimum_speedup = float(gate.get("min_median_speedup_vs_loop", 1.0))
    minimum_round_speedup = float(
        gate.get("min_round_speedup_vs_loop", 1.0)
    )
    maximum_memory_ratio = float(
        gate.get("max_peak_memory_ratio_vs_loop", 1.0)
    )
    maximum_delta = float(gate.get("max_abs_delta_vs_loop", 0.0))
    minimum_dense_speedup = float(
        gate.get("min_median_speedup_vs_dense_adaptive", 1.0)
    )
    minimum_dense_round_speedup = float(
        gate.get("min_round_speedup_vs_dense_adaptive", 1.0)
    )
    maximum_dense_memory_ratio = float(
        gate.get("max_peak_memory_ratio_vs_dense_adaptive", 1.0)
    )
    violations = []
    valid_measurements = (
        math.isfinite(speedup)
        and speedup > 0
        and math.isfinite(minimum_observed_speedup)
        and minimum_observed_speedup > 0
        and math.isfinite(memory_ratio)
        and memory_ratio >= 0
        and math.isfinite(numerical_delta)
        and numerical_delta >= 0
        and math.isfinite(dense_speedup)
        and dense_speedup > 0
        and math.isfinite(dense_minimum_observed_speedup)
        and dense_minimum_observed_speedup > 0
        and math.isfinite(dense_memory_ratio)
        and dense_memory_ratio >= 0
    )
    if not valid_measurements:
        violations.append("invalid_measurement")
    if candidate.get("resolved_backend") != required_backend:
        violations.append("resolved_backend")
    if valid_measurements and speedup < minimum_speedup:
        violations.append("median_speedup")
    if (
        valid_measurements
        and minimum_observed_speedup < minimum_round_speedup
    ):
        violations.append("round_speedup")
    if valid_measurements and memory_ratio > maximum_memory_ratio:
        violations.append("peak_memory")
    if valid_measurements and numerical_delta > maximum_delta:
        violations.append("numerical_parity")
    if valid_measurements and dense_speedup < minimum_dense_speedup:
        violations.append("dense_adaptive_speedup")
    if (
        valid_measurements
        and dense_minimum_observed_speedup
        < minimum_dense_round_speedup
    ):
        violations.append("dense_adaptive_round_speedup")
    if (
        valid_measurements
        and dense_memory_ratio > maximum_dense_memory_ratio
    ):
        violations.append("dense_adaptive_peak_memory")
    evidence.update(
        {
            "required_resolved_backend": required_backend,
            "min_benchmark_schema_version": minimum_schema_version,
            "min_rounds": minimum_rounds,
            "min_median_speedup_vs_loop": minimum_speedup,
            "min_round_speedup_vs_loop": minimum_round_speedup,
            "max_peak_memory_ratio_vs_loop": maximum_memory_ratio,
            "max_abs_delta_threshold": maximum_delta,
            "min_median_speedup_vs_dense_adaptive": minimum_dense_speedup,
            "min_round_speedup_vs_dense_adaptive": (
                minimum_dense_round_speedup
            ),
            "max_peak_memory_ratio_vs_dense_adaptive": (
                maximum_dense_memory_ratio
            ),
            "violations": violations,
        }
    )
    return _result(
        gate,
        "fail" if violations else "pass",
        (
            "target-device visual backend violates deployment thresholds"
            if violations
            else "target-device visual backend meets parity, speed, and memory thresholds"
        ),
        evidence,
    )


def evaluate_visual_efficiency_gate(
    blueprint: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate only the target-device visual preflight gate."""

    gates = blueprint.get("evaluation_gates")
    if (
        not isinstance(gates, Sequence)
        or isinstance(gates, (str, bytes))
    ):
        raise ValueError("blueprint evaluation_gates must be a sequence")
    gate = next(
        (
            item
            for item in gates
            if isinstance(item, Mapping)
            and item.get("id") == "visual_efficiency"
        ),
        None,
    )
    if gate is None:
        raise ValueError("blueprint has no visual_efficiency gate")
    return _visual_efficiency(gate, blueprint, report)


def _training_feasibility(
    gate: Mapping[str, Any],
    blueprint: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pipeline = (
        blueprint.get("training", {})
        .get("pretraining", {})
        .get("input_pipeline", {})
    )
    optimizer = (
        blueprint.get("training", {})
        .get("pretraining", {})
        .get("optimizer", {})
    )
    if pipeline.get("visual_sequence_mode") != "packed":
        return _result(
            gate,
            "insufficient_evidence",
            "the full training probe currently evaluates packed visual execution",
            {
                "visual_sequence_mode": pipeline.get(
                    "visual_sequence_mode"
                )
            },
        )
    if report is None:
        return _result(
            gate,
            "insufficient_evidence",
            "a target-device full-student training-step benchmark is required",
        )
    minimum_schema = int(gate.get("min_benchmark_schema_version", 1))
    try:
        schema = int(report.get("schema_version", 0))
    except (TypeError, ValueError):
        schema = 0
    if (
        schema < minimum_schema
        or report.get("scope")
        != "full_student_multimodal_training_step"
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "training feasibility benchmark schema or scope is invalid",
        )
    expected_student = StudentConfig.from_blueprint(dict(blueprint))
    expected_fingerprint = student_config_fingerprint(expected_student)
    if (
        report.get("student_config") != expected_student.to_dict()
        or report.get("student_config_fingerprint")
        != expected_fingerprint
    ):
        return _result(
            gate,
            "fail",
            "training feasibility evidence was measured on a different student configuration",
            {
                "reported_fingerprint": report.get(
                    "student_config_fingerprint"
                ),
                "expected_fingerprint": expected_fingerprint,
            },
        )
    environment = report.get("environment")
    benchmark = report.get("benchmark_config")
    if not isinstance(environment, Mapping) or not isinstance(
        benchmark, Mapping
    ):
        return _result(
            gate,
            "insufficient_evidence",
            "training feasibility benchmark metadata is incomplete",
        )
    required_device = str(gate.get("required_device_type", "cuda"))
    if environment.get("device_type") != required_device:
        return _result(
            gate,
            "insufficient_evidence",
            f"training feasibility benchmark must run on {required_device}",
            {
                "reported_device_type": environment.get("device_type"),
                "reported_device": environment.get("device"),
            },
        )
    if report.get("status") != "ok":
        return _result(
            gate,
            "fail",
            "the full-student training micro-step failed",
            {
                "error_type": report.get("error_type"),
                "error": report.get("error"),
                "oom": report.get("oom"),
                "failure_memory": report.get("failure_memory"),
            },
        )
    checkpointing = blueprint["training"]["activation_checkpointing"]
    expected_checkpointing = {
        "enabled": bool(checkpointing["enabled"]),
        "components": list(checkpointing["components"]),
        "use_reentrant": bool(checkpointing["use_reentrant"]),
    }
    if report.get("gradient_checkpointing") != expected_checkpointing:
        return _result(
            gate,
            "fail",
            "training feasibility evidence used a different gradient-checkpointing contract",
            {
                "reported_gradient_checkpointing": report.get(
                    "gradient_checkpointing"
                ),
                "expected_gradient_checkpointing": (
                    expected_checkpointing
                ),
            },
        )
    expected_optimizer = OptimizerSpec.from_mapping(optimizer).to_dict()
    optimizer_runtime = report.get("optimizer_runtime")
    reported_optimizer = benchmark.get("optimizer")
    optimizer_contract_matches = (
        reported_optimizer == expected_optimizer
        and isinstance(optimizer_runtime, Mapping)
        and optimizer_runtime.get("spec") == expected_optimizer
        and isinstance(optimizer_runtime.get("implementation"), str)
        and bool(optimizer_runtime["implementation"])
        and (
            expected_optimizer["name"] != "adamw_8bit"
            or bool(optimizer_runtime.get("bitsandbytes_version"))
        )
    )
    if not optimizer_contract_matches:
        return _result(
            gate,
            "fail",
            "training feasibility evidence used a different optimizer contract",
            {
                "reported_optimizer": reported_optimizer,
                "reported_optimizer_runtime": optimizer_runtime,
                "expected_optimizer": expected_optimizer,
            },
        )

    minimum_text = int(gate.get("min_text_tokens", 1))
    minimum_visual = int(gate.get("min_visual_tokens_per_sample", 1))
    minimum_warmup = int(gate.get("min_warmup_steps", 0))
    minimum_measured = int(gate.get("min_measured_steps", 1))
    required_backend = str(
        gate.get("required_resolved_visual_attention_backend", "flex")
    )
    required_precision = str(
        gate.get("required_precision", optimizer.get("precision", "auto"))
    )
    required_micro_batch = int(
        gate.get(
            "required_micro_batch_size",
            optimizer.get("micro_batch_size", 1),
        )
    )
    required_grad_accum = int(optimizer.get("grad_accum_steps", 1))
    maximum_peak_fraction = float(
        gate.get("max_peak_reserved_fraction", 0.95)
    )
    effective_memory = report.get("effective_peak_memory")
    optimizer_state = report.get("optimizer_state")
    training_flops = report.get("training_flops_per_microbatch")
    expected_contrastive_memory = (
        blueprint["training"]["pretraining"].get(
            "contrastive_memory",
            {
                "enabled": False,
                "size": 0,
                "min_negatives": 1,
                "scope": "local_fifo",
            },
        )
    )
    measured_records = report.get("measured_steps") or []
    last_measured = (
        measured_records[-1]
        if isinstance(measured_records, list) and measured_records
        else {}
    )
    evidence = {
        "benchmark_schema_version": schema,
        "device": environment.get("device"),
        "device_name": environment.get("device_name"),
        "device_total_memory_bytes": environment.get(
            "device_total_memory_bytes"
        ),
        "precision": benchmark.get("resolved_precision"),
        "micro_batch_size": benchmark.get("micro_batch_size"),
        "grad_accum_steps": benchmark.get("grad_accum_steps"),
        "microbatches_per_probe_step": benchmark.get(
            "microbatches_per_probe_step"
        ),
        "short_final_batch_gradient_correction": benchmark.get(
            "short_final_batch_gradient_correction"
        ),
        "text_tokens": benchmark.get("text_tokens"),
        "patch_grid": benchmark.get("patch_grid"),
        "visual_tokens_per_sample": (
            int(benchmark["patch_grid"][0])
            * int(benchmark["patch_grid"][1])
            if isinstance(benchmark.get("patch_grid"), Sequence)
            and not isinstance(benchmark.get("patch_grid"), (str, bytes))
            and len(benchmark["patch_grid"]) == 2
            else None
        ),
        "warmup_steps": benchmark.get("warmup_steps"),
        "measured_steps": benchmark.get("measured_steps"),
        "resolved_visual_attention_backend": report.get(
            "resolved_visual_attention_backend"
        ),
        "median_step_ms": report.get("median_step_ms"),
        "p95_step_ms": report.get("p95_step_ms"),
        "steps_per_second": report.get("steps_per_second"),
        "all_finite": report.get("all_finite"),
        "all_optimizer_steps_succeeded": report.get(
            "all_optimizer_steps_succeeded"
        ),
        "gradient_checkpointing": report.get(
            "gradient_checkpointing"
        ),
        "optimizer_state": optimizer_state,
        "optimizer": benchmark.get("optimizer"),
        "optimizer_runtime": optimizer_runtime,
        "training_flops_per_microbatch": training_flops,
        "contrastive_memory": benchmark.get("contrastive_memory"),
        "contrastive_memory_size": last_measured.get(
            "contrastive_memory_size"
        ),
        "contrastive_negative_pairs": last_measured.get(
            "contrastive_negative_pairs"
        ),
        "setup_memory": report.get("setup_memory"),
        "materialization_memory": report.get("materialization_memory"),
        "steady_state_memory": report.get("steady_state_memory"),
        "effective_peak_memory": effective_memory,
        "student_config_fingerprint": report.get(
            "student_config_fingerprint"
        ),
        "parameter_count": report.get("parameter_count"),
    }
    try:
        peak_fraction = float(effective_memory["peak_reserved_fraction"])
        median_ms = float(report["median_step_ms"])
        p95_ms = float(report["p95_step_ms"])
        state_parameters = int(optimizer_state["parameter_states"])
        state_step = float(optimizer_state["max_step"])
        algorithmic_flops = int(training_flops["algorithmic"])
        checkpoint_recompute_flops = int(
            training_flops["checkpoint_recompute"]
        )
        executed_flops = int(training_flops["executed"])
        visual_tokens = int(evidence["visual_tokens_per_sample"])
        dose_sufficient = (
            int(benchmark["text_tokens"]) >= minimum_text
            and visual_tokens >= minimum_visual
            and int(benchmark["micro_batch_size"]) == required_micro_batch
            and int(benchmark["grad_accum_steps"]) == required_grad_accum
            and int(benchmark["microbatches_per_probe_step"]) == 1
            and bool(
                benchmark["short_final_batch_gradient_correction"]
            )
            and int(benchmark["warmup_steps"]) >= minimum_warmup
            and int(benchmark["measured_steps"]) >= minimum_measured
            and benchmark["resolved_precision"] == required_precision
            and bool(benchmark["gradient_checkpointing"])
            == bool(gate["require_gradient_checkpointing"])
            and list(
                benchmark["gradient_checkpointing_components"]
            )
            == list(
                gate[
                    "required_gradient_checkpointing_components"
                ]
            )
            and bool(
                benchmark[
                    "gradient_checkpointing_use_reentrant"
                ]
            )
            == bool(
                gate[
                    "required_gradient_checkpointing_use_reentrant"
                ]
            )
            and benchmark.get("contrastive_memory")
            == expected_contrastive_memory
            and (
                not bool(expected_contrastive_memory.get("enabled"))
                or (
                    int(last_measured["contrastive_memory_size"])
                    == int(expected_contrastive_memory["size"])
                    and int(last_measured["contrastive_negative_pairs"]) > 0
                )
            )
        )
        numeric_evidence = (
            math.isfinite(peak_fraction)
            and 0 <= peak_fraction <= 1
            and math.isfinite(median_ms)
            and median_ms > 0
            and math.isfinite(p95_ms)
            and p95_ms > 0
            and state_parameters > 0
            and state_step
            >= int(benchmark["warmup_steps"])
            + int(benchmark["measured_steps"])
            and algorithmic_flops > 0
            and (
                checkpoint_recompute_flops > 0
                if bool(gate["require_gradient_checkpointing"])
                else checkpoint_recompute_flops == 0
            )
            and executed_flops
            == algorithmic_flops + checkpoint_recompute_flops
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        dose_sufficient = False
        numeric_evidence = False
        peak_fraction = math.inf
    if not dose_sufficient or not numeric_evidence:
        return _result(
            gate,
            "insufficient_evidence",
            "training benchmark dose, optimizer state, latency, or memory evidence is insufficient",
            evidence,
        )
    violations = []
    if report.get("resolved_visual_attention_backend") != required_backend:
        violations.append("resolved_visual_attention_backend")
    if not report.get("all_finite"):
        violations.append("non_finite_training_values")
    if not report.get("all_optimizer_steps_succeeded"):
        violations.append("optimizer_step")
    if peak_fraction > maximum_peak_fraction:
        violations.append("peak_reserved_memory")
    evidence.update(
        {
            "required_precision": required_precision,
            "required_micro_batch_size": required_micro_batch,
            "required_grad_accum_steps": required_grad_accum,
            "required_resolved_visual_attention_backend": (
                required_backend
            ),
            "require_gradient_checkpointing": bool(
                gate["require_gradient_checkpointing"]
            ),
            "required_gradient_checkpointing_components": list(
                gate[
                    "required_gradient_checkpointing_components"
                ]
            ),
            "required_gradient_checkpointing_use_reentrant": bool(
                gate[
                    "required_gradient_checkpointing_use_reentrant"
                ]
            ),
            "required_contrastive_memory": expected_contrastive_memory,
            "min_text_tokens": minimum_text,
            "min_visual_tokens_per_sample": minimum_visual,
            "min_warmup_steps": minimum_warmup,
            "min_measured_steps": minimum_measured,
            "max_peak_reserved_fraction": maximum_peak_fraction,
            "violations": violations,
        }
    )
    return _result(
        gate,
        "fail" if violations else "pass",
        (
            "full-student training step violates deployment thresholds"
            if violations
            else "full-student forward, backward, and optimizer step fit the target device"
        ),
        evidence,
    )


def evaluate_training_feasibility_gate(
    blueprint: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate only the full-student target-device training preflight."""

    gates = blueprint.get("evaluation_gates")
    if (
        not isinstance(gates, Sequence)
        or isinstance(gates, (str, bytes))
    ):
        raise ValueError("blueprint evaluation_gates must be a sequence")
    gate = next(
        (
            item
            for item in gates
            if isinstance(item, Mapping)
            and item.get("id") == "training_feasibility"
        ),
        None,
    )
    if gate is None:
        raise ValueError("blueprint has no training_feasibility gate")
    return _training_feasibility(gate, blueprint, report)


def evaluate_deployment_gates(
    blueprint: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
    current_comparison: Mapping[str, Any],
    current_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline_comparison: Mapping[str, Any] | None = None,
    baseline_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    monolingual_control_comparison: Mapping[str, Any] | None = None,
    visual_backend_report: Mapping[str, Any] | None = None,
    training_feasibility_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every declared gate without treating missing evidence as success."""

    current_heldout = _heldout_rows(current_rows)
    baseline_heldout = _heldout_rows(baseline_rows)
    pairs, pair_error = _matched_rows(current_heldout, baseline_heldout)
    results = []
    for gate in blueprint.get("evaluation_gates", ()):
        gate_id = gate.get("id")
        if gate_id == "parameter_budget":
            result = _parameter_budget(gate, parameter_counts)
        elif gate_id == "generalization":
            result = _generalization(
                gate,
                current_comparison,
                baseline_comparison,
                pairs,
                pair_error,
            )
        elif gate_id == "grounding":
            result = _grounding(
                gate,
                pairs,
                pair_error,
                baseline_comparison is not None,
            )
        elif gate_id == "reasoning":
            result = _reasoning(
                gate,
                pairs,
                pair_error,
                baseline_comparison is not None,
            )
        elif gate_id == "multilingual":
            result = _multilingual(
                gate,
                current_comparison,
                monolingual_control_comparison,
            )
        elif gate_id == "reliability":
            result = _reliability(
                gate,
                current_comparison,
                baseline_comparison,
                pairs,
                pair_error,
            )
        elif gate_id == "generation_stability":
            result = _generation_stability(
                gate,
                current_comparison,
                baseline_comparison,
                pairs,
                pair_error,
            )
        elif gate_id == "visual_efficiency":
            result = evaluate_visual_efficiency_gate(
                blueprint,
                visual_backend_report,
            )
        elif gate_id == "training_feasibility":
            result = evaluate_training_feasibility_gate(
                blueprint,
                training_feasibility_report,
            )
        else:
            result = _result(
                gate,
                "insufficient_evidence",
                f"no evaluator is registered for gate {gate_id!r}",
            )
        results.append(result)
    statuses = [result["status"] for result in results]
    overall = (
        "fail"
        if "fail" in statuses
        else "insufficient_evidence"
        if "insufficient_evidence" in statuses
        else "pass"
    )
    return {
        "schema_version": 1,
        "overall_status": overall,
        "counts": {
            status: statuses.count(status)
            for status in ("pass", "fail", "insufficient_evidence")
        },
        "gates": results,
    }


def load_evaluation_artifacts(
    root: str | Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load a comparison and per-sample split files from an evaluation root."""

    root = Path(root)
    comparison_path = root / "comparison.json"
    if not comparison_path.is_file():
        raise ValueError(f"missing evaluation comparison: {comparison_path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*/per_sample.jsonl")):
        rows[path.parent.name] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return comparison, rows


def load_visual_backend_report(path: str | Path) -> dict[str, Any]:
    """Load one visual backend report without treating malformed JSON as evidence."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing visual backend benchmark: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("visual backend benchmark root must be a mapping")
    return report


def load_training_feasibility_report(path: str | Path) -> dict[str, Any]:
    """Load one full-student training report as deployment evidence."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing training feasibility benchmark: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("training feasibility benchmark root must be a mapping")
    return report


def write_gate_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write a machine-readable gate report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
