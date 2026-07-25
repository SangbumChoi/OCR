"""Fail-closed pilot-to-confirmatory submission evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .pilot_execution import attestation_is_sealed
from .sweep import SweepPlan


BASELINE = "lfm_language_only"
CANDIDATE = "lfm_smol_dual"
PILOT = "docvlm-smol-vision-transfer-pilot"
CONFIRMATORY = "docvlm-smol-vision-transfer-sweep"


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _fingerprint_valid(value: dict[str, Any]) -> bool:
    observed = value.get("fingerprint")
    body = dict(value)
    body.pop("fingerprint", None)
    return observed == _fingerprint(body)


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    status: str,
    evidence: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "evidence": evidence,
        }
    )


def _pilot_attestations(
    comparison: dict[str, Any],
    expected_runs: set[str],
) -> tuple[bool, dict[str, str], list[str], list[str]]:
    raw = comparison.get("execution_attestations")
    if not isinstance(raw, dict):
        return False, {}, sorted(expected_runs), []
    missing = sorted(expected_runs - set(raw))
    unexpected = sorted(set(raw) - expected_runs)
    sealed = {
        run_id: str(attestation["attestation_sha256"])
        for run_id, attestation in raw.items()
        if run_id in expected_runs and attestation_is_sealed(attestation)
    }
    return (
        not missing
        and not unexpected
        and len(sealed) == len(expected_runs),
        sealed,
        missing,
        unexpected,
    )


def audit_smol_confirmatory_submission(
    pilot_plan: SweepPlan,
    confirmatory_plan: SweepPlan,
    *,
    pilot_readiness: dict[str, Any],
    pilot_execution: dict[str, Any],
    pilot_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Authorize the three-seed budget only from sealed pilot evidence."""

    pilot_readiness = _mapping(pilot_readiness, "pilot readiness")
    pilot_execution = _mapping(pilot_execution, "pilot execution")
    checks: list[dict[str, Any]] = []

    readiness_sweep = pilot_readiness.get("sweep") or {}
    readiness_fingerprint_valid = _fingerprint_valid(pilot_readiness)
    readiness_topology_matches = (
        readiness_sweep.get("name") == pilot_plan.name == PILOT
        and readiness_sweep.get("baseline")
        == pilot_plan.baseline
        == BASELINE
        and readiness_sweep.get("replicates")
        == list(pilot_plan.replicates)
        and set(readiness_sweep.get("arms") or [])
        == {variant.arm_id for variant in pilot_plan.variants}
    )
    readiness_pass = (
        pilot_readiness.get("overall_status") == "pass"
        and pilot_readiness.get("pilot_submission_authorized") is True
        and readiness_fingerprint_valid
        and readiness_topology_matches
    )
    _check(
        checks,
        "pilot_submission_contract",
        "pass" if readiness_pass else "fail",
        {
            "status": pilot_readiness.get("overall_status"),
            "fingerprint": pilot_readiness.get("fingerprint"),
            "authorized": pilot_readiness.get(
                "pilot_submission_authorized"
            ),
            "fingerprint_valid": readiness_fingerprint_valid,
            "topology_matches": readiness_topology_matches,
        },
    )

    execution_fingerprint_valid = _fingerprint_valid(pilot_execution)
    execution_complete = (
        pilot_execution.get("state") == "completed_attested"
        and pilot_execution.get("training_execution_attested") is True
        and execution_fingerprint_valid
    )
    _check(
        checks,
        "sealed_pilot_execution",
        "pass" if execution_complete else "pending",
        {
            "state": pilot_execution.get("state"),
            "fingerprint": pilot_execution.get("fingerprint"),
            "training_execution_attested": pilot_execution.get(
                "training_execution_attested"
            ),
            "fingerprint_valid": execution_fingerprint_valid,
        },
    )

    confirmatory_topology = (
        confirmatory_plan.name == CONFIRMATORY
        and confirmatory_plan.baseline == BASELINE
        and tuple(confirmatory_plan.replicates)
        == ("seed_0", "seed_1", "seed_2")
        and len(confirmatory_plan.variants) == 6
        and confirmatory_plan.promotion is not None
        and confirmatory_plan.promotion.minimum_replicates == 3
        and CANDIDATE
        in confirmatory_plan.promotion.eligible_variants
    )
    _check(
        checks,
        "confirmatory_topology",
        "pass" if confirmatory_topology else "fail",
        {
            "sweep": confirmatory_plan.name,
            "fingerprint": confirmatory_plan.fingerprint,
            "baseline": confirmatory_plan.baseline,
            "replicates": list(confirmatory_plan.replicates),
            "run_count": len(confirmatory_plan.variants),
        },
    )

    if pilot_comparison is None:
        for check_id in (
            "pilot_comparison_identity",
            "pilot_attestation_linkage",
            "candidate_required_gates",
            "positive_screening_effect",
        ):
            _check(
                checks,
                check_id,
                "pending",
                {"reason": "pilot comparison is not available"},
            )
    else:
        comparison = _mapping(pilot_comparison, "pilot comparison")
        identity_pass = (
            comparison.get("schema_version") == 6
            and comparison.get("sweep") == PILOT
            and comparison.get("sweep_fingerprint")
            == pilot_plan.fingerprint
            and comparison.get("baseline") == BASELINE
            and comparison.get("replicates") == ["seed_0"]
        )
        _check(
            checks,
            "pilot_comparison_identity",
            "pass" if identity_pass else "fail",
            {
                "schema_version": comparison.get("schema_version"),
                "sweep": comparison.get("sweep"),
                "sweep_fingerprint": comparison.get(
                    "sweep_fingerprint"
                ),
                "expected_sweep_fingerprint": pilot_plan.fingerprint,
                "baseline": comparison.get("baseline"),
                "replicates": comparison.get("replicates"),
            },
        )

        expected_runs = {
            f"{arm}--seed_0" for arm in (BASELINE, CANDIDATE)
        }
        attestations_pass, attestations, missing, unexpected = (
            _pilot_attestations(comparison, expected_runs)
        )
        by_arm = {
            run_id.removesuffix("--seed_0"): digest
            for run_id, digest in attestations.items()
        }
        comparison_attestation_fingerprint = (
            _fingerprint(by_arm) if by_arm else None
        )
        execution_attestation_fingerprint = (
            (pilot_execution.get("local") or {}).get(
                "execution_attestation_fingerprint"
            )
        )
        linkage_pass = (
            execution_complete
            and attestations_pass
            and comparison_attestation_fingerprint
            == execution_attestation_fingerprint
        )
        _check(
            checks,
            "pilot_attestation_linkage",
            "pass"
            if linkage_pass
            else "pending"
            if not execution_complete
            else "fail",
            {
                "comparison_attestation_fingerprint": (
                    comparison_attestation_fingerprint
                ),
                "execution_attestation_fingerprint": (
                    execution_attestation_fingerprint
                ),
                "missing_runs": missing,
                "unexpected_runs": unexpected,
                "sealed_run_count": len(attestations),
            },
        )

        variants = comparison.get("variants") or {}
        baseline_record = variants.get(BASELINE) or {}
        candidate_record = variants.get(CANDIDATE) or {}
        required_gates = (
            []
            if confirmatory_plan.promotion is None
            else list(confirmatory_plan.promotion.required_gates)
        )
        gate_statuses = candidate_record.get("gate_statuses") or {}
        failed_or_missing_gates = sorted(
            gate
            for gate in required_gates
            if gate_statuses.get(gate) != "pass"
        )
        gates_pass = (
            candidate_record.get("gate_status") == "pass"
            and not failed_or_missing_gates
        )
        _check(
            checks,
            "candidate_required_gates",
            "pass" if gates_pass else "fail",
            {
                "gate_status": candidate_record.get("gate_status"),
                "required_gate_count": len(required_gates),
                "failed_or_missing_gates": failed_or_missing_gates,
            },
        )

        baseline_score = (baseline_record.get("metrics") or {}).get(
            "heldout_score"
        )
        candidate_score = (candidate_record.get("metrics") or {}).get(
            "heldout_score"
        )
        scores_present = isinstance(
            baseline_score, (int, float)
        ) and isinstance(candidate_score, (int, float))
        heldout_delta = (
            float(candidate_score) - float(baseline_score)
            if scores_present
            else None
        )
        effect_pass = (
            heldout_delta is not None
            and heldout_delta > 0.0
            and candidate_record.get("heldout_score_conclusion")
            == "improved"
        )
        _check(
            checks,
            "positive_screening_effect",
            "pass" if effect_pass else "fail",
            {
                "baseline_heldout_score": baseline_score,
                "candidate_heldout_score": candidate_score,
                "heldout_delta": heldout_delta,
                "conclusion": candidate_record.get(
                    "heldout_score_conclusion"
                ),
            },
        )

    counts = {
        status: sum(check["status"] == status for check in checks)
        for status in ("pass", "pending", "fail")
    }
    authorized = counts["pending"] == 0 and counts["fail"] == 0
    result = {
        "schema_version": 1,
        "claim_scope": "smol_confirmatory_submission_only",
        "pilot": PILOT,
        "confirmatory_sweep": CONFIRMATORY,
        "overall_status": (
            "pass"
            if authorized
            else "fail"
            if counts["fail"]
            else "pending"
        ),
        "counts": counts,
        "checks": checks,
        "confirmatory_submission_authorized": authorized,
        "quality_claim_authorized": False,
        "promotion_claim_authorized": False,
        "next_action": (
            "Run the three-seed confirmatory sweep on native-BF16 hardware."
            if authorized
            else "Complete and seal a positive Smol vision-transfer pilot."
        ),
    }
    result["fingerprint"] = _fingerprint(result)
    return result
