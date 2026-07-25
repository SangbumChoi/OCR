"""Build compact quality and promotion evidence from a sealed sweep."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sealed_attestation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    scope = value.get("claim_scope")
    quality_authorized = value.get("quality_claim_authorized")
    valid_scope = (
        scope == "execution_contract_only"
        and quality_authorized is False
    ) or (
        scope == "deployment_capability"
        and quality_authorized is True
    )
    digest = str(value.get("attestation_sha256") or "")
    return (
        value.get("contract_status") == "pass"
        and valid_scope
        and len(digest) == 71
        and digest.startswith("sha256:")
        and int(value.get("stage_count") or 0) > 0
    )


def build_confirmatory_evidence(
    comparison: dict[str, Any],
    *,
    expected_sweep: str,
    expected_sweep_fingerprint: str,
    baseline: str,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive bounded claim artifacts without copying metric tables."""

    if comparison.get("schema_version") != 6:
        raise ValueError("confirmatory comparison schema_version must be 6")
    if comparison.get("sweep") != expected_sweep:
        raise ValueError("confirmatory comparison sweep identity mismatch")
    if comparison.get("sweep_fingerprint") != expected_sweep_fingerprint:
        raise ValueError("confirmatory comparison fingerprint mismatch")
    if comparison.get("baseline") != baseline:
        raise ValueError("confirmatory comparison baseline mismatch")
    replicates = comparison.get("replicates")
    if (
        not isinstance(replicates, list)
        or len(replicates) < 3
        or len(set(str(item) for item in replicates)) != len(replicates)
    ):
        raise ValueError(
            "confirmatory comparison requires at least three unique replicates"
        )
    attestations = comparison.get("execution_attestations")
    if not isinstance(attestations, dict) or not attestations:
        raise ValueError(
            "confirmatory comparison has no execution attestations"
        )
    variants = comparison.get("variants")
    if not isinstance(variants, dict) or candidate not in variants:
        raise ValueError("confirmatory comparison has no candidate record")
    promotion = comparison.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError("confirmatory comparison has no promotion result")
    contract = promotion.get("contract") or {}
    required_gates = [str(item) for item in contract.get("required_gates") or []]
    if not required_gates:
        raise ValueError("confirmatory promotion has no required gates")

    candidate_record = variants[candidate]
    if not isinstance(candidate_record, dict):
        raise ValueError("confirmatory candidate record must be a mapping")
    gate_statuses = candidate_record.get("gate_statuses") or {}
    failed_or_missing_gates = sorted(
        gate
        for gate in required_gates
        if gate_statuses.get(gate) != "pass"
    )
    unsealed_runs = sorted(
        run_id
        for run_id, attestation in attestations.items()
        if not _sealed_attestation(attestation)
    )
    expected_runs = {
        f"{arm}--{replicate}"
        for arm in (baseline, candidate)
        for replicate in replicates
    }
    missing_runs = sorted(expected_runs - set(attestations))
    unexpected_runs = sorted(set(attestations) - expected_runs)
    execution_complete = (
        not unsealed_runs
        and not missing_runs
        and not unexpected_runs
    )
    quality_authorized = (
        execution_complete
        and candidate_record.get("gate_status") == "pass"
        and not failed_or_missing_gates
    )
    attestation_summary = {
        run_id: {
            "attestation_sha256": attestation.get("attestation_sha256"),
            "contract_status": attestation.get("contract_status"),
        }
        for run_id, attestation in sorted(attestations.items())
        if isinstance(attestation, dict)
    }
    quality = {
        "schema_version": 1,
        "claim_scope": "heldout_quality_evidence",
        "sweep": expected_sweep,
        "sweep_fingerprint": comparison.get("sweep_fingerprint"),
        "baseline": baseline,
        "candidate": candidate,
        "replicates": len(replicates),
        "execution_attestation_fingerprint": _fingerprint(
            attestation_summary
        ),
        "execution_complete": execution_complete,
        "gate_status": (
            "pass" if quality_authorized else "fail"
        ),
        "required_gates": required_gates,
        "failed_or_missing_gates": failed_or_missing_gates,
        "missing_runs": missing_runs,
        "unexpected_runs": unexpected_runs,
        "unsealed_runs": unsealed_runs,
        "quality_claim_authorized": quality_authorized,
    }
    quality["fingerprint"] = _fingerprint(quality)

    multiple = promotion.get("multiple_comparisons") or {}
    selected = [str(item) for item in promotion.get("selected_variants") or []]
    multiplicity_controlled = (
        multiple.get("method")
        == "bonferroni_one_sided_percentile_bootstrap"
        and int(multiple.get("comparison_count") or 0) > 0
        and 0 < float(multiple.get("familywise_alpha") or 0.0) < 0.5
    )
    promotion_authorized = (
        quality_authorized
        and promotion.get("status") == "promote"
        and candidate in selected
        and multiplicity_controlled
        and len(replicates) >= int(contract.get("minimum_replicates") or 3)
    )
    promotion_evidence = {
        "schema_version": 1,
        "claim_scope": "multi_seed_promotion_evidence",
        "sweep": expected_sweep,
        "sweep_fingerprint": comparison.get("sweep_fingerprint"),
        "baseline": baseline,
        "candidate": candidate,
        "replicates": len(replicates),
        "quality_evidence_fingerprint": quality["fingerprint"],
        "promotion_status": promotion.get("status"),
        "selected_variants": selected,
        "multiplicity_controlled": multiplicity_controlled,
        "multiple_comparisons": {
            "method": multiple.get("method"),
            "comparison_count": multiple.get("comparison_count"),
            "familywise_alpha": multiple.get("familywise_alpha"),
        },
        "promotion_claim_authorized": promotion_authorized,
    }
    promotion_evidence["fingerprint"] = _fingerprint(
        promotion_evidence
    )
    return quality, promotion_evidence
