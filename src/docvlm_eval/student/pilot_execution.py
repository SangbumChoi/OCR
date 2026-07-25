"""Compact execution-state evidence for the native LFM transfer pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PilotExecutionProfile:
    name: str
    expected_arms: frozenset[str]
    claim_scope: str
    next_action: str


LFM_PROFILE = PilotExecutionProfile(
    name="docvlm-lfm-language-transfer-pilot",
    expected_arms=frozenset(
        {"native_random", "lfm_random", "lfm_strict_transfer"}
    ),
    claim_scope="native_lfm_pilot_execution_state_only",
    next_action=(
        "Run notebooks/lfm_selective_transfer_pilot.ipynb on "
        "native-BF16 hardware."
    ),
)
SMOL_PROFILE = PilotExecutionProfile(
    name="docvlm-smol-vision-transfer-pilot",
    expected_arms=frozenset({"lfm_language_only", "lfm_smol_dual"}),
    claim_scope="smol_vision_pilot_execution_state_only",
    next_action=(
        "Run notebooks/smol_vision_transfer_pilot.ipynb on "
        "native-BF16 hardware."
    ),
)
PILOT_NAME = LFM_PROFILE.name
EXPECTED_ARMS = LFM_PROFILE.expected_arms


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


def _pilot_wandb_runs(
    snapshot: dict[str, Any],
    profile: PilotExecutionProfile,
) -> list[dict[str, Any]]:
    prefix = f"{profile.name}--"
    runs = snapshot.get("runs")
    if not isinstance(runs, list):
        raise ValueError("W&B snapshot runs must be a list")
    return [
        run
        for run in runs
        if isinstance(run, dict)
        and str(run.get("name") or "").startswith(prefix)
    ]


def attestation_is_sealed(value: Any) -> bool:
    """Accept a valid execution-only or deployment-capability seal."""

    if not isinstance(value, dict):
        return False
    digest = str(value.get("attestation_sha256") or "")
    scope = value.get("claim_scope")
    quality_authorized = value.get("quality_claim_authorized")
    valid_scope = (
        scope == "execution_contract_only"
        and quality_authorized is False
    ) or (
        scope == "deployment_capability"
        and quality_authorized is True
    )
    return (
        value.get("contract_status") == "pass"
        and valid_scope
        and len(digest) == 71
        and digest.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in digest[7:])
        and isinstance(value.get("stage_count"), int)
        and value["stage_count"] > 0
    )


def _local_state(
    summary: dict[str, Any] | None,
    expected_arms: frozenset[str],
) -> dict[str, Any]:
    if summary is None:
        return {
            "summary_present": False,
            "status": "absent",
            "observed_arms": [],
            "completed_arms": [],
            "attested_arms": [],
            "unexpected_arms": [],
            "duplicate_arms": [],
            "attestation_sha256_by_arm": {},
            "execution_attestation_fingerprint": None,
        }

    variants = summary.get("variants")
    if not isinstance(variants, list):
        raise ValueError("local sweep summary variants must be a list")
    observed: set[str] = set()
    completed: set[str] = set()
    attested: set[str] = set()
    unexpected: set[str] = set()
    arm_counts: Counter[str] = Counter()
    attestation_sha256_by_arm: dict[str, str] = {}
    for item in variants:
        if not isinstance(item, dict):
            continue
        arm = str(item.get("variant") or "")
        if arm not in expected_arms:
            unexpected.add(arm or "<missing>")
            continue
        arm_counts[arm] += 1
        observed.add(arm)
        if item.get("status") == "completed":
            completed.add(arm)
            attestation = item.get("execution_attestation")
            if attestation_is_sealed(attestation):
                attested.add(arm)
                attestation_sha256_by_arm[arm] = str(
                    attestation["attestation_sha256"]
                )
    return {
        "summary_present": True,
        "status": str(summary.get("status") or "unknown"),
        "observed_arms": sorted(observed),
        "completed_arms": sorted(completed),
        "attested_arms": sorted(attested),
        "unexpected_arms": sorted(unexpected),
        "duplicate_arms": sorted(
            arm for arm, count in arm_counts.items() if count > 1
        ),
        "attestation_sha256_by_arm": dict(
            sorted(attestation_sha256_by_arm.items())
        ),
        "execution_attestation_fingerprint": (
            _fingerprint(attestation_sha256_by_arm)
            if attestation_sha256_by_arm
            else None
        ),
    }


def audit_pilot_execution(
    readiness: dict[str, Any],
    wandb_snapshot: dict[str, Any],
    *,
    profile: PilotExecutionProfile,
    local_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one pilot without importing full logs or metric tables."""

    readiness = _mapping(readiness, "readiness")
    wandb_snapshot = _mapping(wandb_snapshot, "W&B snapshot")
    source = _mapping(wandb_snapshot.get("source"), "W&B snapshot source")
    local = _local_state(local_summary, profile.expected_arms)
    pilot_runs = _pilot_wandb_runs(wandb_snapshot, profile)
    all_runs = [
        run for run in wandb_snapshot["runs"] if isinstance(run, dict)
    ]
    state_counts = Counter(str(run.get("state") or "unknown") for run in all_runs)

    expected = set(profile.expected_arms)
    local_completed = set(local["completed_arms"])
    local_attested = set(local["attested_arms"])
    local_integrity = (
        not local["unexpected_arms"] and not local["duplicate_arms"]
    )
    if local["summary_present"]:
        if local["status"] == "failed":
            state = "failed"
        elif local["status"] == "running":
            state = "in_progress"
        elif local["status"] == "completed":
            state = (
                "completed_attested"
                if local_completed == expected
                and local_attested == expected
                and local_integrity
                else "completed_unattested"
            )
        else:
            state = "local_state_unknown"
    elif pilot_runs:
        state = "external_activity_unattested"
    else:
        state = "not_started_in_observed_state"

    training_attested = state == "completed_attested"
    readiness_pass = readiness.get("overall_status") == "pass"
    result = {
        "schema_version": 1,
        "claim_scope": profile.claim_scope,
        "pilot": profile.name,
        "state": state,
        "observation_scope": (
            "The captured local summary and W&B inventory only; absence does "
            "not prove that no run exists outside those observations."
        ),
        "submission_readiness": {
            "status": str(readiness.get("overall_status") or "unknown"),
            "checks": dict(readiness.get("counts") or {}),
            "fingerprint": readiness.get("fingerprint"),
        },
        "local": local,
        "wandb": {
            "observed_at": wandb_snapshot.get("observed_at"),
            "entity": source.get("entity"),
            "project": source.get("project"),
            "observed_runs": len(all_runs),
            "native_pilot_runs": len(pilot_runs),
            "legacy_or_other_runs": len(all_runs) - len(pilot_runs),
            "states": dict(sorted(state_counts.items())),
        },
        "pilot_submission_authorized": bool(
            readiness_pass
            and readiness.get("pilot_submission_authorized") is True
        ),
        "training_execution_attested": training_attested,
        "quality_claim_authorized": False,
        "promotion_claim_authorized": False,
        "next_action": (
            "Inspect the sealed pilot evidence and screening metrics."
            if training_attested
            else profile.next_action
        ),
    }
    result["fingerprint"] = _fingerprint(result)
    return result


def audit_lfm_pilot_execution(
    readiness: dict[str, Any],
    wandb_snapshot: dict[str, Any],
    *,
    local_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve the original LFM execution-audit API."""

    return audit_pilot_execution(
        readiness,
        wandb_snapshot,
        profile=LFM_PROFILE,
        local_summary=local_summary,
    )


def audit_smol_pilot_execution(
    readiness: dict[str, Any],
    wandb_snapshot: dict[str, Any],
    *,
    local_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the observed Smol vision-transfer pilot execution."""

    return audit_pilot_execution(
        readiness,
        wandb_snapshot,
        profile=SMOL_PROFILE,
        local_summary=local_summary,
    )
