"""Compact execution-state evidence for the native LFM transfer pilot."""

from __future__ import annotations

from collections import Counter
from typing import Any


PILOT_NAME = "docvlm-lfm-language-transfer-pilot"
EXPECTED_ARMS = frozenset(
    {"native_random", "lfm_random", "lfm_strict_transfer"}
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _pilot_wandb_runs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = f"{PILOT_NAME}--"
    runs = snapshot.get("runs")
    if not isinstance(runs, list):
        raise ValueError("W&B snapshot runs must be a list")
    return [
        run
        for run in runs
        if isinstance(run, dict)
        and str(run.get("name") or "").startswith(prefix)
    ]


def _attestation_is_sealed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    digest = value.get("attestation_sha256")
    return (
        value.get("contract_status") == "pass"
        and value.get("claim_scope") == "execution_contract_only"
        and value.get("quality_claim_authorized") is False
        and isinstance(digest, str)
        and digest.startswith("sha256:")
        and isinstance(value.get("stage_count"), int)
        and value["stage_count"] > 0
    )


def _local_state(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {
            "summary_present": False,
            "status": "absent",
            "observed_arms": [],
            "completed_arms": [],
            "attested_arms": [],
        }

    variants = summary.get("variants")
    if not isinstance(variants, list):
        raise ValueError("local sweep summary variants must be a list")
    observed: set[str] = set()
    completed: set[str] = set()
    attested: set[str] = set()
    for item in variants:
        if not isinstance(item, dict):
            continue
        arm = str(item.get("variant") or "")
        if arm not in EXPECTED_ARMS:
            continue
        observed.add(arm)
        if item.get("status") == "completed":
            completed.add(arm)
            if _attestation_is_sealed(item.get("execution_attestation")):
                attested.add(arm)
    return {
        "summary_present": True,
        "status": str(summary.get("status") or "unknown"),
        "observed_arms": sorted(observed),
        "completed_arms": sorted(completed),
        "attested_arms": sorted(attested),
    }


def audit_lfm_pilot_execution(
    readiness: dict[str, Any],
    wandb_snapshot: dict[str, Any],
    *,
    local_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize observed execution without importing full logs or metric tables."""

    readiness = _mapping(readiness, "readiness")
    wandb_snapshot = _mapping(wandb_snapshot, "W&B snapshot")
    source = _mapping(wandb_snapshot.get("source"), "W&B snapshot source")
    local = _local_state(local_summary)
    pilot_runs = _pilot_wandb_runs(wandb_snapshot)
    all_runs = [
        run for run in wandb_snapshot["runs"] if isinstance(run, dict)
    ]
    state_counts = Counter(str(run.get("state") or "unknown") for run in all_runs)

    expected = set(EXPECTED_ARMS)
    local_completed = set(local["completed_arms"])
    local_attested = set(local["attested_arms"])
    if local["summary_present"]:
        if local["status"] == "failed":
            state = "failed"
        elif local["status"] == "running":
            state = "in_progress"
        elif local["status"] == "completed":
            state = (
                "completed_attested"
                if local_completed == expected and local_attested == expected
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
    return {
        "schema_version": 1,
        "claim_scope": "native_lfm_pilot_execution_state_only",
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
            else "Run notebooks/lfm_selective_transfer_pilot.ipynb on "
            "native-BF16 hardware."
        ),
    }
