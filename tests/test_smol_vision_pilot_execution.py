from __future__ import annotations

import json
from pathlib import Path

from docvlm_eval.student.pilot_execution import (
    SMOL_PROFILE,
    audit_smol_pilot_execution,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def _read(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _attestation():
    return {
        "attestation_sha256": "sha256:" + ("a" * 64),
        "claim_scope": "execution_contract_only",
        "contract_status": "pass",
        "quality_claim_authorized": False,
        "stage_count": 25,
    }


def test_current_snapshot_has_no_smol_pilot_execution():
    result = audit_smol_pilot_execution(
        _read("smol_vision_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
    )

    assert result["state"] == "not_started_in_observed_state"
    assert result["wandb"]["observed_runs"] == 10
    assert result["wandb"]["native_pilot_runs"] == 0
    assert result["training_execution_attested"] is False
    assert result["quality_claim_authorized"] is False


def test_smol_pilot_requires_both_sealed_local_arms():
    summary = {
        "status": "completed",
        "variants": [
            {
                "variant": arm,
                "status": "completed",
                "execution_attestation": _attestation(),
            }
            for arm in SMOL_PROFILE.expected_arms
        ],
    }
    result = audit_smol_pilot_execution(
        _read("smol_vision_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary=summary,
    )

    assert result["state"] == "completed_attested"
    assert result["training_execution_attested"] is True
    assert result["quality_claim_authorized"] is False

    summary["variants"][0].pop("execution_attestation")
    result = audit_smol_pilot_execution(
        _read("smol_vision_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary=summary,
    )
    assert result["state"] == "completed_unattested"
    assert result["training_execution_attested"] is False


def test_smol_external_activity_is_not_execution_attestation():
    snapshot = _read("lfm_ablation_wandb_snapshot.json")
    snapshot["runs"].append(
        {
            "name": (
                "docvlm-smol-vision-transfer-pilot--pretrain--"
                "lfm_smol_dual--seed_0"
            ),
            "state": "running",
        }
    )
    result = audit_smol_pilot_execution(
        _read("smol_vision_transfer_pilot_readiness.json"),
        snapshot,
    )

    assert result["state"] == "external_activity_unattested"
    assert result["wandb"]["native_pilot_runs"] == 1
    assert result["training_execution_attested"] is False
