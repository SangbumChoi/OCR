from __future__ import annotations

import copy
import json
from pathlib import Path

from docvlm_eval.student.pilot_execution import (
    EXPECTED_ARMS,
    audit_lfm_pilot_execution,
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


def test_current_observation_has_no_native_pilot_execution():
    result = audit_lfm_pilot_execution(
        _read("lfm_selective_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
    )

    assert result["state"] == "not_started_in_observed_state"
    assert result["wandb"]["observed_runs"] == 10
    assert result["wandb"]["native_pilot_runs"] == 0
    assert result["training_execution_attested"] is False
    assert result["quality_claim_authorized"] is False


def test_external_pilot_run_is_activity_not_execution_attestation():
    snapshot = _read("lfm_ablation_wandb_snapshot.json")
    snapshot["runs"].append(
        {
            "name": "docvlm-lfm-language-transfer-pilot--pretrain--"
            "lfm_random--seed_0",
            "state": "running",
        }
    )

    result = audit_lfm_pilot_execution(
        _read("lfm_selective_transfer_pilot_readiness.json"),
        snapshot,
    )

    assert result["state"] == "external_activity_unattested"
    assert result["wandb"]["native_pilot_runs"] == 1
    assert result["training_execution_attested"] is False


def test_completed_local_summary_requires_every_sealed_arm():
    summary = {
        "status": "completed",
        "variants": [
            {
                "variant": arm,
                "status": "completed",
                "execution_attestation": _attestation(),
            }
            for arm in EXPECTED_ARMS
        ],
    }
    result = audit_lfm_pilot_execution(
        _read("lfm_selective_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary=summary,
    )
    assert result["state"] == "completed_attested"
    assert result["training_execution_attested"] is True
    assert result["quality_claim_authorized"] is False
    assert result["local"]["execution_attestation_fingerprint"].startswith(
        "sha256:"
    )
    assert result["fingerprint"].startswith("sha256:")

    incomplete = copy.deepcopy(summary)
    incomplete["variants"][0].pop("execution_attestation")
    result = audit_lfm_pilot_execution(
        _read("lfm_selective_transfer_pilot_readiness.json"),
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary=incomplete,
    )
    assert result["state"] == "completed_unattested"
    assert result["training_execution_attested"] is False


def test_completed_local_summary_accepts_deployment_capability_seals():
    readiness = _read("lfm_selective_transfer_pilot_readiness.json")
    summary = {
        "status": "completed",
        "variants": [
            {
                "variant": arm,
                "status": "completed",
                "execution_attestation": {
                    **_attestation(),
                    "claim_scope": "deployment_capability",
                    "quality_claim_authorized": True,
                },
            }
            for arm in EXPECTED_ARMS
        ],
    }

    result = audit_lfm_pilot_execution(
        readiness,
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary=summary,
    )

    assert result["state"] == "completed_attested"
    assert result["training_execution_attested"] is True


def test_completed_local_summary_rejects_duplicate_or_unexpected_arms():
    readiness = _read("lfm_selective_transfer_pilot_readiness.json")
    variants = [
        {
            "variant": arm,
            "status": "completed",
            "execution_attestation": _attestation(),
        }
        for arm in EXPECTED_ARMS
    ]
    variants.extend(
        [
            copy.deepcopy(variants[0]),
            {
                "variant": "unexpected",
                "status": "completed",
                "execution_attestation": _attestation(),
            },
        ]
    )

    result = audit_lfm_pilot_execution(
        readiness,
        _read("lfm_ablation_wandb_snapshot.json"),
        local_summary={"status": "completed", "variants": variants},
    )

    assert result["state"] == "completed_unattested"
    assert result["training_execution_attested"] is False
    assert result["local"]["duplicate_arms"]
    assert result["local"]["unexpected_arms"] == ["unexpected"]


def test_failed_local_summary_wins_over_external_activity():
    snapshot = _read("lfm_ablation_wandb_snapshot.json")
    snapshot["runs"].append(
        {
            "name": "docvlm-lfm-language-transfer-pilot--sft--"
            "lfm_strict_transfer--seed_0",
            "state": "crashed",
        }
    )
    result = audit_lfm_pilot_execution(
        _read("lfm_selective_transfer_pilot_readiness.json"),
        snapshot,
        local_summary={"status": "failed", "variants": []},
    )
    assert result["state"] == "failed"
    assert result["training_execution_attested"] is False
