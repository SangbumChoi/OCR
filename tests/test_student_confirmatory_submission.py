from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from docvlm_eval.student.confirmatory_submission import (
    audit_smol_confirmatory_submission,
)
from docvlm_eval.student.pilot_execution import (
    audit_smol_pilot_execution,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
ARMS = ("lfm_language_only", "lfm_smol_dual")


def _plans(tmp_path: Path):
    pilot = compile_sweep_plan(
        ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "pilot",
    )
    confirmatory = compile_sweep_plan(
        ROOT / "configs" / "sub1b_smol_vision_transfer_sweep.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "confirmatory",
    )
    return pilot, confirmatory


def _attestation(seed: str):
    return {
        "attestation_sha256": "sha256:" + (seed * 64),
        "claim_scope": "deployment_capability",
        "contract_status": "pass",
        "quality_claim_authorized": True,
        "stage_count": 25,
    }


def _execution():
    readiness = json.loads(
        (
            RESULTS / "smol_vision_transfer_pilot_readiness.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (RESULTS / "lfm_ablation_wandb_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "status": "completed",
        "variants": [
            {
                "variant": arm,
                "status": "completed",
                "execution_attestation": _attestation(
                    "a" if arm == ARMS[0] else "b"
                ),
            }
            for arm in ARMS
        ],
    }
    return readiness, audit_smol_pilot_execution(
        readiness,
        snapshot,
        local_summary=summary,
    )


def _comparison(pilot, confirmatory):
    required_gates = confirmatory.promotion.required_gates
    return {
        "schema_version": 6,
        "sweep": pilot.name,
        "sweep_fingerprint": pilot.fingerprint,
        "baseline": ARMS[0],
        "replicates": ["seed_0"],
        "execution_attestations": {
            f"{arm}--seed_0": _attestation(
                "a" if arm == ARMS[0] else "b"
            )
            for arm in ARMS
        },
        "variants": {
            ARMS[0]: {
                "metrics": {"heldout_score": 0.40},
            },
            ARMS[1]: {
                "metrics": {"heldout_score": 0.43},
                "heldout_score_conclusion": "improved",
                "gate_status": "pass",
                "gate_statuses": {
                    gate: "pass" for gate in required_gates
                },
            },
        },
    }


def test_positive_sealed_pilot_authorizes_confirmatory_budget(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness, execution = _execution()

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=_comparison(pilot, confirmatory),
    )

    assert result["overall_status"] == "pass"
    assert result["counts"] == {"pass": 7, "pending": 0, "fail": 0}
    assert result["confirmatory_submission_authorized"] is True
    assert result["quality_claim_authorized"] is False


def test_missing_pilot_execution_remains_pending(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness = json.loads(
        (
            RESULTS / "smol_vision_transfer_pilot_readiness.json"
        ).read_text(encoding="utf-8")
    )
    execution = json.loads(
        (
            RESULTS / "smol_vision_transfer_pilot_execution_state.json"
        ).read_text(encoding="utf-8")
    )

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=None,
    )

    assert result["overall_status"] == "pending"
    assert result["counts"] == {"pass": 2, "pending": 5, "fail": 0}
    assert result["confirmatory_submission_authorized"] is False


def test_nonpositive_pilot_effect_rejects_confirmatory_budget(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness, execution = _execution()
    comparison = _comparison(pilot, confirmatory)
    comparison["variants"][ARMS[1]]["metrics"]["heldout_score"] = 0.39
    comparison["variants"][ARMS[1]][
        "heldout_score_conclusion"
    ] = "regressed"

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=comparison,
    )

    assert result["overall_status"] == "fail"
    assert result["confirmatory_submission_authorized"] is False
    effect = next(
        check
        for check in result["checks"]
        if check["id"] == "positive_screening_effect"
    )
    assert effect["status"] == "fail"


def test_attestation_mismatch_rejects_confirmatory_budget(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness, execution = _execution()
    comparison = _comparison(pilot, confirmatory)
    comparison["execution_attestations"][
        f"{ARMS[1]}--seed_0"
    ] = _attestation("c")

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=comparison,
    )

    assert result["confirmatory_submission_authorized"] is False
    linkage = next(
        check
        for check in result["checks"]
        if check["id"] == "pilot_attestation_linkage"
    )
    assert linkage["status"] == "fail"


def test_tampered_execution_artifact_rejects_confirmatory_budget(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness, execution = _execution()
    execution["next_action"] = "tampered"

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=_comparison(pilot, confirmatory),
    )

    assert result["confirmatory_submission_authorized"] is False
    execution_check = next(
        check
        for check in result["checks"]
        if check["id"] == "sealed_pilot_execution"
    )
    assert execution_check["status"] == "pending"
    assert execution_check["evidence"]["fingerprint_valid"] is False


def test_failed_required_gate_rejects_confirmatory_budget(tmp_path):
    pilot, confirmatory = _plans(tmp_path)
    readiness, execution = _execution()
    comparison = copy.deepcopy(_comparison(pilot, confirmatory))
    comparison["variants"][ARMS[1]]["gate_statuses"][
        "generation_stability"
    ] = "fail"

    result = audit_smol_confirmatory_submission(
        pilot,
        confirmatory,
        pilot_readiness=readiness,
        pilot_execution=execution,
        pilot_comparison=comparison,
    )

    assert result["confirmatory_submission_authorized"] is False
    gates = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_required_gates"
    )
    assert gates["evidence"]["failed_or_missing_gates"] == [
        "generation_stability"
    ]
