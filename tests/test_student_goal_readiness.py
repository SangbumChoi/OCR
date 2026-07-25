from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from docvlm_eval.student.goal_readiness import (
    audit_end_to_end_goal_readiness,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def _plan(tmp_path: Path):
    return compile_sweep_plan(
        ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def _audit(
    tmp_path: Path,
    execution_state: Path,
    *,
    quality_evidence: Path | None = None,
    public_dataset_readiness: Path | None = None,
    pilot_readiness: Path | None = None,
):
    return audit_end_to_end_goal_readiness(
        _plan(tmp_path),
        repo_root=ROOT,
        method_catalog_path=(
            ROOT / "configs" / "frontier_method_catalog.jsonl"
        ),
        method_evidence_path=(
            ROOT / "configs" / "frontier_method_evidence.yaml"
        ),
        synth_config_path=ROOT / "configs" / "synth_data.yaml",
        public_dataset_readiness_path=(
            public_dataset_readiness
            or RESULTS / "public_udd_training_readiness.json"
        ),
        vision_preflight_path=(
            RESULTS
            / "selective_transfer_smol_vision_real_source_preflight.json"
        ),
        language_preflight_path=(
            RESULTS / "selective_transfer_lfm_real_source_preflight.json"
        ),
        pilot_readiness_path=(
            pilot_readiness
            or RESULTS / "smol_vision_transfer_pilot_readiness.json"
        ),
        execution_state_path=execution_state,
        quality_evidence_path=quality_evidence,
    )


def test_current_goal_is_implementation_ready_but_not_complete(tmp_path):
    result = _audit(
        tmp_path,
        RESULTS / "smol_vision_transfer_pilot_execution_state.json",
    )

    assert result["overall_status"] == (
        "implementation_ready_execution_pending"
    )
    assert result["implementation_ready"] is True
    assert result["execution_complete"] is False
    assert result["quality_proven"] is False
    assert result["goal_complete"] is False
    assert result["counts"] == {"pass": 9, "pending": 3, "fail": 0}
    assert result["next_required_evidence"] == [
        "target_gpu_execution",
        "heldout_quality_evidence",
        "multi_seed_promotion_evidence",
    ]


def test_goal_audit_rejects_quality_claim_without_execution(tmp_path):
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "claim_scope": "heldout_quality_evidence",
                "execution_attestation_fingerprint": (
                    "sha256:" + ("a" * 64)
                ),
                "gate_status": "pass",
                "quality_claim_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    result = _audit(
        tmp_path,
        RESULTS / "smol_vision_transfer_pilot_execution_state.json",
        quality_evidence=quality,
    )

    quality = next(
        check
        for check in result["checks"]
        if check["id"] == "heldout_quality_evidence"
    )
    assert quality["status"] == "fail"
    assert result["overall_status"] == "not_ready"
    assert result["goal_complete"] is False


def test_goal_audit_rejects_tampered_public_dataset_evidence(tmp_path):
    source = json.loads(
        (RESULTS / "public_udd_training_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    source["dataset"]["task_counts"].pop("kie")
    tampered = tmp_path / "public-dataset.json"
    tampered.write_text(json.dumps(source), encoding="utf-8")

    result = _audit(
        tmp_path,
        RESULTS / "smol_vision_transfer_pilot_execution_state.json",
        public_dataset_readiness=tampered,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "uploaded_multitask_dataset"
    )
    assert check["status"] == "fail"
    assert "fingerprint mismatch" in check["evidence"]["errors"]
    assert "required task coverage is incomplete" in check["evidence"]["errors"]
    assert result["implementation_ready"] is False


def _seal(payload: dict) -> dict:
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["fingerprint"] = (
        f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    )
    return payload


def test_goal_audit_rejects_stale_compiled_pilot_readiness(tmp_path):
    readiness = json.loads(
        (
            RESULTS / "smol_vision_transfer_pilot_readiness.json"
        ).read_text(encoding="utf-8")
    )
    readiness["sweep"]["plan_fingerprint"] = "sha256:" + ("b" * 64)
    stale = tmp_path / "stale-readiness.json"
    stale.write_text(json.dumps(_seal(readiness)), encoding="utf-8")

    result = _audit(
        tmp_path,
        RESULTS / "smol_vision_transfer_pilot_execution_state.json",
        pilot_readiness=stale,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "pilot_submission_contract"
    )
    assert check["status"] == "fail"
    assert check["evidence"]["fingerprint_valid"] is True
    assert result["implementation_ready"] is False


def test_goal_audit_rejects_tampered_execution_state(tmp_path):
    execution = json.loads(
        (
            RESULTS / "smol_vision_transfer_pilot_execution_state.json"
        ).read_text(encoding="utf-8")
    )
    execution["state"] = "completed_attested"
    execution["training_execution_attested"] = True
    tampered = tmp_path / "execution.json"
    tampered.write_text(json.dumps(execution), encoding="utf-8")

    result = _audit(tmp_path, tampered)

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "target_gpu_execution"
    )
    assert check["status"] == "fail"
    assert check["evidence"]["fingerprint_valid"] is False
    assert result["execution_complete"] is False
