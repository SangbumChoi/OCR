from __future__ import annotations

import json
import sys
from pathlib import Path

from docvlm_eval.student.sweep import compile_sweep_plan
from docvlm_eval.student.transfer_readiness import audit_lfm_transfer_pilot


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "configs" / "sub1b_lfm_language_transfer_pilot.yaml"
PREFLIGHT = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_lfm_real_source_preflight.json"
)
SOURCE_SELECTION = (
    ROOT / "docs" / "results" / "selective_transfer_source_matrix.json"
)


def _plan(tmp_path: Path):
    return compile_sweep_plan(
        SWEEP,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def test_lfm_transfer_pilot_readiness_passes_current_contract(tmp_path):
    plan = _plan(tmp_path)
    result = audit_lfm_transfer_pilot(
        plan,
        repo_root=ROOT,
        sweep_path=SWEEP,
        preflight_path=PREFLIGHT,
        source_selection_path=SOURCE_SELECTION,
    )

    assert result["overall_status"] == "pass"
    assert result["pilot_submission_authorized"] is True
    assert result["quality_claim_authorized"] is False
    assert result["target_cuda_feasibility_claim_authorized"] is False
    assert result["counts"] == {"pass": 15, "fail": 0}
    assert result["sweep"]["plan_fingerprint"] == plan.fingerprint
    budget = next(
        check
        for check in result["checks"]
        if check["id"] == "bounded_screening_budget"
    )
    assert budget["evidence"]["public_sampling_strategy"] == "task_stratified"
    assert budget["evidence"]["public_min_rows_per_task"] == 16


def test_lfm_transfer_pilot_readiness_rejects_tampered_payload_evidence(
    tmp_path,
):
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    preflight["transfer"]["value_verified"] = False
    tampered = tmp_path / "tampered-preflight.json"
    tampered.write_text(json.dumps(preflight), encoding="utf-8")

    result = audit_lfm_transfer_pilot(
        _plan(tmp_path),
        repo_root=ROOT,
        sweep_path=SWEEP,
        preflight_path=tampered,
        source_selection_path=SOURCE_SELECTION,
    )

    assert result["overall_status"] == "fail"
    assert result["pilot_submission_authorized"] is False
    transfer_check = next(
        check
        for check in result["checks"]
        if check["id"] == "executed_transfer_integrity"
    )
    assert transfer_check["status"] == "fail"


def test_lfm_transfer_pilot_readiness_rejects_tampered_source_matrix(
    tmp_path,
):
    source_selection = json.loads(
        SOURCE_SELECTION.read_text(encoding="utf-8")
    )
    source_selection["quality_claim_authorized"] = True
    tampered = tmp_path / "tampered-source-selection.json"
    tampered.write_text(json.dumps(source_selection), encoding="utf-8")

    result = audit_lfm_transfer_pilot(
        _plan(tmp_path),
        repo_root=ROOT,
        sweep_path=SWEEP,
        preflight_path=PREFLIGHT,
        source_selection_path=tampered,
    )

    assert result["overall_status"] == "fail"
    source_check = next(
        check
        for check in result["checks"]
        if check["id"] == "cross_model_source_selection"
    )
    assert source_check["status"] == "fail"
