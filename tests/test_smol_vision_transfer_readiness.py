from __future__ import annotations

import json
import sys
from pathlib import Path

from docvlm_eval.student.sweep import compile_sweep_plan
from docvlm_eval.student.transfer_readiness import (
    audit_smol_vision_transfer_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml"
VISION = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_smol_vision_real_source_preflight.json"
)
LANGUAGE = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_lfm_real_source_preflight.json"
)
SELECTION = (
    ROOT / "docs" / "results" / "selective_transfer_source_matrix.json"
)


def _plan(tmp_path: Path):
    return compile_sweep_plan(
        SWEEP,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def _audit(tmp_path: Path, *, vision: Path = VISION):
    return audit_smol_vision_transfer_pilot(
        _plan(tmp_path),
        repo_root=ROOT,
        sweep_path=SWEEP,
        vision_preflight_path=vision,
        language_preflight_path=LANGUAGE,
        source_selection_path=SELECTION,
    )


def test_smol_vision_transfer_pilot_readiness_passes(tmp_path):
    plan = _plan(tmp_path)
    result = audit_smol_vision_transfer_pilot(
        plan,
        repo_root=ROOT,
        sweep_path=SWEEP,
        vision_preflight_path=VISION,
        language_preflight_path=LANGUAGE,
        source_selection_path=SELECTION,
    )

    assert result["overall_status"] == "pass"
    assert result["pilot_submission_authorized"] is True
    assert result["quality_claim_authorized"] is False
    assert result["target_cuda_feasibility_claim_authorized"] is False
    assert result["counts"] == {"pass": 14, "fail": 0}
    assert result["sweep"]["plan_fingerprint"] == plan.fingerprint
    budget = next(
        check
        for check in result["checks"]
        if check["id"] == "bounded_screening_budget"
    )
    assert budget["evidence"]["public_sampling_strategy"] == "task_stratified"
    assert budget["evidence"]["public_min_rows_per_task"] == 16


def test_smol_vision_readiness_rejects_tampered_scope(tmp_path):
    payload = json.loads(VISION.read_text(encoding="utf-8"))
    payload["transfer"]["vision_scope"] = "all"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    result = _audit(tmp_path, vision=tampered)

    assert result["overall_status"] == "fail"
    check = next(
        item
        for item in result["checks"]
        if item["id"] == "executed_vision_transfer_integrity"
    )
    assert check["status"] == "fail"


def test_smol_vision_readiness_rejects_unstratified_public_cap(tmp_path):
    plan = _plan(tmp_path)
    dual = next(
        variant for variant in plan.variants if variant.arm_id == "lfm_smol_dual"
    )
    public = dual.plan.raw_spec["data"]["components"][1]["hub"]
    public["sampling_strategy"] = "global_hash"
    public["min_rows_per_task"] = 0

    result = audit_smol_vision_transfer_pilot(
        plan,
        repo_root=ROOT,
        sweep_path=SWEEP,
        vision_preflight_path=VISION,
        language_preflight_path=LANGUAGE,
        source_selection_path=SELECTION,
    )

    budget = next(
        check
        for check in result["checks"]
        if check["id"] == "bounded_screening_budget"
    )
    assert result["pilot_submission_authorized"] is False
    assert budget["status"] == "fail"
