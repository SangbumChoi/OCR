from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from docvlm_eval.student.pilot_handoff import (
    build_smol_pilot_handoff,
    restore_smol_pilot_handoff,
    verify_smol_pilot_handoff,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "lfm_language_only--seed_0",
    "lfm_smol_dual--seed_0",
)


def _plan(tmp_path: Path):
    return compile_sweep_plan(
        ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def _attestation(character: str):
    return {
        "attestation_sha256": "sha256:" + (character * 64),
        "claim_scope": "deployment_capability",
        "contract_status": "pass",
        "quality_claim_authorized": True,
        "stage_count": 25,
    }


def _sources(tmp_path: Path, plan):
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    attestations = {
        run_id: _attestation("a" if index == 0 else "b")
        for index, run_id in enumerate(RUNS)
    }
    summary = {
        "status": "completed",
        "variants": [
            {
                "run": run_id,
                "variant": run_id.split("--", 1)[0],
                "replicate": "seed_0",
                "status": "completed",
                "execution_attestation": attestation,
            }
            for run_id, attestation in attestations.items()
        ],
        "comparison": str(sweep_root / "comparison.json"),
    }
    comparison = {
        "schema_version": 6,
        "sweep": plan.name,
        "sweep_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "replicates": ["seed_0"],
        "execution_attestations": attestations,
        "variants": {
            "lfm_language_only": {
                "metrics": {"heldout_score": 0.4},
            },
            "lfm_smol_dual": {
                "metrics": {"heldout_score": 0.45},
            },
        },
    }
    (sweep_root / "sweep_run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (sweep_root / "comparison.json").write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    return sweep_root


def test_smol_pilot_handoff_build_verify_restore(tmp_path):
    plan = _plan(tmp_path)
    sweep_root = _sources(tmp_path, plan)
    result = build_smol_pilot_handoff(
        plan,
        sweep_root=sweep_root,
        output_root=tmp_path / "handoffs",
    )

    assert result["reused"] is False
    assert result["expected_runs"] == list(RUNS)
    assert result["quality_claim_authorized"] is False
    verification = verify_smol_pilot_handoff(
        result["root"],
        plan=plan,
    )
    assert verification["manifest"]["fingerprint"] == result["fingerprint"]

    restored_root = tmp_path / "restored"
    restored = restore_smol_pilot_handoff(
        result["root"],
        plan=plan,
        sweep_root=restored_root,
    )
    assert restored["restored"] == [
        "sweep_run_summary.json",
        "comparison.json",
    ]
    assert restored["reused"] == []

    reused = restore_smol_pilot_handoff(
        result["root"],
        plan=plan,
        sweep_root=restored_root,
    )
    assert reused["restored"] == []
    assert reused["reused"] == [
        "sweep_run_summary.json",
        "comparison.json",
    ]


def test_smol_pilot_handoff_detects_file_tampering(tmp_path):
    plan = _plan(tmp_path)
    result = build_smol_pilot_handoff(
        plan,
        sweep_root=_sources(tmp_path, plan),
        output_root=tmp_path / "handoffs",
    )
    comparison = Path(result["root"]) / "comparison.json"
    comparison.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file integrity failed"):
        verify_smol_pilot_handoff(result["root"], plan=plan)


def test_smol_pilot_handoff_rejects_attestation_mismatch(tmp_path):
    plan = _plan(tmp_path)
    sweep_root = _sources(tmp_path, plan)
    summary_path = sweep_root / "sweep_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["variants"][0]["execution_attestation"] = _attestation("c")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="attestations do not match"):
        build_smol_pilot_handoff(
            plan,
            sweep_root=sweep_root,
            output_root=tmp_path / "handoffs",
        )


def test_smol_pilot_handoff_refuses_different_local_evidence(tmp_path):
    plan = _plan(tmp_path)
    result = build_smol_pilot_handoff(
        plan,
        sweep_root=_sources(tmp_path, plan),
        output_root=tmp_path / "handoffs",
    )
    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    (restored_root / "comparison.json").write_text(
        json.dumps({"different": True}),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="different local evidence"):
        restore_smol_pilot_handoff(
            result["root"],
            plan=plan,
            sweep_root=restored_root,
        )


def test_smol_pilot_handoff_rejects_unsealed_run(tmp_path):
    plan = _plan(tmp_path)
    sweep_root = _sources(tmp_path, plan)
    summary_path = sweep_root / "sweep_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    broken = copy.deepcopy(summary)
    broken["variants"][0]["execution_attestation"][
        "contract_status"
    ] = "fail"
    summary_path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="not sealed"):
        build_smol_pilot_handoff(
            plan,
            sweep_root=sweep_root,
            output_root=tmp_path / "handoffs",
        )
