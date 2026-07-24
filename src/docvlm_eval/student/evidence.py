"""Deterministic evidence attestation for native student experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .experiment import (
    ExperimentPlan,
    ExperimentRunner,
    ExperimentStage,
    _atomic_write_json,
    _fingerprint,
)


_TRAINING_STAGES = ("pretrain", "sft", "preference", "rlvr")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if isinstance(value, dict):
            last = value
    return last


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path, root: Path, *, hash_mode: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path, root),
        "bytes": path.stat().st_size,
    }
    if hash_mode == "full" or path.stat().st_size <= 1024 * 1024:
        record["sha256"] = _sha256_file(path)
    else:
        record["sha256"] = None
    return record


def _files_for_path(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    return []


def _checkpoint_path(root: Path, stage_name: str) -> Path | None:
    pointer = root / "artifacts" / stage_name / "latest_checkpoint.txt"
    if not pointer.is_file():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _unique_files(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()})


def _stage_files(stage: ExperimentStage, root: Path, state: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for artifact in stage.artifacts:
        files.extend(_files_for_path(Path(artifact.path)))
    log = state.get("log")
    if log:
        files.extend(_files_for_path(Path(str(log))))
    checkpoint = _checkpoint_path(root, stage.name)
    if checkpoint is not None:
        files.extend(_files_for_path(checkpoint))
    return _unique_files(files)


def _check(check_id: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    }


def _training_proof(root: Path, stage_name: str) -> dict[str, Any] | None:
    checkpoint = _checkpoint_path(root, stage_name)
    if checkpoint is None:
        return None
    trainer_state = checkpoint / "trainer_state.json"
    student_metadata = checkpoint / "student" / "metadata.json"
    return {
        "checkpoint": _display_path(checkpoint, root),
        "trainer_state": _read_json(trainer_state) if trainer_state.is_file() else None,
        "student_metadata": (
            _read_json(student_metadata) if student_metadata.is_file() else None
        ),
        "last_metric": _last_jsonl(
            root / "artifacts" / stage_name / "metrics.jsonl"
        ),
    }


def _semantic_evidence(
    plan: ExperimentPlan,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"training": {}}

    initialization_path = root / "artifacts" / "initial" / "metadata.json"
    initialization = (
        _read_json(initialization_path) if initialization_path.is_file() else None
    )
    evidence["initialization"] = initialization
    parameter_total = int(
        ((initialization or {}).get("parameter_counts") or {}).get("total", 0)
    )
    checks.append(
        _check(
            "deployment_parameter_budget",
            0 < parameter_total < 1_000_000_000,
            {
                "actual_parameters": parameter_total,
                "max_parameters_exclusive": 1_000_000_000,
            },
        )
    )

    stage_names = set(plan.stage_names)
    for stage_name in _TRAINING_STAGES:
        if stage_name not in stage_names:
            continue
        proof = _training_proof(root, stage_name)
        evidence["training"][stage_name] = proof
        state = (proof or {}).get("trainer_state") or {}
        if stage_name in {"pretrain", "sft"}:
            progress = int(state.get("global_step", 0))
            progress_key = "global_step"
        elif stage_name == "preference":
            progress = min(
                int(state.get("preference_step", 0)),
                int(state.get("optimizer_step", 0)),
            )
            progress_key = "preference_and_optimizer_step"
        else:
            progress = min(
                int(state.get("rollout_step", 0)),
                int(state.get("optimizer_step", 0)),
            )
            progress_key = "rollout_and_optimizer_step"
        checks.append(
            _check(
                f"{stage_name}_optimization_progress",
                progress > 0,
                {progress_key: progress, "trainer_state": state},
            )
        )

    evaluation_root = root / "artifacts" / "evaluation"
    comparison_path = evaluation_root / "comparison.json"
    gates_path = evaluation_root / "gates.json"
    manifest_path = evaluation_root / "manifest.json"
    comparison = _read_json(comparison_path) if comparison_path.is_file() else None
    gates = _read_json(gates_path) if gates_path.is_file() else None
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    evidence["evaluation"] = {
        "manifest": manifest,
        "comparison": comparison,
        "gates": gates,
    }
    splits = (comparison or {}).get("splits") or {}
    split_counts = {
        split: int((splits.get(split) or {}).get("n_samples", 0))
        for split in ("train", "heldout")
    }
    checks.append(
        _check(
            "train_and_heldout_generation",
            all(count > 0 for count in split_counts.values()),
            {"n_samples": split_counts},
        )
    )
    expected_final_stage = next(
        (
            name
            for name in ("rlvr", "preference", "sft")
            if name in stage_names
        ),
        None,
    )
    observed_stage = (
        ((manifest or {}).get("checkpoint_metadata") or {}).get("run_stage")
    )
    observed_base_stage = (
        str(observed_stage).split(":", 1)[0] if observed_stage is not None else None
    )
    checks.append(
        _check(
            "evaluation_uses_final_checkpoint",
            expected_final_stage is not None
            and observed_base_stage == expected_final_stage,
            {
                "expected_stage": expected_final_stage,
                "observed_stage": observed_stage,
            },
        )
    )
    return evidence, checks


def build_experiment_attestation(
    plan: ExperimentPlan,
    *,
    repo_root: str | Path,
    output: str | Path | None = None,
    hash_mode: str = "full",
) -> dict[str, Any]:
    """Build a deterministic, independently re-verifiable experiment attestation."""
    if hash_mode not in {"full", "metadata"}:
        raise ValueError("hash_mode must be 'full' or 'metadata'")
    root = Path(plan.root).resolve()
    runner = ExperimentRunner(plan, repo_root=repo_root)
    signatures = runner.signatures()
    stage_records = []
    contract_checks: list[dict[str, Any]] = []
    for stage in plan.stages:
        state = runner._load_state(stage.name) or {}
        expected_signature = signatures[stage.name]
        files = _stage_files(stage, root, state)
        artifacts_valid = stage.artifacts_valid()
        signature_matches = state.get("signature") == expected_signature
        completed = state.get("status") == "completed"
        stage_record = {
            "stage": stage.name,
            "state_status": state.get("status", "missing"),
            "signature": state.get("signature"),
            "expected_signature": expected_signature,
            "signature_matches": signature_matches,
            "artifacts_valid": artifacts_valid,
            "started_at_unix": state.get("started_at_unix"),
            "finished_at_unix": state.get("finished_at_unix"),
            "duration_seconds": state.get("duration_seconds"),
            "return_code": state.get("return_code"),
            "files": [
                _file_record(path, root, hash_mode=hash_mode) for path in files
            ],
        }
        stage_record["content_sha256"] = _fingerprint(stage_record["files"])
        stage_records.append(stage_record)
        contract_checks.append(
            _check(
                f"stage:{stage.name}",
                completed and signature_matches and artifacts_valid and bool(files),
                {
                    "state_status": stage_record["state_status"],
                    "signature_matches": signature_matches,
                    "artifacts_valid": artifacts_valid,
                    "file_count": len(files),
                },
            )
        )

    semantic_evidence, semantic_checks = _semantic_evidence(plan, root)
    contract_checks.extend(semantic_checks)
    contract_status = (
        "pass"
        if contract_checks and all(item["status"] == "pass" for item in contract_checks)
        else "fail"
    )
    gates = semantic_evidence["evaluation"].get("gates") or {}
    capability_status = str(gates.get("overall_status") or "missing")
    quality_claim_authorized = (
        contract_status == "pass" and capability_status == "pass"
    )
    control_files = []
    for name in (
        "resolved_blueprint.yaml",
        "experiment_plan.json",
        "experiment_spec.json",
        "run_summary.json",
    ):
        path = root / name
        if path.is_file():
            control_files.append(_file_record(path, root, hash_mode="full"))
    attested_times = [
        float(record["finished_at_unix"])
        for record in stage_records
        if record["finished_at_unix"] is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": plan.name,
        "experiment_root": str(root),
        "experiment_fingerprint": plan.fingerprint,
        "input_fingerprints": plan.input_fingerprints,
        "hash_mode": hash_mode,
        "attested_at_unix": max(attested_times) if attested_times else None,
        "control_files": control_files,
        "stages": stage_records,
        "semantic_evidence": semantic_evidence,
        "contract_checks": contract_checks,
        "contract_status": contract_status,
        "capability_status": capability_status,
        "quality_claim_authorized": quality_claim_authorized,
        "claim_scope": (
            "deployment_capability"
            if quality_claim_authorized
            else "execution_contract_only"
        ),
    }
    payload["attestation_sha256"] = _fingerprint(payload)
    if output is not None:
        _atomic_write_json(Path(output), payload)
    return payload


def verify_experiment_attestation(
    plan: ExperimentPlan,
    attestation: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Recompute an attestation from current files and compare it byte-semantically."""
    path = Path(attestation)
    observed = _read_json(path)
    hash_mode = str(observed.get("hash_mode") or "")
    expected = build_experiment_attestation(
        plan,
        repo_root=repo_root,
        hash_mode=hash_mode,
    )
    valid = observed == expected
    return {
        "valid": valid,
        "path": str(path.resolve()),
        "observed_attestation_sha256": observed.get("attestation_sha256"),
        "expected_attestation_sha256": expected.get("attestation_sha256"),
        "contract_status": expected["contract_status"],
        "capability_status": expected["capability_status"],
        "claim_scope": expected["claim_scope"],
    }
