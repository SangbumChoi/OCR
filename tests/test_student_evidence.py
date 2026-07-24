import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from docvlm_eval.student.evidence import (
    build_experiment_attestation,
    verify_experiment_attestation,
)
from docvlm_eval.student.experiment import (
    Artifact,
    ExperimentPlan,
    ExperimentRunner,
    ExperimentStage,
    build_experiment_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parameter_attestation(total: int = 587_019) -> dict:
    return {
        "schema_version": 1,
        "source": "runtime_numel",
        "architecture_fingerprint": "sha256:test-architecture",
        "parameter_counts": {"total": total},
        "trainability": {
            "trainable_parameters": total,
            "frozen_parameters": 0,
            "trainable_fraction": 1.0,
        },
        "deployment": {
            "parameters_including_task_heads": total,
            "temporary_task_head_parameters": 100,
            "parameters_without_task_heads": total - 100,
        },
        "budget": {
            "max_parameters_exclusive": 1_000_000_000,
            "within_budget": True,
        },
    }


def _evidence_plan(
    tmp_path: Path,
    *,
    include_initialization: bool = True,
) -> ExperimentPlan:
    root = tmp_path / "run"
    initial = root / "artifacts" / "initial" / "metadata.json"
    stages = []
    if include_initialization:
        _write_json(
            initial,
            {
                "initialization_arm": "I0_random",
                "initialization_seed": 5,
                "parameter_counts": {"total": 587_019},
                "parameter_attestation": _parameter_attestation(),
                "transfer_reports": [],
            },
        )
        stages.append(
            ExperimentStage(
                "initialize_student",
                (sys.executable, "-c", "pass"),
                (),
                (Artifact(str(initial)),),
            )
        )
    for stage_name, trainer_state in (
        ("pretrain", {"global_step": 1, "effective_tokens_seen": 804}),
        ("sft", {"global_step": 1, "effective_tokens_seen": 1024}),
        ("rlvr", {"rollout_step": 1, "optimizer_step": 1}),
    ):
        checkpoint = (
            root
            / "artifacts"
            / stage_name
            / "checkpoints"
            / "step-00000001"
        )
        _write_json(checkpoint / "trainer_state.json", trainer_state)
        _write_json(
            checkpoint / "student" / "metadata.json",
            {
                "run_stage": (
                    "pretraining" if stage_name == "pretrain" else stage_name
                ),
                "parameter_attestation": _parameter_attestation(),
            },
        )
        pointer = root / "artifacts" / stage_name / "latest_checkpoint.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(checkpoint), encoding="utf-8")
        stages.append(
            ExperimentStage(
                stage_name,
                (sys.executable, "-c", "pass"),
                (stages[-1].name,) if stages else (),
                (Artifact(str(pointer)),),
            )
        )

    evaluation = root / "artifacts" / "evaluation"
    manifest = evaluation / "manifest.json"
    comparison = evaluation / "comparison.json"
    gates = evaluation / "gates.json"
    _write_json(
        manifest,
        {"checkpoint_metadata": {"run_stage": "rlvr"}},
    )
    _write_json(
        comparison,
        {
            "splits": {
                "train": {"n_samples": 1, "score": 0.0},
                "heldout": {"n_samples": 1, "score": 0.0},
            }
        },
    )
    _write_json(
        gates,
        {
            "schema_version": 1,
            "overall_status": "insufficient_evidence",
            "gates": [],
        },
    )
    stages.append(
        ExperimentStage(
            "evaluate",
            (sys.executable, "-c", "pass"),
            ("rlvr",),
            (Artifact(str(manifest)), Artifact(str(comparison)), Artifact(str(gates))),
        )
    )
    plan = ExperimentPlan(
        name="evidence-test",
        root=str(root),
        blueprint=str(root / "resolved_blueprint.yaml"),
        resolved_blueprint={"schema_version": 1},
        raw_spec={"schema_version": 1},
        components=(),
        stages=tuple(stages),
        fingerprint="sha256:evidence-test",
        input_fingerprints={"source_code": {"sha256": "sha256:test"}},
    )
    runner = ExperimentRunner(plan, repo_root=ROOT)
    signatures = runner.signatures()
    for stage in plan.stages:
        log = root / "logs" / f"{stage.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"{stage.name} completed\n", encoding="utf-8")
        _write_json(
            root / "state" / "stages" / f"{stage.name}.json",
            {
                "stage": stage.name,
                "status": "completed",
                "signature": signatures[stage.name],
                "return_code": 0,
                "started_at_unix": 1.0,
                "finished_at_unix": 2.0,
                "duration_seconds": 1.0,
                "log": str(log),
            },
        )
    _write_json(root / "run_summary.json", {"fingerprint": plan.fingerprint})
    return plan


def test_attestation_separates_execution_contract_from_capability_claim(tmp_path):
    plan = _evidence_plan(tmp_path)
    output = Path(plan.root) / "evidence_attestation.json"

    attestation = build_experiment_attestation(
        plan,
        repo_root=ROOT,
        output=output,
    )

    assert attestation["contract_status"] == "pass"
    assert attestation["capability_status"] == "insufficient_evidence"
    assert attestation["claim_scope"] == "execution_contract_only"
    assert attestation["quality_claim_authorized"] is False
    assert all(
        check["status"] == "pass" for check in attestation["contract_checks"]
    )
    assert verify_experiment_attestation(
        plan,
        output,
        repo_root=ROOT,
    )["valid"]


def test_attestation_verification_detects_checkpoint_tampering(tmp_path):
    plan = _evidence_plan(tmp_path)
    output = Path(plan.root) / "evidence_attestation.json"
    build_experiment_attestation(plan, repo_root=ROOT, output=output)
    trainer_state = (
        Path(plan.root)
        / "artifacts"
        / "rlvr"
        / "checkpoints"
        / "step-00000001"
        / "trainer_state.json"
    )
    _write_json(trainer_state, {"rollout_step": 2, "optimizer_step": 1})

    result = verify_experiment_attestation(
        plan,
        output,
        repo_root=ROOT,
    )

    assert result["valid"] is False
    assert (
        result["observed_attestation_sha256"]
        != result["expected_attestation_sha256"]
    )


def test_attestation_uses_runtime_checkpoint_budget_for_continuation(tmp_path):
    plan = _evidence_plan(tmp_path, include_initialization=False)
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "blueprint-reference")
    config = tmp_path / "blueprint-reference.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    blueprint_reference = build_experiment_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
    )
    plan = replace(plan, resolved_blueprint=blueprint_reference.resolved_blueprint)

    attestation = build_experiment_attestation(plan, repo_root=ROOT)

    budget = next(
        check
        for check in attestation["contract_checks"]
        if check["id"] == "deployment_parameter_budget"
    )
    assert budget["status"] == "pass"
    assert budget["evidence"] == {
        "actual_parameters": 587_019,
        "max_parameters_exclusive": 1_000_000_000,
        "source": "rlvr_checkpoint",
        "architecture_fingerprint": "sha256:test-architecture",
    }


def test_attestation_rejects_missing_runtime_parameter_measurement(tmp_path):
    plan = _evidence_plan(tmp_path)
    initialization = (
        Path(plan.root) / "artifacts" / "initial" / "metadata.json"
    )
    initialization_payload = json.loads(
        initialization.read_text(encoding="utf-8")
    )
    initialization_payload.pop("parameter_attestation")
    _write_json(initialization, initialization_payload)
    for stage_name in ("pretrain", "sft", "rlvr"):
        metadata = (
            Path(plan.root)
            / "artifacts"
            / stage_name
            / "checkpoints"
            / "step-00000001"
            / "student"
            / "metadata.json"
        )
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload.pop("parameter_attestation")
        _write_json(metadata, payload)

    attestation = build_experiment_attestation(plan, repo_root=ROOT)
    budget = next(
        check
        for check in attestation["contract_checks"]
        if check["id"] == "deployment_parameter_budget"
    )

    assert attestation["contract_status"] == "fail"
    assert budget["status"] == "fail"
    assert budget["evidence"]["source"] == "missing_runtime_attestation"
