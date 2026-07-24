import json
import sys
from pathlib import Path

from docvlm_eval.student.evidence import (
    build_experiment_attestation,
    verify_experiment_attestation,
)
from docvlm_eval.student.experiment import (
    Artifact,
    ExperimentPlan,
    ExperimentRunner,
    ExperimentStage,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evidence_plan(tmp_path: Path) -> ExperimentPlan:
    root = tmp_path / "run"
    initial = root / "artifacts" / "initial" / "metadata.json"
    _write_json(
        initial,
        {
            "initialization_arm": "I0_random",
            "initialization_seed": 5,
            "parameter_counts": {"total": 587_019},
            "transfer_reports": [],
        },
    )
    stages = [
        ExperimentStage(
            "initialize_student",
            (sys.executable, "-c", "pass"),
            (),
            (Artifact(str(initial)),),
        )
    ]
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
            {"run_stage": "pretraining" if stage_name == "pretrain" else stage_name},
        )
        pointer = root / "artifacts" / stage_name / "latest_checkpoint.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(checkpoint), encoding="utf-8")
        stages.append(
            ExperimentStage(
                stage_name,
                (sys.executable, "-c", "pass"),
                (stages[-1].name,),
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
