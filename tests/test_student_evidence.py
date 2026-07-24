import hashlib
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
from docvlm_eval.student.model import build_initialization_lineage


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


def _initialization_lineage() -> dict:
    return build_initialization_lineage(
        initialization_arm="I0_random",
        initialization_seed=5,
        transfer_reports=[],
        architecture_fingerprint="sha256:test-architecture",
    )


def _fingerprint(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _transfer_lineage() -> tuple[dict, str]:
    source_files = [
        {
            "path": "model.pt",
            "bytes": 4,
            "sha256": f"sha256:{'a' * 64}",
        }
    ]
    source_fingerprint = _fingerprint(source_files)
    mappings = [
        {
            "target": "vision.patch_embed.weight",
            "source": "vision.patch_embed.weight",
            "method": "exact",
            "target_shape": [1],
            "source_shape": [1],
            "target_dtype": "float32",
            "source_dtype": "float32",
            "copied_parameters": 1,
            "copied_value_fingerprint": f"sha256:{'b' * 64}",
        }
    ]
    report = {
        "component": "vision",
        "source_topology_fingerprint": f"sha256:{'c' * 64}",
        "target_topology_fingerprint": f"sha256:{'d' * 64}",
        "source_identity": {
            "schema_version": 1,
            "kind": "checkpoint_content",
            "files": source_files,
            "total_bytes": 4,
            "content_fingerprint": source_fingerprint,
        },
        "copied_tensors": 1,
        "copied_keys": ["vision.patch_embed.weight"],
        "tensor_mappings": mappings,
        "mapping_fingerprint": _fingerprint(mappings),
        "copied_values_fingerprint": _fingerprint(
            [mappings[0]["copied_value_fingerprint"]]
        ),
        "value_verified": True,
    }
    return (
        build_initialization_lineage(
            initialization_arm="I1_vision",
            initialization_seed=5,
            transfer_reports=[report],
            architecture_fingerprint="sha256:test-architecture",
        ),
        source_fingerprint,
    )


def _evidence_plan(
    tmp_path: Path,
    *,
    include_initialization: bool = True,
    initialization_lineage: dict | None = None,
    input_fingerprints: dict | None = None,
) -> ExperimentPlan:
    root = tmp_path / "run"
    initial = root / "artifacts" / "initial" / "metadata.json"
    lineage = initialization_lineage or _initialization_lineage()
    stages = []
    if include_initialization:
        _write_json(
            initial,
            {
                "initialization_arm": lineage["initialization_arm"],
                "initialization_seed": lineage["initialization_seed"],
                "parameter_counts": {"total": 587_019},
                "parameter_attestation": _parameter_attestation(),
                "transfer_reports": lineage["transfer_reports"],
                "initialization_lineage": lineage,
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
                "initialization_lineage": lineage,
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
        input_fingerprints={
            "source_code": {"sha256": "sha256:test"},
            **(input_fingerprints or {}),
        },
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


def test_attestation_rejects_training_lineage_drift(tmp_path):
    plan = _evidence_plan(tmp_path)
    metadata = (
        Path(plan.root)
        / "artifacts"
        / "sft"
        / "checkpoints"
        / "step-00000001"
        / "student"
        / "metadata.json"
    )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["initialization_lineage"] = build_initialization_lineage(
        initialization_arm="I0_random",
        initialization_seed=99,
        transfer_reports=[],
        architecture_fingerprint="sha256:test-architecture",
    )
    _write_json(metadata, payload)

    attestation = build_experiment_attestation(plan, repo_root=ROOT)
    lineage_check = next(
        check
        for check in attestation["contract_checks"]
        if check["id"] == "sft_initialization_lineage"
    )

    assert attestation["contract_status"] == "fail"
    assert lineage_check["status"] == "fail"


def test_attestation_binds_transfer_lineage_to_planned_source_content(
    tmp_path,
):
    lineage, source_fingerprint = _transfer_lineage()
    matching = _evidence_plan(
        tmp_path / "matching",
        initialization_lineage=lineage,
        input_fingerprints={
            "initialization_vision_source": {
                "content_fingerprint": source_fingerprint,
                "sha256": source_fingerprint,
            }
        },
    )
    matching_attestation = build_experiment_attestation(
        matching,
        repo_root=ROOT,
    )
    matching_check = next(
        check
        for check in matching_attestation["contract_checks"]
        if check["id"] == "selective_transfer_source_identity"
    )

    mismatching = _evidence_plan(
        tmp_path / "mismatching",
        initialization_lineage=lineage,
        input_fingerprints={
            "initialization_vision_source": {
                "content_fingerprint": f"sha256:{'e' * 64}",
                "sha256": f"sha256:{'e' * 64}",
            }
        },
    )
    mismatching_attestation = build_experiment_attestation(
        mismatching,
        repo_root=ROOT,
    )
    mismatching_check = next(
        check
        for check in mismatching_attestation["contract_checks"]
        if check["id"] == "selective_transfer_source_identity"
    )

    assert matching_check["status"] == "pass"
    assert matching_check["evidence"]["sources"][0]["matches"] is True
    assert mismatching_attestation["contract_status"] == "fail"
    assert mismatching_check["status"] == "fail"
    assert mismatching_check["evidence"]["sources"][0]["matches"] is False
