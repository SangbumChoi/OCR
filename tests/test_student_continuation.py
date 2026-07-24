import json
import sys
from pathlib import Path

import pytest
import yaml

from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.continuation import (
    build_curriculum_samples,
    prepare_next_round_spec,
    resolve_continuation_contract,
    write_round_spec,
)
from docvlm_eval.student.experiment import build_experiment_plan
from docvlm_eval.student.model import DocumentVLMStudent
from docvlm_eval.student.synthesis_policy import (
    file_fingerprint,
    payload_fingerprint,
)
from docvlm_eval.student.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _attested_record(path, parent_root):
    return {
        "path": str(path.resolve().relative_to(parent_root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": file_fingerprint(path),
    }


def _parent_run(tmp_path):
    parent_root = tmp_path / "round-000"
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(parent_root)
    raw["synthetic"]["validation_count"] = 1
    raw["synthetic"]["validation_seed"] = 1707
    raw["synthetic"]["adaptation_policy"] = {
        "enabled": True,
        "config": "configs/sub1b_synthesis_policy.yaml",
        "budget": 1,
        "seed": 91,
    }
    config = tmp_path / "round-000.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    plan = build_experiment_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
    )
    _write_json(parent_root / "experiment_plan.json", plan.to_dict())
    _write_json(parent_root / "experiment_spec.json", plan.raw_spec)
    summary = {
        "schema_version": 2,
        "fingerprint": plan.fingerprint,
        "pipeline_complete": True,
        "stages": [
            {
                "stage": stage.name,
                "state_status": "completed",
                "signature_matches": True,
                "artifacts_valid": True,
            }
            for stage in plan.stages
        ],
    }
    _write_json(parent_root / "run_summary.json", summary)

    tokenizer = DocumentTokenizer.train(
        [" ".join(f"token-{index:04d}" for index in range(2_000))],
        vocab_size=512,
        min_frequency=1,
        show_progress=False,
    )
    tokenizer_root = parent_root / "artifacts" / "tokenizer"
    tokenizer.save_pretrained(tokenizer_root)
    checkpoint_root = (
        parent_root
        / "artifacts"
        / "rlvr"
        / "checkpoints"
        / "step-00000001"
    )
    student_root = checkpoint_root / "student"
    student = DocumentVLMStudent(
        StudentConfig.from_blueprint(plan.resolved_blueprint)
    )
    student.save_pretrained(
        student_root,
        {
            "run_stage": "rlvr",
            "tokenizer_fingerprint": tokenizer.fingerprint,
        },
    )
    pointer = parent_root / "artifacts" / "rlvr" / "latest_checkpoint.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(checkpoint_root), encoding="utf-8")

    replay_samples = parent_root / "artifacts" / "samples" / "train.jsonl"
    replay_samples.parent.mkdir(parents=True, exist_ok=True)
    replay_samples.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": f"parent-{index}",
                    "image_path": "image.png",
                    "question": "Question?",
                    "answers": ["42"],
                    "answer_type": "kie",
                    "metric": "anls",
                    "meta": {},
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    source = (
        parent_root
        / "artifacts"
        / "evaluation"
        / "validation"
        / "per_sample.jsonl"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps({"sample_id": "validation-0", "split": "validation"})
        + "\n",
        encoding="utf-8",
    )
    policy = {
        "schema_version": 1,
        "policy": "factor_shrinkage_failure_curriculum",
        "training_authorized": True,
        "claim_scope": "next_training_batch",
        "source": {
            "split": "validation",
            "path": str(source),
            "fingerprint": file_fingerprint(source),
            "rows": 1,
        },
        "budget": 1,
        "seed": 91,
        "jobs": [
            {
                "arm_id": "arm-0",
                "generator_case": "hard_table",
                "language": "en",
                "difficulty_level": 2,
                "layout_family": "classic-v1",
                "composition_tier": "single_document",
                "count": 1,
                "seed": 91,
                "output_subdir": "job-0000",
            }
        ],
    }
    policy["plan_fingerprint"] = payload_fingerprint(policy)
    policy_path = (
        parent_root / "artifacts" / "synthetic" / "next_train_plan.json"
    )
    _write_json(policy_path, policy)

    control_paths = [
        parent_root / "experiment_plan.json",
        parent_root / "experiment_spec.json",
        parent_root / "run_summary.json",
    ]
    handoff_paths = [
        replay_samples,
        policy_path,
        student_root / "student_config.json",
        student_root / "model.pt",
        student_root / "metadata.json",
    ]
    attestation = {
        "schema_version": 1,
        "experiment_root": str(parent_root),
        "experiment_fingerprint": plan.fingerprint,
        "hash_mode": "full",
        "contract_status": "pass",
        "control_files": [
            _attested_record(path, parent_root) for path in control_paths
        ],
        "stages": [
            {
                "stage": "handoff",
                "files": [
                    _attested_record(path, parent_root)
                    for path in handoff_paths
                ],
            }
        ],
    }
    attestation["attestation_sha256"] = payload_fingerprint(attestation)
    _write_json(parent_root / "evidence_attestation.json", attestation)
    return parent_root, plan, student_root


def test_continuation_compiles_model_preserving_adaptation_dag(tmp_path):
    parent_root, parent_plan, student_root = _parent_run(tmp_path)
    child_root = tmp_path / "round-001"
    spec = prepare_next_round_spec(
        parent_root=parent_root,
        output_root=child_root,
        round_index=1,
        replay_fraction=0.5,
        replay_seed=20_001,
    )
    config = tmp_path / "round-001.yaml"
    write_round_spec(spec, config)

    plan = build_experiment_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
    )

    assert plan.name == "docvlm-tiny-smoke--round-001"
    assert plan.stage_names == [
        "attest_continuation",
        "synthetic_train",
        "synthetic_validation",
        "synthetic_heldout",
        "validate_synthetic_splits",
        "build_train_samples",
        "build_heldout_samples",
        "build_validation_samples",
        "build_curriculum_samples",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
        "plan_next_synthetic_batch",
    ]
    assert not {
        "train_tokenizer",
        "initialize_student",
        "pretrain",
        "mix_pretraining_data",
        "generate_teacher_predictions",
    }.intersection(plan.stage_names)
    sft = next(stage for stage in plan.stages if stage.name == "sft")
    assert str(student_root) in sft.command
    assert str(child_root / "artifacts" / "tokenizer") in sft.command
    assert "build_curriculum_samples" in sft.dependencies
    attest = next(
        stage for stage in plan.stages if stage.name == "attest_continuation"
    )
    assert str(parent_root / "artifacts" / "tokenizer") not in attest.command
    assert str(child_root / "artifacts" / "tokenizer") in attest.command
    contract = plan.input_fingerprints["continuation_contract"]
    assert contract["parent_experiment_fingerprint"] == parent_plan.fingerprint
    assert contract["optimizer_policy"] == "reset_per_stage"
    assert contract["replay_source_kind"] == "base_train"
    assert contract["replay_origin_rounds"] == [0]
    assert contract["schema_version"] == 2
    assert spec["evaluation"]["baseline_checkpoint_stage"] == "inherited"
    assert spec["evaluation"]["baseline_evaluation"] is None
    baseline = next(
        stage for stage in plan.stages if stage.name == "evaluate_baseline"
    )
    assert baseline.command[
        baseline.command.index("--checkpoint") + 1
    ] == str(student_root)
    assert baseline.dependencies[0] == "attest_continuation"
    assert "build_curriculum_samples" in baseline.dependencies
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert evaluate.command[
        evaluate.command.index("--baseline-evaluation") + 1
    ] == str(child_root / "artifacts" / "evaluation_baseline")


def test_continuation_rejects_checkpoint_content_changed_after_attestation(
    tmp_path,
):
    parent_root, plan, student_root = _parent_run(tmp_path)
    with (student_root / "model.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="full attestation"):
        resolve_continuation_contract(
            {
                "enabled": True,
                "parent_root": str(parent_root),
                "round_index": 1,
                "optimizer_policy": "reset_per_stage",
                "replay_fraction": 0.5,
                "replay_seed": 20_001,
            },
            repo_root=ROOT,
            blueprint=plan.resolved_blueprint,
        )


def test_curriculum_samples_keep_all_new_rows_and_deterministic_replay(
    tmp_path,
):
    current = tmp_path / "current.jsonl"
    replay = tmp_path / "replay.jsonl"
    current.write_text(
        "\n".join(
            json.dumps({"sample_id": f"new-{index}", "meta": {}})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        "\n".join(
            json.dumps({"sample_id": f"old-{index}", "meta": {}})
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined.jsonl"
    memory = tmp_path / "memory.jsonl"
    manifest_path = tmp_path / "combined.manifest.json"

    manifest = build_curriculum_samples(
        current_samples=current,
        replay_samples=replay,
        replay_fraction=0.5,
        replay_seed=13,
        parent_round_index=0,
        output=output,
        memory_output=memory,
        manifest_output=manifest_path,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["new_sample_count"] == 2
    assert manifest["selected_replay_count"] == 2
    assert manifest["realized_replay_fraction"] == 0.5
    assert manifest["memory_sample_count"] == 7
    assert manifest["memory_origin_counts"] == {"0": 5, "1": 2}
    assert manifest["selected_replay_origin_counts"] == {"0": 2}
    assert {
        row["sample_id"]
        for row in rows
        if row["meta"]["curriculum_role"] == "new_failure_batch"
    } == {"new-0", "new-1"}
    assert sum(
        row["meta"]["curriculum_role"] == "parent_replay"
        for row in rows
    ) == 2
    assert len({row["sample_id"] for row in rows}) == 4
    assert json.loads(
        manifest_path.read_text(encoding="utf-8")
    ) == manifest

    next_current = tmp_path / "next-current.jsonl"
    next_current.write_text(
        "\n".join(
            json.dumps({"sample_id": f"next-{index}", "meta": {}})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    next_output = tmp_path / "next-combined.jsonl"
    next_memory = tmp_path / "next-memory.jsonl"
    next_manifest = build_curriculum_samples(
        current_samples=next_current,
        replay_samples=memory,
        replay_fraction=0.5,
        replay_seed=17,
        parent_round_index=1,
        output=next_output,
        memory_output=next_memory,
        manifest_output=tmp_path / "next-combined.manifest.json",
    )
    next_rows = [
        json.loads(line)
        for line in next_output.read_text(encoding="utf-8").splitlines()
    ]
    next_memory_rows = [
        json.loads(line)
        for line in next_memory.read_text(encoding="utf-8").splitlines()
    ]

    assert next_manifest["replay_origin_rounds"] == [0, 1]
    assert next_manifest["selected_replay_origin_counts"] == {"0": 1, "1": 1}
    assert next_manifest["memory_sample_count"] == 9
    assert next_manifest["memory_origin_counts"] == {"0": 5, "1": 2, "2": 2}
    assert {
        row["meta"]["curriculum_origin_round_index"]
        for row in next_rows
        if row["meta"]["curriculum_role"] == "parent_replay"
    } == {0, 1}
    assert {
        row["meta"]["curriculum_origin_round_index"]
        for row in next_memory_rows
    } == {0, 1, 2}

    malformed_memory = tmp_path / "malformed-memory.jsonl"
    malformed_rows = [
        json.loads(line)
        for line in memory.read_text(encoding="utf-8").splitlines()
    ]
    malformed_rows[0]["meta"]["curriculum_origin_round_index"] = 1
    malformed_memory.write_text(
        "\n".join(json.dumps(row) for row in malformed_rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="origin lineage"):
        build_curriculum_samples(
            current_samples=next_current,
            replay_samples=malformed_memory,
            replay_fraction=0.5,
            replay_seed=17,
            parent_round_index=1,
            output=tmp_path / "rejected.jsonl",
            memory_output=tmp_path / "rejected-memory.jsonl",
            manifest_output=tmp_path / "rejected.manifest.json",
        )
