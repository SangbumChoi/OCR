import json
import sys
from pathlib import Path

import pytest

from docvlm_eval.student.experiment import (
    Artifact,
    ExperimentPlan,
    ExperimentRunner,
    ExperimentStage,
    _resolve_command,
    _with_training_resume,
    build_experiment_plan,
)
from docvlm_eval.student.config import StudentConfig


ROOT = Path(__file__).resolve().parents[1]


def test_default_experiment_compiles_complete_stage_dag():
    plan = build_experiment_plan(
        ROOT / "configs" / "sub1b_experiment.yaml",
        repo_root=ROOT,
        python=sys.executable,
    )
    assert plan.stage_names == [
        "synthetic_train",
        "synthetic_heldout",
        "validate_synthetic_splits",
        "build_synthetic_udd",
        "build_train_samples",
        "build_heldout_samples",
        "mix_pretraining_data",
        "train_tokenizer",
        "initialize_student",
        "pretrain",
        "sft",
        "rlvr",
        "evaluate",
    ]
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    assert pipeline["balance_by"] == "component"
    assert pipeline["group_weights"] == {"synthetic_documents": 1.0}
    sft = next(stage for stage in plan.stages if stage.name == "sft")
    assert sft.command[0] == sys.executable
    assert sft.command.count(sys.executable) == 1
    assert "@student:pretrain" in sft.command


def test_tiny_student_can_host_byte_level_tokenizer():
    assert StudentConfig.tiny(vocab_size=512).language.vocab_size >= 260


def test_tiny_experiment_resolves_one_consistent_pipeline():
    plan = build_experiment_plan(
        ROOT / "configs" / "sub1b_experiment_tiny.yaml",
        repo_root=ROOT,
        python=sys.executable,
    )
    tiny = StudentConfig.tiny(vocab_size=512)
    assert plan.resolved_blueprint["student"]["vision"]["image_size"] == tiny.vision.image_size
    assert plan.resolved_blueprint["tokenizer"]["vocab_size"] == tiny.language.vocab_size
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    assert pipeline["max_image_long_side"] == tiny.vision.image_size
    initialize = next(stage for stage in plan.stages if stage.name == "initialize_student")
    assert initialize.command[initialize.command.index("--tiny-vocab-size") + 1] == "512"


def test_invalid_experiment_rejects_equal_split_seeds(tmp_path):
    raw = (ROOT / "configs" / "sub1b_experiment.yaml").read_text(encoding="utf-8")
    config = tmp_path / "invalid.yaml"
    config.write_text(raw.replace("heldout_seed: 7007", "heldout_seed: 7"), encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def _runner_plan(tmp_path: Path) -> ExperimentPlan:
    root = tmp_path / "run"
    first = root / "first.txt"
    second = root / "second.txt"
    stages = (
        ExperimentStage(
            "first",
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(first)!r}).write_text('one')",
            ),
            (),
            (Artifact(str(first)),),
        ),
        ExperimentStage(
            "second",
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(second)!r}).write_text('two')",
            ),
            ("first",),
            (Artifact(str(second)),),
        ),
    )
    return ExperimentPlan(
        name="test",
        root=str(root),
        blueprint=str(root / "resolved_blueprint.yaml"),
        resolved_blueprint={"schema_version": 1},
        raw_spec={"schema_version": 1},
        components=(),
        stages=stages,
        fingerprint="sha256:test",
    )


def test_runner_dry_run_is_read_only_and_resume_checks_artifacts(tmp_path):
    plan = _runner_plan(tmp_path)
    runner = ExperimentRunner(plan, repo_root=ROOT)
    dry_run = runner.run(dry_run=True)
    assert [stage["name"] for stage in dry_run["stages"]] == ["first", "second"]
    assert not Path(plan.root).exists()

    first_run = runner.run()
    assert [item["status"] for item in first_run["outcomes"]] == [
        "completed",
        "completed",
    ]
    resumed = runner.run()
    assert [item["status"] for item in resumed["outcomes"]] == ["skipped", "skipped"]

    (Path(plan.root) / "second.txt").unlink()
    repaired = runner.run()
    assert [item["status"] for item in repaired["outcomes"]] == ["skipped", "completed"]
    state = json.loads(
        (Path(plan.root) / "state" / "stages" / "second.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"


def test_checkpoint_placeholder_and_training_resume(tmp_path):
    root = tmp_path / "run"
    checkpoint = tmp_path / "checkpoint-1"
    student = checkpoint / "student"
    student.mkdir(parents=True)
    (student / "model.pt").write_bytes(b"model")
    output = root / "artifacts" / "pretrain"
    output.mkdir(parents=True)
    (output / "latest_checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")

    command = _resolve_command(("tool", "@student:pretrain"), root)
    assert command == ["tool", str(student)]
    assert _with_training_resume(command, "pretrain", root, eligible=True) == [
        "tool",
        str(student),
        "--resume",
        "latest",
    ]
    assert _with_training_resume(command, "pretrain", root, eligible=False) == command
