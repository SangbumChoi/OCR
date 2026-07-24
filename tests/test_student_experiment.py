import json
import sys
from pathlib import Path

import pytest
import yaml

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
        "visual_backend_benchmark",
        "training_feasibility_benchmark",
        "synthetic_train",
        "synthetic_heldout",
        "validate_synthetic_splits",
        "build_synthetic_udd",
        "build_train_samples",
        "build_heldout_samples",
        "acquire_component_public_udd",
        "mix_pretraining_data",
        "export_teacher_requests",
        "generate_teacher_predictions",
        "apply_teacher_targets",
        "train_tokenizer",
        "initialize_student",
        "pretrain",
        "sft",
        "rlvr",
        "evaluate",
    ]
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    assert pipeline["balance_by"] == "component"
    assert pipeline["group_weights"] == {
        "synthetic_documents": pytest.approx(0.45),
        "public_udd": pytest.approx(0.55),
    }
    sequence_targets = plan.resolved_blueprint["training"]["pretraining"]["distillation"][
        "sequence_targets"
    ]
    assert sequence_targets["probability"] == pytest.approx(0.5)
    acquisition = next(
        stage for stage in plan.stages if stage.name == "acquire_component_public_udd"
    )
    assert acquisition.command[acquisition.command.index("--revision") + 1] == (
        "f5eb52104627d20ddd1eab2130ad78f87cb0d7c9"
    )
    mixture = next(stage for stage in plan.stages if stage.name == "mix_pretraining_data")
    assert "acquire_component_public_udd" in mixture.dependencies
    sft = next(stage for stage in plan.stages if stage.name == "sft")
    assert sft.command[0] == sys.executable
    assert sft.command.count(sys.executable) == 1
    assert "@student:pretrain" in sft.command
    export = next(
        stage for stage in plan.stages if stage.name == "export_teacher_requests"
    )
    assert export.command[export.command.index("--max-requests") + 1] == "4096"
    generate = next(
        stage
        for stage in plan.stages
        if stage.name == "generate_teacher_predictions"
    )
    assert generate.command[generate.command.index("--model-revision") + 1] == (
        "919fde3d022e3f90a4716006f993938ee8c2eb97"
    )
    apply = next(
        stage for stage in plan.stages if stage.name == "apply_teacher_targets"
    )
    assert (
        apply.command[apply.command.index("--accepted-target-count") + 1]
        == "400"
    )
    tokenizer = next(
        stage for stage in plan.stages if stage.name == "train_tokenizer"
    )
    assert "--exclude-teacher-targets" in tokenizer.command
    visual_benchmark = next(
        stage for stage in plan.stages if stage.name == "visual_backend_benchmark"
    )
    assert visual_benchmark.command[
        visual_benchmark.command.index("--patch-grids") + 1
    ] == "40x63,63x40"
    assert visual_benchmark.command[
        visual_benchmark.command.index("--backends") + 1 :
        visual_benchmark.command.index("--warmup-iterations")
    ] == (
        "loop",
        "auto",
        "flex",
        "dense_adaptive",
        "dense_fixed_square",
    )
    assert "--require-flex" not in visual_benchmark.command
    assert "--require-deployment-gate" in visual_benchmark.command
    assert visual_benchmark.command[
        visual_benchmark.command.index("--rounds") + 1
    ] == "3"
    assert visual_benchmark.artifacts[0].path.endswith(
        "artifacts/benchmarks/visual_backend.json"
    )
    training_benchmark = next(
        stage
        for stage in plan.stages
        if stage.name == "training_feasibility_benchmark"
    )
    assert training_benchmark.dependencies == (
        "visual_backend_benchmark",
    )
    assert training_benchmark.command[
        training_benchmark.command.index("--patch-grid") + 1
    ] == "40x63"
    assert training_benchmark.command[
        training_benchmark.command.index("--text-tokens") + 1
    ] == "2048"
    assert "--require-deployment-gate" in training_benchmark.command
    assert training_benchmark.artifacts[0].path.endswith(
        "artifacts/benchmarks/training_feasibility.json"
    )
    initialize = next(
        stage for stage in plan.stages if stage.name == "initialize_student"
    )
    assert "visual_backend_benchmark" in initialize.dependencies
    assert "training_feasibility_benchmark" in initialize.dependencies
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert "--visual-backend-benchmark" in evaluate.command
    assert "--training-feasibility-benchmark" in evaluate.command
    assert "--no-kv-cache" not in evaluate.command
    assert "visual_backend_benchmark" in evaluate.dependencies
    assert "training_feasibility_benchmark" in evaluate.dependencies


def test_experiment_can_compile_full_prefix_generation_ablation(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["use_kv_cache"] = False
    config = tmp_path / "uncached-evaluation.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = build_experiment_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
    )
    evaluate = next(
        stage for stage in plan.stages if stage.name == "evaluate"
    )

    assert "--no-kv-cache" in evaluate.command


def test_experiment_rejects_non_boolean_generation_cache(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["use_kv_cache"] = "yes"
    config = tmp_path / "invalid-cache.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use_kv_cache must be a boolean"):
        build_experiment_plan(
            config,
            repo_root=ROOT,
            python=sys.executable,
        )


def test_experiment_can_evaluate_the_sft_checkpoint_without_rlvr(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["posttraining"]["rlvr"] = {
        "enabled": False,
        "max_steps": None,
        "replay_every_steps": None,
        "replay_loss_coefficient": None,
        "replay_samples": None,
    }
    config = tmp_path / "sft-only.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    assert "rlvr" not in plan.stage_names
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert "@student:sft" in evaluate.command
    assert evaluate.dependencies == (
        "sft",
        "build_train_samples",
        "build_heldout_samples",
    )


def test_experiment_rejects_ignored_disabled_rlvr_overrides(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["posttraining"]["rlvr"]["enabled"] = False
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "invalid-sft-only.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="disabled RLVR cannot set"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_tiny_student_can_host_byte_level_tokenizer():
    assert StudentConfig.tiny(vocab_size=512).language.vocab_size >= 260


def test_tiny_experiment_resolves_one_consistent_pipeline():
    plan = build_experiment_plan(
        ROOT / "configs" / "sub1b_experiment_tiny.yaml",
        repo_root=ROOT,
        python=sys.executable,
    )
    tiny = StudentConfig.tiny(vocab_size=512)
    assert len(plan.stages) == 16
    assert plan.resolved_blueprint["student"]["vision"]["image_size"] == tiny.vision.image_size
    assert plan.resolved_blueprint["tokenizer"]["vocab_size"] == tiny.language.vocab_size
    assert "visual_backend_benchmark" not in plan.stage_names
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    assert pipeline["max_image_long_side"] == tiny.vision.image_size
    initialize = next(stage for stage in plan.stages if stage.name == "initialize_student")
    assert initialize.command[initialize.command.index("--tiny-vocab-size") + 1] == "512"
    assert initialize.command[initialize.command.index("--seed") + 1] == "5"
    assert "export_teacher_requests" in plan.stage_names
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert any(artifact.path.endswith("gates.json") for artifact in evaluate.artifacts)
    generate = next(
        stage for stage in plan.stages if stage.name == "generate_teacher_predictions"
    )
    assert generate.command[generate.command.index("--model") + 1] == "dummy-echo"
    rlvr = next(stage for stage in plan.stages if stage.name == "rlvr")
    assert rlvr.command[rlvr.command.index("--replay-every-steps") + 1] == "1"
    assert (
        rlvr.command[rlvr.command.index("--replay-loss-coefficient") + 1]
        == "0.1"
    )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"sequence_lengths": [0]}, "positive sequence_lengths"),
        ({"backends": ["auto", "flex"]}, "must include loop"),
        (
            {"backends": ["loop"], "require_flex": True},
            "require_flex needs auto or flex",
        ),
        ({"iterations": 0}, "iterations must be positive"),
        ({"rounds": 0}, "rounds must be positive"),
        (
            {"sequence_lengths": [65]},
            "sequence_lengths exceed the resolved visual position grid",
        ),
        ({"require_flex": "yes"}, "require_flex must be a boolean"),
        (
            {"require_deployment_gate": "yes"},
            "require_deployment_gate must be a boolean",
        ),
        (
            {"backends": ["loop", "dense_adaptive"]},
            "dense policies require patch_grids",
        ),
    ],
)
def test_experiment_rejects_invalid_visual_backend_benchmark(
    tmp_path,
    patch,
    message,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["runtime"]["visual_backend_benchmark"] = {
        "enabled": True,
        "sequence_lengths": [3, 5],
        "backends": ["loop", "auto"],
        **patch,
    }
    config = tmp_path / "invalid-visual-benchmark.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_experiment_rejects_patch_grid_beyond_visual_position_side(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["runtime"]["visual_backend_benchmark"] = {
        "enabled": True,
        "patch_grids": [[1, 9]],
        "backends": ["loop", "dense_adaptive"],
    }
    config = tmp_path / "invalid-visual-grid.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="patch_grids exceed the resolved visual position grid",
    ):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"patch_grid": [0, 2]}, "positive.*patch_grid"),
        ({"text_tokens": 0}, "text_tokens must be positive"),
        ({"micro_batch_size": 0}, "micro_batch_size must be positive"),
        ({"measured_steps": 0}, "measured_steps must be positive"),
        ({"warmup_steps": -1}, "warmup_steps must be non-negative"),
        (
            {"packed_attention_backend": "dense"},
            "packed_attention_backend is unsupported",
        ),
        (
            {"require_deployment_gate": "yes"},
            "require_deployment_gate must be a boolean",
        ),
    ],
)
def test_experiment_rejects_invalid_training_feasibility_benchmark(
    tmp_path,
    patch,
    message,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["runtime"]["training_feasibility_benchmark"] = {
        "enabled": True,
        "patch_grid": [1, 2],
        "text_tokens": 8,
        "micro_batch_size": 1,
        "measured_steps": 1,
        **patch,
    }
    config = tmp_path / "invalid-training-benchmark.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_experiment_supports_independent_train_and_heldout_synthetic_counts(
    tmp_path,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["synthetic"]["train_count"] = 2
    raw["synthetic"]["heldout_count"] = 5
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    train = next(stage for stage in plan.stages if stage.name == "synthetic_train")
    heldout = next(
        stage for stage in plan.stages if stage.name == "synthetic_heldout"
    )

    assert train.command[train.command.index("--count") + 1] == "2"
    assert heldout.command[heldout.command.index("--count") + 1] == "5"


def test_invalid_experiment_rejects_equal_split_seeds(tmp_path):
    raw = (ROOT / "configs" / "sub1b_experiment.yaml").read_text(encoding="utf-8")
    config = tmp_path / "invalid.yaml"
    config.write_text(raw.replace("heldout_seed: 7007", "heldout_seed: 7"), encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_invalid_experiment_rejects_negative_initialization_seed(tmp_path):
    raw = (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(encoding="utf-8")
    config = tmp_path / "invalid.yaml"
    config.write_text(raw.replace("seed: 5", "seed: -1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization.seed must be non-negative"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_experiment_rejects_unpinned_sequence_teacher(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["sequence_teacher"]["revision"] = None
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "unpinned-teacher.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned 40-character"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_experiment_rejects_online_teacher_loss_without_checkpoint(tmp_path):
    blueprint = yaml.safe_load(
        (ROOT / "configs" / "sub1b_architecture.yaml").read_text(
            encoding="utf-8"
        )
    )
    blueprint["training"]["pretraining"]["losses"]["teacher_kl"] = 0.2
    blueprint_path = tmp_path / "blueprint.yaml"
    blueprint_path.write_text(
        yaml.safe_dump(blueprint, sort_keys=False),
        encoding="utf-8",
    )
    experiment = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    experiment["blueprint"] = str(blueprint_path)
    experiment["output_root"] = str(tmp_path / "output")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(experiment, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require a native teacher"):
        build_experiment_plan(
            experiment_path,
            repo_root=ROOT,
            python=sys.executable,
        )


def _write_initialization_experiment(tmp_path, initialization):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["initialization"].update(initialization)
    raw["blueprint"] = str(ROOT / "configs" / "sub1b_architecture.yaml")
    raw["synthetic"]["config"] = str(ROOT / "configs" / "synth_data.yaml")
    raw["output_root"] = str(tmp_path / "output")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_experiment_compiles_pinned_hub_initialization_sources(tmp_path):
    revision = "a" * 40
    config = _write_initialization_experiment(
        tmp_path,
        {
            "arm": "I4_selective",
            "vision_family": "siglip",
            "vision_source": {
                "hub": {
                    "repo_id": "google/siglip-base-patch16-224",
                    "revision": revision,
                }
            },
            "language_family": "llama",
            "language_source": {
                "hub": {
                    "repo_id": "Qwen/Qwen2.5-1.5B",
                    "revision": revision,
                }
            },
        },
    )

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    initialize = next(
        stage for stage in plan.stages if stage.name == "initialize_student"
    )

    assert plan.stage_names[:2] == [
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
    ]
    assert initialize.dependencies == (
        "train_tokenizer",
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
    )
    assert "@checkpoint:vision" in initialize.command
    assert "@checkpoint:language" in initialize.command
    assert "--vision-family" in initialize.command
    assert "--language-family" in initialize.command
    assert next(
        stage
        for stage in plan.stages
        if stage.name == "acquire_vision_checkpoint"
    ).artifacts[0].kind == "checkpoint_manifest"


def test_experiment_fingerprints_local_initialization_sources(tmp_path):
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"first")
    config = _write_initialization_experiment(
        tmp_path,
        {
            "arm": "I4_selective",
            "vision_family": "student",
            "vision_source": str(checkpoint),
            "language_family": "student",
            "language_source": str(checkpoint),
        },
    )

    first = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    (checkpoint / "model.pt").write_bytes(b"second")
    second = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    assert first.fingerprint != second.fingerprint
    assert (
        first.input_fingerprints["initialization_vision_source"]["sha256"]
        != second.input_fingerprints["initialization_vision_source"]["sha256"]
    )


def test_experiment_rejects_transfer_arm_without_required_source(tmp_path):
    config = _write_initialization_experiment(
        tmp_path,
        {"arm": "I1_vision"},
    )

    with pytest.raises(ValueError, match="requires sources"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_experiment_fingerprint_tracks_synthetic_config_content(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    synthetic_config = tmp_path / "synth.yaml"
    original = (ROOT / "configs" / "synth_data.yaml").read_text(encoding="utf-8")
    synthetic_config.write_text(original, encoding="utf-8")
    raw["synthetic"]["config"] = str(synthetic_config)
    raw["blueprint"] = str(ROOT / "configs" / "sub1b_architecture.yaml")
    raw["output_root"] = str(tmp_path / "output")
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    first = build_experiment_plan(
        experiment_config,
        repo_root=ROOT,
        python=sys.executable,
    )
    synthetic_config.write_text(original + "\n# provenance change\n", encoding="utf-8")
    second = build_experiment_plan(
        experiment_config,
        repo_root=ROOT,
        python=sys.executable,
    )

    assert first.fingerprint != second.fingerprint
    assert (
        first.input_fingerprints["synthetic_config"]["sha256"]
        != second.input_fingerprints["synthetic_config"]["sha256"]
    )
    assert first.input_fingerprints["python_source"]["files"] > 20
    assert first.input_fingerprints["python_source"]["sha256"].startswith("sha256:")


def test_experiment_tracks_configured_rlvr_replay_samples(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"sample_id":"replay-1"}\n', encoding="utf-8")
    raw["posttraining"]["rlvr"]["replay_samples"] = str(replay)
    raw["blueprint"] = str(ROOT / "configs" / "sub1b_architecture.yaml")
    raw["synthetic"]["config"] = str(ROOT / "configs" / "synth_data.yaml")
    raw["output_root"] = str(tmp_path / "output")
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = build_experiment_plan(
        experiment_config,
        repo_root=ROOT,
        python=sys.executable,
    )
    rlvr = next(stage for stage in plan.stages if stage.name == "rlvr")

    assert rlvr.command[rlvr.command.index("--replay-samples") + 1] == str(
        replay
    )
    assert plan.input_fingerprints["rlvr_replay_samples"]["bytes"] > 0


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


def test_runner_clears_owned_output_directory_when_signature_changes(tmp_path):
    root = tmp_path / "run"
    output = root / "generated"
    artifact = output / "result.txt"
    command = (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "output = Path(sys.argv[2]); output.mkdir(); "
            "(output / 'result.txt').write_text(sys.argv[3])"
        ),
        "--output",
        str(output),
        "first",
    )
    stage = ExperimentStage(
        "generate",
        command,
        (),
        (Artifact(str(artifact)),),
    )
    first_plan = ExperimentPlan(
        name="test",
        root=str(root),
        blueprint=str(root / "resolved_blueprint.yaml"),
        resolved_blueprint={"schema_version": 1},
        raw_spec={"schema_version": 1},
        components=(),
        stages=(stage,),
        fingerprint="sha256:first",
    )
    ExperimentRunner(first_plan, repo_root=ROOT).run()
    assert artifact.read_text(encoding="utf-8") == "first"

    second_stage = ExperimentStage(
        "generate",
        (*command[:-1], "second"),
        (),
        (Artifact(str(artifact)),),
    )
    second_plan = ExperimentPlan(
        name="test",
        root=str(root),
        blueprint=str(root / "resolved_blueprint.yaml"),
        resolved_blueprint={"schema_version": 1},
        raw_spec={"schema_version": 1},
        components=(),
        stages=(second_stage,),
        fingerprint="sha256:second",
    )
    result = ExperimentRunner(second_plan, repo_root=ROOT).run()

    assert result["outcomes"] == [{"stage": "generate", "status": "completed"}]
    assert artifact.read_text(encoding="utf-8") == "second"
    state = json.loads(
        (root / "state" / "stages" / "generate.json").read_text(encoding="utf-8")
    )
    assert state["invalidated_outputs"] == [str(output)]


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
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    manifest = root / "artifacts" / "initialization_sources" / "vision_checkpoint.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "kind": "huggingface_model_checkpoint",
                "snapshot_path": str(source),
                "files": [
                    {"path": "config.json", "bytes": 2},
                    {"path": "model.safetensors", "bytes": 7},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _resolve_command(("tool", "@checkpoint:vision"), root) == [
        "tool",
        str(source),
    ]
    assert _with_training_resume(command, "pretrain", root, eligible=True) == [
        "tool",
        str(student),
        "--resume",
        "latest",
    ]
    assert _with_training_resume(command, "pretrain", root, eligible=False) == command
