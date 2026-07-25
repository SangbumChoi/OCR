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
from docvlm_eval.student.synthesis_policy import payload_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_default_experiment_compiles_complete_stage_dag():
    plan = build_experiment_plan(
        ROOT / "configs" / "sub1b_experiment.yaml",
        repo_root=ROOT,
        python=sys.executable,
    )
    assert plan.stage_names == [
        "audit_method_evidence",
        "audit_weight_commonality",
        "visual_backend_benchmark",
        "training_feasibility_benchmark",
        "synthetic_train",
        "synthetic_validation",
        "synthetic_heldout",
        "validate_synthetic_splits",
        "build_synthetic_udd",
        "build_train_samples",
        "build_heldout_samples",
        "build_validation_udd",
        "build_validation_samples",
        "acquire_component_public_udd",
        "mix_pretraining_data",
        "export_teacher_requests",
        "generate_teacher_predictions",
        "apply_teacher_targets",
        "train_tokenizer",
        "audit_generation_budgets",
        "initialize_student",
        "pretrain",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
        "plan_next_synthetic_batch",
    ]
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    method_audit = next(
        stage
        for stage in plan.stages
        if stage.name == "audit_method_evidence"
    )
    assert method_audit.dependencies == ()
    assert method_audit.artifacts[0].path.endswith(
        "artifacts/data/method_evidence.json"
    )
    assert plan.input_fingerprints["frontier_method_catalog"]["sha256"]
    assert plan.input_fingerprints["frontier_method_evidence"]["sha256"]
    assert (
        plan.input_fingerprints["frontier_method_evidence_contract"][
            "status"
        ]
        == "pass"
    )
    weight_audit = next(
        stage
        for stage in plan.stages
        if stage.name == "audit_weight_commonality"
    )
    assert weight_audit.dependencies == ("audit_method_evidence",)
    assert weight_audit.artifacts[0].path.endswith(
        "artifacts/data/weight_commonality_audit.json"
    )
    assert plan.input_fingerprints[
        "small_vlm_architecture_catalog"
    ]["sha256"]
    assert plan.input_fingerprints[
        "small_vlm_weight_commonality"
    ]["sha256"]
    assert (
        plan.input_fingerprints[
            "small_vlm_weight_commonality_contract"
        ]["status"]
        == "pass"
    )
    assert pipeline["balance_by"] == "component"
    assert pipeline["group_weights"] == {
        "synthetic_documents": pytest.approx(0.45),
        "public_udd": pytest.approx(0.55),
    }
    sequence_targets = plan.resolved_blueprint["training"]["pretraining"]["distillation"][
        "sequence_targets"
    ]
    assert sequence_targets["probability"] == pytest.approx(0.5)
    evaluate = next(
        stage for stage in plan.stages if stage.name == "evaluate"
    )
    assert evaluate.command[
        evaluate.command.index("--max-new-tokens-hard-cap") + 1
    ] == "512"
    budget_flags = [
        evaluate.command[index + 1]
        for index, value in enumerate(evaluate.command)
        if value == "--answer-type-token-budget"
    ]
    assert budget_flags == [
        "ocr-full=512",
        "reading-order=384",
        "table*=512",
        "chart*=256",
        "pubtabnet=512",
        "omnidocbench=512",
        "recognition_fullpage=512",
        "im2latex=384",
        "latexocr=384",
        "formula*=384",
        "H-comprehension=256",
        "H-accounting=256",
    ]
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
    audit = next(
        stage
        for stage in plan.stages
        if stage.name == "audit_generation_budgets"
    )
    assert audit.dependencies == (
        "train_tokenizer",
        "build_train_samples",
        "build_heldout_samples",
        "build_validation_samples",
    )
    assert audit.command.count("--split") == 3
    assert audit.command.count("--evaluation-token-budget") == 12
    assert "--calibration-split" in audit.command
    assert audit.artifacts[0].path.endswith(
        "artifacts/data/generation_budget_audit.json"
    )
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
    assert visual_benchmark.dependencies == ("audit_method_evidence",)
    training_benchmark = next(
        stage
        for stage in plan.stages
        if stage.name == "training_feasibility_benchmark"
    )
    assert training_benchmark.dependencies == (
        "audit_method_evidence",
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
    assert "audit_method_evidence" in initialize.dependencies
    assert "audit_weight_commonality" in initialize.dependencies
    assert "audit_generation_budgets" in initialize.dependencies
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    baseline = next(
        stage for stage in plan.stages if stage.name == "evaluate_baseline"
    )
    assert str(
        Path(plan.root) / "artifacts" / "initial"
    ) == baseline.command[baseline.command.index("--checkpoint") + 1]
    assert "--baseline-evaluation" not in baseline.command
    assert baseline.dependencies == (
        "initialize_student",
        "build_train_samples",
        "build_heldout_samples",
        "build_validation_samples",
    )
    baseline_splits = [
        baseline.command[index + 1]
        for index, value in enumerate(baseline.command[:-1])
        if value == "--split"
    ]
    final_splits = [
        evaluate.command[index + 1]
        for index, value in enumerate(evaluate.command[:-1])
        if value == "--split"
    ]
    assert baseline_splits == final_splits
    for flag in (
        "--config",
        "--tokenizer",
        "--precision",
        "--max-new-tokens",
        "--max-new-tokens-hard-cap",
        "--seed",
        "--calibration-source-split",
        "--calibration-fraction",
        "--calibration-min-samples",
        "--calibration-correct-threshold",
        "--calibration-min-temperature",
        "--calibration-max-temperature",
        "--calibration-seed",
    ):
        assert baseline.command[baseline.command.index(flag) + 1] == (
            evaluate.command[evaluate.command.index(flag) + 1]
        )
    assert evaluate.command[
        evaluate.command.index("--baseline-evaluation") + 1
    ].endswith("artifacts/evaluation_baseline")
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


def test_experiment_rejects_unbounded_answer_type_generation_budget(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["max_new_tokens_hard_cap"] = 16
    raw["evaluation"]["max_new_tokens_by_answer_type"] = {
        "table*": 17,
    }
    config = tmp_path / "invalid-generation-budget.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="answer-type generation budgets",
    ):
        build_experiment_plan(
            config,
            repo_root=ROOT,
            python=sys.executable,
        )


def test_experiment_rejects_invalid_generation_budget_audit(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["generation_budget_audit"] = {
        "enabled": True,
        "minimum_coverage": 1.1,
    }
    config = tmp_path / "invalid-generation-budget-audit.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="minimum_coverage"):
        build_experiment_plan(
            config,
            repo_root=ROOT,
            python=sys.executable,
        )


def test_experiment_can_use_a_pretraining_checkpoint_as_internal_baseline(
    tmp_path,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["baseline_checkpoint_stage"] = "pretrain"
    config = tmp_path / "pretrain-baseline.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    baseline = next(
        stage for stage in plan.stages if stage.name == "evaluate_baseline"
    )

    assert "@student:pretrain" in baseline.command
    assert baseline.dependencies[0] == "pretrain"
    assert any(
        artifact.path.endswith("heldout/per_sample.jsonl")
        for artifact in baseline.artifacts
    )


def test_experiment_rejects_ambiguous_or_nonpreceding_baselines(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["evaluation"]["baseline_evaluation"] = "external"
    config = tmp_path / "ambiguous-baseline.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    raw["evaluation"]["baseline_evaluation"] = None
    raw["evaluation"]["baseline_checkpoint_stage"] = "rlvr"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must precede"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    raw["evaluation"]["baseline_checkpoint_stage"] = "inherited"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="requires continuation"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


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
        "evaluate_baseline",
    )


def test_experiment_can_run_preference_stage_from_sft(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["posttraining"]["preference"] = {
        "enabled": True,
        "max_steps": 1,
    }
    raw["posttraining"]["rlvr"] = {
        "enabled": False,
        "max_steps": None,
        "replay_every_steps": None,
        "replay_loss_coefficient": None,
        "replay_samples": None,
    }
    config = tmp_path / "preference.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    assert "preference" in plan.stage_names
    assert "rlvr" not in plan.stage_names
    preference = next(
        stage for stage in plan.stages if stage.name == "preference"
    )
    assert preference.dependencies == ("sft",)
    assert "@student:sft" in preference.command
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert "@student:preference" in evaluate.command
    assert evaluate.dependencies == (
        "preference",
        "build_train_samples",
        "build_heldout_samples",
        "evaluate_baseline",
    )


def test_experiment_runs_preference_then_rlvr_with_frozen_sft_reference(
    tmp_path,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["posttraining"]["preference"]["enabled"] = True
    raw["posttraining"]["preference"]["max_steps"] = 1
    config = tmp_path / "sequential-posttraining.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)

    preference = next(
        stage for stage in plan.stages if stage.name == "preference"
    )
    rlvr = next(stage for stage in plan.stages if stage.name == "rlvr")
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")

    assert preference.dependencies == ("sft",)
    assert rlvr.dependencies == ("preference",)
    assert rlvr.command[rlvr.command.index("--checkpoint") + 1] == (
        "@student:preference"
    )
    assert rlvr.command[
        rlvr.command.index("--reference-checkpoint") + 1
    ] == "@student:sft"
    assert "@student:rlvr" in evaluate.command


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


def test_experiment_rejects_ignored_disabled_preference_overrides(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["posttraining"]["preference"]["max_steps"] = 1
    config = tmp_path / "invalid-disabled-preference.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="disabled preference cannot set"):
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
    assert len(plan.stages) == 17
    assert plan.resolved_blueprint["student"]["vision"]["image_size"] == tiny.vision.image_size
    assert plan.resolved_blueprint["tokenizer"]["vocab_size"] == tiny.language.vocab_size
    assert "visual_backend_benchmark" not in plan.stage_names
    pipeline = plan.resolved_blueprint["training"]["pretraining"]["input_pipeline"]
    assert pipeline["max_image_long_side"] == tiny.vision.image_size
    assert pipeline["max_text_tokens"] == 768
    optimizer_sections = (
        plan.resolved_blueprint["training"]["pretraining"]["optimizer"],
        plan.resolved_blueprint["training"]["posttraining"]["sft"]["optimizer"],
        plan.resolved_blueprint["training"]["posttraining"]["preference"]["optimizer"],
        plan.resolved_blueprint["training"]["posttraining"]["rlvr"]["optimizer"],
    )
    assert all(section["name"] == "adamw" for section in optimizer_sections)
    initialize = next(stage for stage in plan.stages if stage.name == "initialize_student")
    assert initialize.command[initialize.command.index("--tiny-vocab-size") + 1] == "512"
    assert initialize.command[initialize.command.index("--seed") + 1] == "5"
    assert "export_teacher_requests" in plan.stage_names
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    assert any(artifact.path.endswith("gates.json") for artifact in evaluate.artifacts)
    assert "evaluate_baseline" in evaluate.dependencies
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


def test_experiment_compiles_temperature_calibration_contract(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")

    assert "--no-temperature-calibration" not in evaluate.command
    assert evaluate.command[
        evaluate.command.index("--calibration-source-split") + 1
    ] == "validation"
    assert evaluate.command[
        evaluate.command.index("--calibration-min-samples") + 1
    ] == "20"
    assert evaluate.command[
        evaluate.command.index("--calibration-seed") + 1
    ] == "47"


def test_experiment_builds_separate_pretraining_validation_split(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["synthetic"]["validation_count"] = 3
    raw["synthetic"]["validation_seed"] = 3017
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    validation = next(
        stage for stage in plan.stages if stage.name == "synthetic_validation"
    )
    leakage = next(
        stage
        for stage in plan.stages
        if stage.name == "validate_synthetic_splits"
    )
    pretrain = next(stage for stage in plan.stages if stage.name == "pretrain")

    assert validation.command[validation.command.index("--count") + 1] == "3"
    assert validation.command[validation.command.index("--split-name") + 1] == (
        "validation"
    )
    assert "synthetic_validation" in leakage.dependencies
    assert "validation=" in " ".join(leakage.command)
    assert "build_validation_udd" in pretrain.dependencies
    assert "--eval-src" in pretrain.command


def test_experiment_emits_next_batch_plan_from_validation(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["synthetic"]["validation_count"] = 1
    raw["synthetic"]["validation_seed"] = 3017
    raw["synthetic"]["adaptation_policy"] = {
        "enabled": True,
        "config": "configs/sub1b_synthesis_policy.yaml",
        "budget": 5,
        "seed": 99,
    }
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    evaluate = next(stage for stage in plan.stages if stage.name == "evaluate")
    policy = next(
        stage
        for stage in plan.stages
        if stage.name == "plan_next_synthetic_batch"
    )

    assert any(
        argument.endswith("validation.jsonl")
        for argument in evaluate.command
    )
    assert "build_validation_samples" in evaluate.dependencies
    assert policy.dependencies == ("evaluate",)
    assert policy.command[policy.command.index("--budget") + 1] == "5"
    assert policy.command[policy.command.index("--seed") + 1] == "99"
    assert policy.command[
        policy.command.index("--baseline-per-sample") + 1
    ] == str(
        tmp_path
        / "output"
        / "artifacts"
        / "evaluation_baseline"
        / "validation"
        / "per_sample.jsonl"
    )
    baseline = next(
        stage for stage in plan.stages if stage.name == "evaluate_baseline"
    )
    assert any(
        artifact.path.endswith(
            "evaluation_baseline/validation/per_sample.jsonl"
        )
        for artifact in baseline.artifacts
    )


def test_experiment_can_generate_train_split_from_authorized_plan(tmp_path):
    generation_plan = {
        "schema_version": 1,
        "policy": "test",
        "training_authorized": True,
        "source": {
            "split": "validation",
            "path": "/tmp/validation.jsonl",
            "fingerprint": "sha256:test",
            "rows": 1,
        },
        "budget": 1,
        "jobs": [
            {
                "arm_id": "sha256:arm",
                "generator_case": "hard_table",
                "language": "en",
                "difficulty_level": 2,
                "layout_family": "compact-v1",
                "composition_tier": "single_document",
                "count": 1,
                "seed": 7,
                "output_subdir": "job-0000",
            }
        ],
    }
    generation_plan["plan_fingerprint"] = payload_fingerprint(
        generation_plan
    )
    plan_path = tmp_path / "next_train_plan.json"
    plan_path.write_text(
        json.dumps(generation_plan),
        encoding="utf-8",
    )
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    raw["synthetic"]["training_policy_plan"] = str(plan_path)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_experiment_plan(config, repo_root=ROOT, python=sys.executable)
    train = next(
        stage for stage in plan.stages if stage.name == "synthetic_train"
    )

    assert train.command[1].endswith(
        "generate_from_synthesis_policy.py"
    )
    assert train.command[train.command.index("--plan") + 1] == str(plan_path)
    assert "synthetic_training_policy_plan" in plan.input_fingerprints


def test_invalid_experiment_rejects_equal_split_seeds(tmp_path):
    raw = (ROOT / "configs" / "sub1b_experiment.yaml").read_text(encoding="utf-8")
    config = tmp_path / "invalid.yaml"
    config.write_text(raw.replace("heldout_seed: 7007", "heldout_seed: 7"), encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        build_experiment_plan(config, repo_root=ROOT, python=sys.executable)


def test_invalid_experiment_rejects_equal_validation_seed(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["synthetic"]["validation_count"] = 1
    raw["synthetic"]["validation_seed"] = raw["synthetic"]["train_seed"]
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="train, validation, and heldout"):
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


def test_selective_tiny_smoke_compiles_cross_architecture_fixtures():
    plan = build_experiment_plan(
        ROOT / "configs" / "sub1b_experiment_selective_tiny.yaml",
        repo_root=ROOT,
        python=sys.executable,
    )
    assert plan.stage_names[:2] == [
        "build_vision_fixture_checkpoint",
        "build_language_fixture_checkpoint",
    ]
    initialize = next(
        stage
        for stage in plan.stages
        if stage.name == "initialize_student"
    )
    assert initialize.dependencies == (
        "train_tokenizer",
        "build_vision_fixture_checkpoint",
        "build_language_fixture_checkpoint",
    )
    assert (
        plan.input_fingerprints["initialization_vision_fixture"]["spec"][
            "vision_layers"
        ]
        == 3
    )
    assert (
        plan.input_fingerprints["initialization_language_fixture"]["spec"][
            "language_mlp_width"
        ]
        == 320
    )
    assert plan.stage_names[-1] == "plan_next_synthetic_batch"


def test_checkpoint_fixtures_are_restricted_to_tiny_experiments(tmp_path):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_experiment_selective_tiny.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["initialization"]["tiny"] = False
    raw["output_root"] = str(tmp_path / "output")
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixtures require tiny=true"):
        build_experiment_plan(
            path,
            repo_root=ROOT,
            python=sys.executable,
        )


def test_experiment_wires_resume_stable_wandb_runs_to_training_stages(
    tmp_path,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["blueprint"] = str(ROOT / "configs" / "sub1b_architecture.yaml")
    raw["synthetic"]["config"] = str(ROOT / "configs" / "synth_data.yaml")
    raw["output_root"] = str(tmp_path / "output")
    sections = {
        "pretrain": raw["pretraining"],
        "sft": raw["posttraining"]["sft"],
        "rlvr": raw["posttraining"]["rlvr"],
        "evaluate": raw["evaluation"],
    }
    for stage_name, section in sections.items():
        section.update(
            {
                "wandb_project": "docvlm-native",
                "wandb_entity": "sbdc",
                "wandb_run": f"trial--{stage_name}",
                "wandb_group": "trial",
                "wandb_tags": ["native-student", f"stage:{stage_name}"],
            }
        )
    path = tmp_path / "tracked.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    first = build_experiment_plan(path, repo_root=ROOT, python=sys.executable)
    second = build_experiment_plan(path, repo_root=ROOT, python=sys.executable)

    for stage_name in sections:
        first_command = next(
            stage.command
            for stage in first.stages
            if stage.name == stage_name
        )
        second_command = next(
            stage.command
            for stage in second.stages
            if stage.name == stage_name
        )
        assert first_command[first_command.index("--wandb-project") + 1] == (
            "docvlm-native"
        )
        assert first_command[first_command.index("--wandb-group") + 1] == (
            "trial"
        )
        run_id = first_command[first_command.index("--wandb-id") + 1]
        repeated_id = second_command[
            second_command.index("--wandb-id") + 1
        ]
        assert len(run_id) == 32
        assert run_id == repeated_id

    baseline_command = next(
        stage.command
        for stage in first.stages
        if stage.name == "evaluate_baseline"
    )
    baseline_run_id = baseline_command[
        baseline_command.index("--wandb-id") + 1
    ]
    final_run_id = next(
        stage.command
        for stage in first.stages
        if stage.name == "evaluate"
    )
    final_run_id = final_run_id[final_run_id.index("--wandb-id") + 1]
    assert len(baseline_run_id) == 32
    assert baseline_run_id != final_run_id
    assert baseline_command[
        baseline_command.index("--wandb-run") + 1
    ] == "trial--evaluate--baseline"
    assert "checkpoint-stage:initial" in baseline_command


def test_experiment_rejects_invalid_training_wandb_tags(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["blueprint"] = str(ROOT / "configs" / "sub1b_architecture.yaml")
    raw["synthetic"]["config"] = str(ROOT / "configs" / "synth_data.yaml")
    raw["output_root"] = str(tmp_path / "output")
    raw["posttraining"]["rlvr"]["wandb_tags"] = "not-a-list"
    path = tmp_path / "invalid-tracking.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="posttraining.rlvr.wandb_tags",
    ):
        build_experiment_plan(path, repo_root=ROOT, python=sys.executable)


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
    assert resumed["schema_version"] == 2
    assert resumed["pipeline_complete"] is True
    assert [stage["state_status"] for stage in resumed["stages"]] == [
        "completed",
        "completed",
    ]
    assert [stage["invocation_status"] for stage in resumed["stages"]] == [
        "skipped",
        "skipped",
    ]
    assert all(stage["signature_matches"] for stage in resumed["stages"])
    assert all(stage["artifacts_valid"] for stage in resumed["stages"])

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
