import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from docvlm_eval.student.sweep import (
    ParetoObjective,
    ParetoPromotionRule,
    PromotionRule,
    SweepRunner,
    _promotion_decision,
    aggregate_sweep_results,
    apply_json_patch,
    compile_sweep_plan,
)


ROOT = Path(__file__).resolve().parents[1]

QUALITY_PROMOTION_CONTRACTS = {
    "sub1b_adaptive_mixture_sweep.yaml": (
        "heldout_score",
        {"L1-locate", "L1-region", "multilingual"},
    ),
    "sub1b_box_iou_loss_sweep.yaml": (
        "axis.L1-region",
        {"L1-locate", "ocr-full", "multilingual"},
    ),
    "sub1b_contrastive_memory_sweep.yaml": (
        "heldout_score",
        {"L1-region", "ocr-full", "reading-order"},
    ),
    "sub1b_contrastive_objective_sweep.yaml": (
        "heldout_score",
        {"L1-region", "ocr-full", "reading-order"},
    ),
    "sub1b_composition_curriculum_sweep.yaml": (
        "heldout_score",
        {
            "H-comprehension",
            "H-accounting",
            "grounding",
            "multilingual",
            "ocr-full",
        },
    ),
    "sub1b_preference_method_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual"},
    ),
    "sub1b_preference_objective_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual"},
    ),
    "sub1b_preference_source_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual"},
    ),
    "sub1b_pretraining_loss_sweep.yaml": (
        "heldout_score",
        {"L1-region", "ocr-full", "reading-order"},
    ),
    "sub1b_rlvr_advantage_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual"},
    ),
    "sub1b_rlvr_reward_sweep.yaml": (
        "heldout_score",
        {"L1-locate", "L1-region", "multilingual"},
    ),
    "sub1b_sequence_teacher_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual", "ocr-full"},
    ),
    "sub1b_sft_target_sweep.yaml": (
        "axis.L1-region",
        {"L1-locate", "multilingual", "ocr-full"},
    ),
    "sub1b_token_relation_distillation_sweep.yaml": (
        "heldout_score",
        {"L1-region", "multilingual", "ocr-full", "reading-order"},
    ),
}

FULL_PROMOTION_GATES = {
    "parameter_budget",
    "generalization",
    "grounding",
    "reasoning",
    "multilingual",
    "reliability",
}


def _tiny_sweep(tmp_path: Path, *, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sweep_tiny.yaml").read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _compile_tiny(tmp_path: Path):
    return compile_sweep_plan(
        _tiny_sweep(tmp_path),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def test_json_patch_supports_mapping_and_list_operations_without_mutation():
    source = {"model": {"layers": [1, 2], "width": 64}}

    patched = apply_json_patch(
        source,
        [
            {"op": "replace", "path": "/model/width", "value": 96},
            {"op": "add", "path": "/model/layers/-", "value": 3},
            {"op": "remove", "path": "/model/layers/0"},
        ],
    )

    assert patched == {"model": {"layers": [2, 3], "width": 96}}
    assert source == {"model": {"layers": [1, 2], "width": 64}}
    with pytest.raises(ValueError, match="does not exist"):
        apply_json_patch(source, [{"op": "replace", "path": "/missing", "value": 1}])


def test_tiny_sweep_compiles_matched_independent_experiments(tmp_path):
    plan = _compile_tiny(tmp_path)

    assert plan.baseline == "baseline"
    assert plan.replicates == ("seed_0", "seed_1")
    assert plan.promotion is not None
    assert plan.promotion.primary_metric == "heldout_score"
    assert plan.promotion.minimum_replicates == 2
    assert [contrast.to_dict() for contrast in plan.contrasts] == [
        {
            "id": "sequence_distillation_effect",
            "hypothesis": (
                "Measures the paired effect of removing sequence distillation."
            ),
            "coefficients": {
                "baseline": -1.0,
                "no_sequence_distillation": 1.0,
            },
        }
    ]
    assert plan.to_dict()["promotion"]["required_gates"] == [
        "parameter_budget",
        "generalization",
    ]
    assert [(variant.arm_id, variant.replicate_id) for variant in plan.variants] == [
        ("baseline", "seed_0"),
        ("no_sequence_distillation", "seed_0"),
        ("baseline", "seed_1"),
        ("no_sequence_distillation", "seed_1"),
    ]
    baseline = plan.variants[0]
    ablation = plan.variants[1]
    assert baseline.parameters["total"] == 587_019
    assert ablation.parameters["total"] == 587_019
    assert "generate_teacher_predictions" in baseline.plan.stage_names
    assert "generate_teacher_predictions" not in ablation.plan.stage_names
    assert Path(baseline.plan.root) != Path(ablation.plan.root)
    evaluation = baseline.plan.raw_spec["evaluation"]
    assert evaluation["wandb_group"] == "docvlm-tiny-sweep"
    assert evaluation["wandb_run"] == "docvlm-tiny-sweep--baseline--seed_0"
    assert "variant:baseline" in evaluation["wandb_tags"]
    assert "replicate:seed_0" in evaluation["wandb_tags"]
    for stage_name, section in {
        "pretrain": baseline.plan.raw_spec["pretraining"],
        "sft": baseline.plan.raw_spec["posttraining"]["sft"],
        "preference": baseline.plan.raw_spec["posttraining"]["preference"],
        "rlvr": baseline.plan.raw_spec["posttraining"]["rlvr"],
    }.items():
        assert section["wandb_group"] == "docvlm-tiny-sweep"
        assert section["wandb_run"] == (
            "docvlm-tiny-sweep--baseline--seed_0"
            f"--{stage_name}"
        )
        assert f"stage:{stage_name}" in section["wandb_tags"]
        assert "variant:baseline" in section["wandb_tags"]
        assert "replicate:seed_0" in section["wandb_tags"]
    assert (
        plan.control_values_by_replicate["seed_0"][
            "experiment:/pretraining/max_steps"
        ]
        == 1
    )
    assert (
        plan.control_values_by_replicate["seed_1"]["experiment:/initialization/seed"]
        == 105
    )
    initialize = next(
        stage for stage in plan.variants[2].plan.stages if stage.name == "initialize_student"
    )
    assert initialize.command[initialize.command.index("--seed") + 1] == "105"

    other = compile_sweep_plan(
        _tiny_sweep(tmp_path / "second"),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "other-compiled",
    )
    assert [variant.fingerprint for variant in plan.variants] != [
        variant.fingerprint for variant in other.variants
    ]


def test_initialization_sweep_compiles_pinned_transfer_arms(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_initialization_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "initialization-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 15
    assert plan.baseline == "random"
    selective = next(
        variant
        for variant in plan.variants
        if variant.arm_id == "selective" and variant.replicate_id == "seed_0"
    )
    assert selective.plan.raw_spec["initialization"]["arm"] == "I4_selective"
    assert selective.plan.stage_names[:4] == [
        "visual_backend_benchmark",
        "training_feasibility_benchmark",
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
    ]
    initialize = next(
        stage
        for stage in selective.plan.stages
        if stage.name == "initialize_student"
    )
    assert initialize.dependencies == (
        "train_tokenizer",
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
        "visual_backend_benchmark",
        "training_feasibility_benchmark",
    )


def test_attention_geometry_transfer_factorial_compiles_four_cells(tmp_path):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_attention_geometry_transfer_factorial.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "attention-factorial.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 12
    assert len(plan.contrasts) == 5
    assert plan.promotion is None
    expected = {
        "native_random": (799_919_884, 24, 8, 10_000.0, "I0_random"),
        "qwen_random": (781_513_996, 12, 2, 1_000_000.0, "I0_random"),
        "native_strict_transfer": (
            799_919_884,
            24,
            8,
            10_000.0,
            "I6_strict_structured",
        ),
        "qwen_strict_transfer": (
            781_513_996,
            12,
            2,
            1_000_000.0,
            "I6_strict_structured",
        ),
    }
    for variant in plan.variants:
        total, heads, kv_heads, rope_base, arm = expected[
            variant.arm_id
        ]
        language = variant.plan.resolved_blueprint["student"]["language"]
        assert variant.parameters["total"] == total
        assert language["attention_heads"] == heads
        assert language["kv_heads"] == kv_heads
        assert language["rope_base"] == rope_base
        if variant.arm_id.startswith("qwen_"):
            assert language["rope_layout"] == "half_split"
            assert language["attention_bias"] is False
            assert language["mlp_bias"] is False
        assert variant.plan.raw_spec["initialization"]["arm"] == arm
    interaction = next(
        contrast
        for contrast in plan.contrasts
        if contrast.id == "geometry_x_transfer"
    )
    assert interaction.coefficients == {
        "native_random": 1.0,
        "qwen_random": -1.0,
        "native_strict_transfer": -1.0,
        "qwen_strict_transfer": 1.0,
    }


def test_pretraining_loss_sweep_compiles_active_leave_one_out_arms(tmp_path):
    from docvlm_eval.student.pretrain import (
        PretrainConfig,
        pretraining_supervision_contract,
    )

    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_pretraining_loss_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "loss-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 15
    assert plan.baseline == "full_objective"
    expected_removed = {
        "no_autoregressive": "autoregressive",
        "no_region_text_contrastive": "region_text_contrastive",
        "no_box_regression": "box_regression",
        "no_orientation": "orientation",
    }
    for variant in plan.variants:
        training = PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        contract = pretraining_supervision_contract(
            training,
            has_online_teacher=False,
        )
        assert contract["online_teacher_losses"] == []
        assert all(stage["active_losses"] for stage in contract["stages"])
        if variant.arm_id in expected_removed:
            removed = expected_removed[variant.arm_id]
            assert all(
                removed not in stage["active_losses"]
                for stage in contract["stages"]
            )
        assert "loss-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_token_relation_sweep_matches_teacher_and_representation_loss_budget(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_token_relation_distillation_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "relation-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "hidden_anchors"
    by_arm = {
        variant.arm_id: variant
        for variant in plan.variants
        if variant.replicate_id == "seed_0"
    }
    hidden = by_arm["hidden_anchors"].plan.resolved_blueprint[
        "training"
    ]["pretraining"]
    relational = by_arm["token_relations"].plan.resolved_blueprint[
        "training"
    ]["pretraining"]
    assert hidden["losses"]["teacher_kl"] == 0.15
    assert relational["losses"]["teacher_kl"] == 0.15
    assert hidden["losses"]["hidden_feature_distillation"] == 0.1
    assert hidden["losses"]["token_relation_distillation"] == 0.0
    assert relational["losses"]["hidden_feature_distillation"] == 0.0
    assert relational["losses"]["token_relation_distillation"] == 0.1
    assert hidden["distillation"]["relation_max_tokens"] == 0
    assert relational["distillation"]["relation_max_tokens"] == 128
    assert {
        variant.plan.raw_spec["pretraining"]["teacher_checkpoint"]
        for variant in plan.variants
    } == {"artifacts/native_teacher/student"}


def test_contrastive_objective_sweep_compiles_paired_fixed_compute_arms(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_contrastive_objective_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "contrastive-objective-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "siglip"
    for variant in plan.variants:
        heads = variant.plan.resolved_blueprint["student"]["task_heads"]
        assert heads["contrastive_objective"] == variant.arm_id
        assert heads["contrastive_temperature"] == 0.07
        assert heads["contrastive_bias_init"] == -10.0
        memory = variant.plan.resolved_blueprint["training"]["pretraining"][
            "contrastive_memory"
        ]
        assert memory["enabled"] is True
        assert memory["size"] == 1024
        assert variant.parameters["total"] == 799_919_884
        assert "contrastive-objective-ablation" in (
            variant.plan.raw_spec["evaluation"]["wandb_tags"]
        )


def test_contrastive_memory_sweep_compiles_paired_fixed_compute_arms(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_contrastive_memory_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "contrastive-memory-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "memory_1024"
    for variant in plan.variants:
        memory = variant.plan.resolved_blueprint["training"]["pretraining"][
            "contrastive_memory"
        ]
        assert memory["enabled"] is (variant.arm_id == "memory_1024")
        assert memory["size"] == 1024
        assert memory["min_negatives"] == 16
        optimizer = variant.plan.resolved_blueprint["training"][
            "pretraining"
        ]["optimizer"]
        assert optimizer["schedule_unit"] == "student_flops"
        assert optimizer["stop_at_student_flops"] is True
        assert "contrastive-memory-ablation" in (
            variant.plan.raw_spec["evaluation"]["wandb_tags"]
        )


def test_connector_family_sweep_compiles_compute_matched_pareto_arms(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_connector_family_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "connector-family-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "gated_resampler"
    assert plan.promotion is not None
    assert plan.promotion.to_dict()["mode"] == "pareto"
    assert "parameters.total" in {
        objective["metric"]
        for objective in plan.promotion.to_dict()["objectives"]
    }
    parameters = {}
    for variant in plan.variants:
        connector = variant.plan.resolved_blueprint["student"]["connector"]
        assert connector["family"] == variant.arm_id
        parameters.setdefault(
            variant.arm_id,
            variant.parameters["total"],
        )
        optimizer = variant.plan.resolved_blueprint["training"][
            "pretraining"
        ]["optimizer"]
        assert optimizer["stop_at_student_flops"] is True
        assert optimizer["total_student_flops"] == 165669831748966989312
    assert parameters == {
        "gated_resampler": 799_919_884,
        "average_pool_projector": 767_942_922,
    }


def test_box_iou_loss_sweep_compiles_paired_fixed_compute_arms(tmp_path):
    from docvlm_eval.student.pretrain import PretrainConfig

    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_box_iou_loss_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "box-iou-loss-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "giou"
    baseline_by_replicate = {
        variant.replicate_id: PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        for variant in plan.variants
        if variant.arm_id == "giou"
    }
    for variant in plan.variants:
        training = PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        baseline = baseline_by_replicate[variant.replicate_id]
        assert training.box_iou_loss == variant.arm_id
        assert training.loss_weights == baseline.loss_weights
        assert training.curriculum == baseline.curriculum
        assert training.max_steps is None
        assert training.stop_at_total_tokens is False
        assert training.stop_at_student_flops is True
        assert training.schedule_unit == "student_flops"
        assert training.curriculum.unit == "training_compute_fraction"
        assert (
            training.total_student_flops
            == 165_669_831_748_966_989_312
        )
        assert (
            training.warmup_student_flops
            == 828_349_158_744_834_946
        )
        assert "box-iou-loss-ablation" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_adaptive_mixture_sweep_compiles_paired_validation_arms(tmp_path):
    from docvlm_eval.student.pretrain import PretrainConfig

    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_adaptive_mixture_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "adaptive-mixture-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "fixed_uniform"
    expected_step_sizes = {
        "fixed_uniform": None,
        "adaptive_eta025": 0.25,
        "adaptive_eta050": 0.5,
    }
    for variant in plan.variants:
        experiment = variant.plan.raw_spec
        training = PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        assert experiment["synthetic"]["validation_count"] == 100
        assert experiment["pretraining"]["eval_group_by"] == "task"
        assert len(experiment["data"]["components"]) == 1
        assert experiment["data"]["components"][0]["weight"] == 1.0
        assert "synthetic_validation" in variant.plan.stage_names
        assert "build_validation_udd" in variant.plan.stage_names
        if expected_step_sizes[variant.arm_id] is None:
            assert training.adaptive_mixture.enabled is False
        else:
            assert training.adaptive_mixture.enabled is True
            assert training.adaptive_mixture.step_size == expected_step_sizes[
                variant.arm_id
            ]


def test_composition_curriculum_sweep_compiles_paired_arms(tmp_path):
    from docvlm_eval.student.pretrain import PretrainConfig

    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_composition_curriculum_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "composition-curriculum-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "static_final_mix"
    for variant in plan.variants:
        training = PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        schedule = training.composition_curriculum
        assert schedule.fingerprint is not None
        assert (
            len(schedule.stages) == 1
            if variant.arm_id == "static_final_mix"
            else len(schedule.stages) == 3
        )
        assert "composition-curriculum" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_sft_target_sweep_compiles_three_sft_only_targets(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sft_target_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "sft-target-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "evidence_linked"
    expected_modes = {
        "evidence_linked": "evidence_linked",
        "free_rationale": "free_rationale",
        "answer_only": "answer_only",
    }
    for variant in plan.variants:
        assert (
            variant.plan.raw_spec["posttraining"]["sft"]["target_mode"]
            == expected_modes[variant.arm_id]
        )
        assert "rlvr" not in variant.plan.stage_names
        evaluate = next(
            stage for stage in variant.plan.stages if stage.name == "evaluate"
        )
        assert "@student:sft" in evaluate.command
        assert "sft-target-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_rlvr_reward_sweep_compiles_sft_and_reward_controls(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_rlvr_reward_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "rlvr-reward-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 15
    assert plan.baseline == "full_reward"
    for variant in plan.variants:
        evaluate = next(
            stage for stage in variant.plan.stages if stage.name == "evaluate"
        )
        if variant.arm_id == "sft_only":
            assert "rlvr" not in variant.plan.stage_names
            assert "@student:sft" in evaluate.command
        else:
            assert "rlvr" in variant.plan.stage_names
            assert "@student:rlvr" in evaluate.command
        reward_mix = variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]["rlvr"]["reward_mix"]
        rationale_verifier = variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]["rlvr"]["rationale_verifier"]
        if variant.arm_id == "semantic_rationale":
            assert rationale_verifier == "evidence_semantic"
        elif variant.arm_id == "full_reward":
            assert rationale_verifier == "evidence_program_trace"
        if variant.arm_id == "correctness_only":
            assert reward_mix["answer_correctness"] == 1.0
            assert all(
                weight == 0.0
                for name, weight in reward_mix.items()
                if name != "answer_correctness"
            )
        if variant.arm_id == "no_rationale_reward":
            assert reward_mix["grounded_rationale_consistency"] == 0.0
            assert sum(reward_mix.values()) == pytest.approx(1.0)
            assert all(
                weight > 0.0
                for name, weight in reward_mix.items()
                if name != "grounded_rationale_consistency"
            )
        assert "rlvr-reward-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_rlvr_advantage_sweep_compiles_compute_matched_estimators(tmp_path):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_rlvr_advantage_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "rlvr-advantage-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "group_standardized"
    baseline_by_replicate = {
        variant.replicate_id: variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]["rlvr"]
        for variant in plan.variants
        if variant.arm_id == "group_standardized"
    }
    for variant in plan.variants:
        rlvr = variant.plan.resolved_blueprint["training"]["posttraining"][
            "rlvr"
        ]
        baseline_rlvr = baseline_by_replicate[variant.replicate_id]
        expected = (
            "leave_one_out"
            if variant.arm_id == "leave_one_out"
            else "group_standardized"
        )
        assert rlvr["advantage_estimator"] == expected
        assert rlvr["group_size"] == 8
        assert rlvr["reward_mix"] == baseline_rlvr["reward_mix"]
        assert rlvr["rollout"] == baseline_rlvr["rollout"]
        assert rlvr["optimizer"] == baseline_rlvr["optimizer"]
        assert rlvr["optimizer"]["max_steps"] is None
        assert rlvr["optimizer"]["stop_at_student_flops"] is True
        assert (
            rlvr["optimizer"]["total_student_flops"]
            == 192_000_000_000_000_000
        )
        assert "rlvr" in variant.plan.stage_names
        assert "rlvr-advantage-ablation" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_preference_method_sweep_compiles_compute_matched_dpo_and_grpo(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_preference_method_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "preference-method-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "grpo"
    for variant in plan.variants:
        blueprint = variant.plan.resolved_blueprint
        posttraining = blueprint["training"]["posttraining"]
        preference_optimizer = posttraining["preference"]["optimizer"]
        rlvr_optimizer = posttraining["rlvr"]["optimizer"]
        assert posttraining["preference"]["group_size"] == posttraining[
            "rlvr"
        ]["group_size"]
        assert (
            posttraining["preference"]["preference_source"]
            == "reference_verifier_ranked"
        )
        assert posttraining["preference"]["rollout"] == posttraining["rlvr"][
            "rollout"
        ]
        assert preference_optimizer["max_steps"] is None
        assert rlvr_optimizer["max_steps"] is None
        assert preference_optimizer["stop_at_student_flops"] is True
        assert rlvr_optimizer["stop_at_student_flops"] is True
        assert preference_optimizer["total_student_flops"] == (
            rlvr_optimizer["total_student_flops"]
        )
        assert preference_optimizer["total_student_flops"] == (
            192_000_000_000_000_000
        )
        assert preference_optimizer["seed"] == rlvr_optimizer["seed"]
        if variant.arm_id == "dpo":
            assert "preference" in variant.plan.stage_names
            assert "rlvr" not in variant.plan.stage_names
        else:
            assert "rlvr" in variant.plan.stage_names
            assert "preference" not in variant.plan.stage_names
        assert "preference-method-ablation" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_preference_objective_sweep_compiles_compute_matched_dpo_and_ipo(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_preference_objective_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "preference-objective-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "dpo"
    for variant in plan.variants:
        posttraining = variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]
        preference = posttraining["preference"]
        assert preference["objective"] == variant.arm_id
        assert preference["optimizer"]["max_steps"] is None
        assert preference["optimizer"]["stop_at_student_flops"] is True
        assert (
            preference["optimizer"]["total_student_flops"]
            == 192_000_000_000_000_000
        )
        assert "preference" in variant.plan.stage_names
        assert "rlvr" not in variant.plan.stage_names
        assert "preference-objective-ablation" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_preference_source_sweep_compiles_matched_bootstrap_arms(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_preference_source_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "preference-source-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 6
    assert plan.baseline == "reference_only"
    expected_sources = {
        "reference_only": "reference_verifier_ranked",
        "gold_anchored": "gold_anchored_verifier_ranked",
    }
    for variant in plan.variants:
        preference = variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]["preference"]
        assert preference["preference_source"] == expected_sources[
            variant.arm_id
        ]
        assert preference["objective"] == "dpo"
        assert preference["optimizer"]["max_steps"] is None
        assert preference["optimizer"]["stop_at_student_flops"] is True
        assert (
            preference["optimizer"]["total_student_flops"]
            == 192_000_000_000_000_000
        )
        assert "preference" in variant.plan.stage_names
        assert "rlvr" not in variant.plan.stage_names
        assert "preference-source-ablation" in variant.plan.raw_spec[
            "evaluation"
        ]["wandb_tags"]


def test_sequence_teacher_sweep_compiles_pinned_fixed_dose_arms(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sequence_teacher_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "sequence-teacher-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "gold_only"
    expected = {
        "lfm": (
            "lfm2_5-vl-1.6b",
            "919fde3d022e3f90a4716006f993938ee8c2eb97",
        ),
        "qwen": (
            "qwen3_5-0.8b",
            "2fc06364715b967f1860aea9cf38778875588b17",
        ),
    }
    for variant in plan.variants:
        teacher = variant.plan.raw_spec["sequence_teacher"]
        assert teacher["max_requests"] == 4096
        assert teacher["accepted_target_count"] == 400
        assert (
            variant.plan.raw_spec["tokenizer"]["include_teacher_targets"]
            is False
        )
        if variant.arm_id == "gold_only":
            assert "generate_teacher_predictions" not in variant.plan.stage_names
            continue
        assert (teacher["model"], teacher["revision"]) == expected[
            variant.arm_id
        ]
        generate = next(
            stage
            for stage in variant.plan.stages
            if stage.name == "generate_teacher_predictions"
        )
        assert generate.command[generate.command.index("--model-revision") + 1] == (
            teacher["revision"]
        )
        apply = next(
            stage
            for stage in variant.plan.stages
            if stage.name == "apply_teacher_targets"
        )
        assert (
            apply.command[apply.command.index("--accepted-target-count") + 1]
            == "400"
        )


def test_lfm_language_transfer_sweep_compiles_aligned_sub1b_runs(tmp_path):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_lfm_language_transfer_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "lfm-language-transfer-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "native_random"
    assert {variant.arm_id for variant in plan.variants} == {
        "native_random",
        "lfm_random",
        "lfm_strict_transfer",
    }
    for variant in plan.variants:
        initialization = variant.plan.raw_spec["initialization"]
        source = initialization["language_source"]["hub"]
        assert initialization["language_family"] == "lfm2"
        assert source == {
            "repo_id": "LiquidAI/LFM2.5-VL-1.6B",
            "revision": "919fde3d022e3f90a4716006f993938ee8c2eb97",
        }
        if variant.arm_id.startswith("lfm_"):
            assert variant.parameters["total"] == 814_207_243
        else:
            assert variant.parameters["total"] == 799_919_884
        expected_arm = (
            "I8_lfm_aligned_language"
            if variant.arm_id == "lfm_strict_transfer"
            else "I0_random"
        )
        assert initialization["arm"] == expected_arm


def test_visual_canvas_sweep_decomposes_packing_bucketing_and_canvas(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_visual_canvas_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "visual-canvas-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 12
    assert plan.baseline == "packed"
    assert plan.promotion is not None
    assert plan.promotion.to_dict()["mode"] == "pareto"
    assert "efficiency.executed_visual_tokens_per_sample" in {
        objective["metric"]
        for objective in plan.promotion.to_dict()["objectives"]
    }
    policies = {
        variant.arm_id: (
            variant.plan.resolved_blueprint["training"]["pretraining"][
                "input_pipeline"
            ]["visual_sequence_mode"],
            variant.plan.resolved_blueprint["training"]["pretraining"][
                "input_pipeline"
            ]["visual_canvas_mode"],
            variant.plan.resolved_blueprint["training"]["pretraining"][
                "input_pipeline"
            ]["aspect_ratio_bucketing"],
        )
        for variant in plan.variants
    }
    assert policies == {
        "packed": ("packed", "batch_adaptive", False),
        "dense_adaptive_bucketed": ("dense", "batch_adaptive", True),
        "dense_adaptive_unbucketed": ("dense", "batch_adaptive", False),
        "dense_fixed_square": ("dense", "fixed_square", False),
    }
    assert all(
        variant.plan.resolved_blueprint["training"]["pretraining"]["optimizer"][
            "micro_batch_size"
        ]
        == 2
        for variant in plan.variants
    )
    assert all(
        variant.plan.resolved_blueprint["training"]["pretraining"]["optimizer"][
            "grad_accum_steps"
        ]
        == 4
        for variant in plan.variants
    )
    assert all(
        "visual-canvas-ablation"
        in variant.plan.raw_spec["evaluation"]["wandb_tags"]
        for variant in plan.variants
    )
    for variant in plan.variants:
        benchmark_enabled = variant.arm_id == "packed"
        assert (
            variant.plan.raw_spec["runtime"]["visual_backend_benchmark"][
                "require_deployment_gate"
            ]
            is False
        )
        assert (
            variant.plan.raw_spec["runtime"]["visual_backend_benchmark"][
                "enabled"
            ]
            is benchmark_enabled
        )
        assert (
            "visual_backend_benchmark" in variant.plan.stage_names
        ) is benchmark_enabled


def test_sweep_fingerprint_ignores_only_the_temporary_compile_location(tmp_path):
    config = _tiny_sweep(tmp_path)
    first = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-a",
    )
    second = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-b",
    )

    assert first.fingerprint == second.fingerprint
    assert [variant.fingerprint for variant in first.variants] == [
        variant.fingerprint for variant in second.variants
    ]


def test_sweep_without_replicates_keeps_single_run_per_arm_contract(tmp_path):
    def mutate(raw):
        raw.pop("replicates")
        raw.pop("replicate_controls")

    plan = compile_sweep_plan(
        _tiny_sweep(tmp_path, mutate=mutate),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert plan.replicates == ("default",)
    assert [variant.id for variant in plan.variants] == [
        "baseline",
        "no_sequence_distillation",
    ]
    assert plan.variants[0].plan.raw_spec["evaluation"]["wandb_run"] == (
        "docvlm-tiny-sweep--baseline"
    )


def test_sweep_rejects_a_variant_that_changes_a_matched_control(tmp_path):
    def mutate(raw):
        raw["variants"][1]["experiment_patches"].append(
            {"op": "replace", "path": "/evaluation/seed", "value": 99}
        )

    with pytest.raises(ValueError, match="violates matched controls"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path, mutate=mutate),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled",
        )


def test_sweep_rejects_invalid_linear_contrasts(tmp_path):
    def unknown_variant(raw):
        raw["contrasts"][0]["coefficients"] = {
            "baseline": -1.0,
            "missing": 1.0,
        }

    with pytest.raises(ValueError, match="known variants"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "unknown", mutate=unknown_variant),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "unknown-compiled",
        )

    def nonzero_sum(raw):
        raw["contrasts"][0]["coefficients"][
            "no_sequence_distillation"
        ] = 2.0

    with pytest.raises(ValueError, match="sum to zero"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "sum", mutate=nonzero_sum),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "sum-compiled",
        )


def test_sweep_rejects_undeclared_or_incomplete_replicate_dimensions(tmp_path):
    def undeclared(raw):
        raw["replicates"][0]["experiment_patches"].append(
            {"op": "replace", "path": "/synthetic/count", "value": 2}
        )

    with pytest.raises(ValueError, match="undeclared replicate controls"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "undeclared", mutate=undeclared),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled-undeclared",
        )

    def incomplete(raw):
        raw["replicates"][0]["experiment_patches"].pop()

    with pytest.raises(ValueError, match="does not set every replicate control"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "incomplete", mutate=incomplete),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled-incomplete",
        )


def test_sweep_rejects_an_invalid_promotion_contract(tmp_path):
    def mutate(raw):
        raw["promotion"]["familywise_alpha"] = 0.75
        raw["promotion"]["required_axis_deltas"] = {"L1-region": "zero"}

    with pytest.raises(ValueError, match="promotion.familywise_alpha"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path, mutate=mutate),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled",
        )


def test_sweep_rejects_an_empty_axis_primary_metric(tmp_path):
    def mutate(raw):
        raw["promotion"]["primary_metric"] = "axis."

    with pytest.raises(
        ValueError,
        match="promotion.primary_metric axis name must be non-empty",
    ):
        compile_sweep_plan(
            _tiny_sweep(tmp_path, mutate=mutate),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled",
        )


def test_sweep_accepts_a_colon_delimited_axis_primary_metric(tmp_path):
    def mutate(raw):
        raw["promotion"]["primary_metric"] = "axis.probe:direction"

    plan = compile_sweep_plan(
        _tiny_sweep(tmp_path, mutate=mutate),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert plan.promotion is not None
    assert plan.promotion.primary_metric == "axis.probe:direction"


@pytest.mark.parametrize(
    ("config_name", "expected"),
    sorted(QUALITY_PROMOTION_CONTRACTS.items()),
)
def test_quality_sweeps_declare_fail_closed_promotion_contracts(
    config_name,
    expected,
):
    raw = yaml.safe_load(
        (ROOT / "configs" / config_name).read_text(encoding="utf-8")
    )
    promotion = raw["promotion"]
    primary_metric, required_axes = expected

    assert promotion["primary_metric"] == primary_metric
    assert promotion["direction"] == "maximize"
    assert promotion["minimum_effect"] == 0.005
    assert promotion["minimum_replicates"] == 3
    assert promotion["familywise_alpha"] == 0.05
    assert promotion["max_promotions"] == 1
    assert set(promotion["required_gates"]) == FULL_PROMOTION_GATES
    assert set(promotion["required_axis_deltas"]) == required_axes
    assert set(promotion["required_axis_deltas"].values()) == {0.0}


def test_promotion_corrects_all_candidates_and_axis_guardrails():
    plan = SimpleNamespace(
        promotion=PromotionRule(
            primary_metric="axis.L1-region",
            direction="maximize",
            minimum_effect=0.05,
            minimum_replicates=2,
            familywise_alpha=0.05,
            max_promotions=1,
            required_gates=("parameter_budget",),
            required_axis_deltas={"ocr-full": 0.0},
        ),
        baseline="baseline",
        replicates=("seed_0", "seed_1"),
        fingerprint="sha256:test-promotion",
    )
    arm_records = {
        "baseline": {},
        "higher_score_but_regressed": {
            "gate_statuses": {"parameter_budget": "pass"},
            "parameters": {"total": 700},
        },
        "safe_gain": {
            "gate_statuses": {"parameter_budget": "pass"},
            "parameters": {"total": 600},
        },
    }

    def record(replicate, score, target_axis, guardrail_axis):
        return {
            "replicate_id": replicate,
            "delta_vs_baseline": {"heldout_score": score},
            "heldout_axis_delta_vs_baseline": {
                "L1-region": target_axis,
                "ocr-full": guardrail_axis,
            },
        }

    records_by_arm = {
        "higher_score_but_regressed": {
            "seed_0": record("seed_0", 0.3, 0.2, -0.01),
            "seed_1": record("seed_1", 0.3, 0.2, 0.02),
        },
        "safe_gain": {
            "seed_0": record("seed_0", 0.1, 0.1, 0.01),
            "seed_1": record("seed_1", 0.12, 0.12, 0.01),
        },
    }

    decision = _promotion_decision(
        plan,
        arm_records,
        records_by_arm,
    )

    assert decision["status"] == "promote"
    assert decision["selected_variants"] == ["safe_gain"]
    assert decision["candidates"]["higher_score_but_regressed"][
        "decision"
    ] == "reject"
    assert decision["candidates"]["higher_score_but_regressed"][
        "evidence"
    ]["regressed_axes"] == ["ocr-full"]
    assert decision["candidates"]["safe_gain"]["evidence"][
        "primary_metric"
    ] == "axis.L1-region"
    assert decision["multiple_comparisons"]["candidate_count"] == 2
    assert decision["multiple_comparisons"]["comparison_count"] == 4
    assert decision["multiple_comparisons"][
        "per_comparison_alpha"
    ] == pytest.approx(0.0125)


def test_pareto_promotion_filters_dominated_candidates_and_uses_priority():
    plan = SimpleNamespace(
        promotion=ParetoPromotionRule(
            objectives=(
                ParetoObjective(
                    metric="heldout_score",
                    direction="maximize",
                    minimum_effect=-0.02,
                    required_improvement=False,
                ),
                ParetoObjective(
                    metric="parameters.total",
                    direction="minimize",
                    minimum_effect=-50.0,
                    required_improvement=True,
                ),
                ParetoObjective(
                    metric="decision.cache_bytes",
                    direction="minimize",
                    minimum_effect=0.0,
                    required_improvement=True,
                ),
            ),
            selection_order=(
                "heldout_score",
                "decision.cache_bytes",
                "parameters.total",
            ),
            minimum_replicates=2,
            familywise_alpha=0.05,
            max_promotions=1,
            required_gates=("parameter_budget",),
            required_axis_deltas={},
        ),
        baseline="baseline",
        replicates=("seed_0", "seed_1"),
        fingerprint="sha256:test-pareto-promotion",
    )
    candidate_values = {
        "efficient_balanced": (-0.005, -100.0, -50.0, 700),
        "cache_specialist": (0.01, 20.0, -100.0, 820),
        "dominated": (-0.01, -50.0, -20.0, 750),
        "quality_regression": (-0.03, -200.0, -120.0, 600),
    }
    arm_records = {
        "baseline": {},
        **{
            arm_id: {
                "gate_statuses": {"parameter_budget": "pass"},
                "parameters": {"total": parameters},
                "decision_metrics": {
                    "cache_bytes": 1_000.0 + cache_delta
                },
            }
            for arm_id, (
                _,
                _,
                cache_delta,
                parameters,
            ) in candidate_values.items()
        },
    }

    def record(
        replicate,
        score_delta,
        parameter_delta,
        cache_delta,
    ):
        return {
            "replicate_id": replicate,
            "delta_vs_baseline": {"heldout_score": score_delta},
            "parameter_delta_vs_baseline": {
                "total": parameter_delta
            },
            "decision_metric_delta_vs_baseline": {
                "cache_bytes": cache_delta
            },
            "heldout_axis_delta_vs_baseline": {},
        }

    records_by_arm = {
        arm_id: {
            replicate: record(
                replicate,
                score_delta,
                parameter_delta,
                cache_delta,
            )
            for replicate in plan.replicates
        }
        for arm_id, (
            score_delta,
            parameter_delta,
            cache_delta,
            _,
        ) in candidate_values.items()
    }

    decision = _promotion_decision(
        plan,
        arm_records,
        records_by_arm,
    )

    assert decision["status"] == "promote"
    assert decision["pareto_frontier"] == [
        "cache_specialist",
        "efficient_balanced",
    ]
    assert decision["selected_variants"] == ["cache_specialist"]
    assert decision["candidates"]["dominated"]["decision"] == (
        "pareto_dominated"
    )
    assert decision["candidates"]["quality_regression"]["decision"] == (
        "reject"
    )
    assert decision["candidates"]["quality_regression"]["evidence"][
        "regressed_objectives"
    ] == ["heldout_score"]
    assert decision["multiple_comparisons"]["comparison_count"] == 12


def test_full_sweep_compiles_loss_sft_and_reward_ablation_contracts(tmp_path):
    plan = compile_sweep_plan(
        ROOT / "configs" / "sub1b_sweep.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )
    variants = {
        variant.arm_id: variant
        for variant in plan.variants
        if variant.replicate_id == "seed_0"
    }

    assert set(variants) == {
        "baseline",
        "no_sequence_distillation",
        "no_spatial_auxiliary",
        "sft_answer_only",
        "correctness_only_rlvr",
        "no_supervised_replay",
    }
    assert len(plan.variants) == 18
    no_spatial = variants["no_spatial_auxiliary"].plan.resolved_blueprint
    assert no_spatial["training"]["pretraining"]["losses"]["box_regression"] == 0.0
    assert (
        no_spatial["training"]["pretraining"]["curriculum"]["stages"][0][
            "loss_weights"
        ]["box_regression"]
        == 0.0
    )
    assert (
        variants["sft_answer_only"].plan.raw_spec["posttraining"]["sft"]["target_mode"]
        == "answer_only"
    )
    reward_mix = variants["correctness_only_rlvr"].plan.resolved_blueprint["training"][
        "posttraining"
    ]["rlvr"]["reward_mix"]
    assert reward_mix["answer_correctness"] == 1.0
    assert sum(reward_mix.values()) == pytest.approx(1.0)
    no_replay = variants["no_supervised_replay"].plan.resolved_blueprint[
        "training"
    ]["posttraining"]["rlvr"]["supervised_replay"]
    assert no_replay == {"every_steps": 0, "loss_coefficient": 0.0}


def _comparison(score: float, milliseconds: float) -> dict:
    def summary(split: str, split_score: float):
        return {
            "split": split,
            "dataset_size": 2,
            "n_samples": 2,
            "score": split_score,
            "reward": split_score / 2,
            "valid_structure_fraction": 0.75,
            "answer_rate": 1.0,
            "elapsed_seconds": milliseconds * 2 / 1000,
            "milliseconds_per_sample": milliseconds,
            "by_answer_type": {
                "chart": {
                    "n": 2,
                    "score": split_score,
                    "reward": split_score / 2,
                    "valid_structure_fraction": 0.75,
                    "answer_rate": 1.0,
                }
            },
            "by_source": {},
            "by_language": {},
            "by_robustness_axis": {
                "document_family": {
                    "chart": {
                        "n": 2,
                        "score": split_score,
                        "reward": split_score / 2,
                        "valid_structure_fraction": 0.75,
                        "answer_rate": 1.0,
                    }
                }
            },
            "reward_components": {},
        }

    heldout = summary("heldout", score)
    train = summary("train", score + 0.1)
    return {
        "splits": {"train": train, "heldout": heldout},
        "train_minus_heldout": {
            "headline": {
                "score": 0.1,
                "reward": 0.05,
                "valid_structure_fraction": 0.0,
                "answer_rate": 0.0,
            },
            "by_answer_type": {},
            "by_robustness_axis": {
                "document_family": {
                    "chart": {
                        "score": 0.1,
                        "reward": 0.05,
                        "valid_structure_fraction": 0.0,
                        "answer_rate": 0.0,
                    }
                }
            },
        },
    }


def test_sweep_aggregates_baseline_deltas_ranking_and_markdown(tmp_path):
    plan = _compile_tiny(tmp_path)
    scores = {
        ("baseline", "seed_0"): (0.2, 20.0),
        ("baseline", "seed_1"): (0.3, 22.0),
        ("no_sequence_distillation", "seed_0"): (0.35, 24.0),
        ("no_sequence_distillation", "seed_1"): (0.4, 25.0),
    }
    for variant in plan.variants:
        samples = Path(variant.plan.root) / "artifacts" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        image = samples / "document.png"
        image.write_bytes(f"image-{variant.replicate_id}".encode())
        row = {
            "sample_id": "sample-1",
            "image_path": str(image),
            "question": "Read the chart.",
            "answers": ["42"],
            "answer_type": "chart",
            "metric": "relaxed_acc",
            "meta": {"language": "en"},
        }
        for split in ("train", "heldout"):
            (samples / f"{split}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
        pretrain = Path(variant.plan.root) / "artifacts" / "pretrain"
        checkpoint = pretrain / "checkpoints" / "step-00000001"
        checkpoint.mkdir(parents=True)
        dense_tokens = (
            100 if variant.arm_id == "baseline" else 60
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "student_flops_seen": dense_tokens * 1_000,
                    "dense_visual_tokens_seen": dense_tokens,
                    "valid_visual_tokens_seen": 50,
                    "visual_samples_seen": 2,
                    "visual_attention_backend": "loop",
                }
            ),
            encoding="utf-8",
        )
        (pretrain / "latest_checkpoint.txt").write_text(
            str(checkpoint),
            encoding="utf-8",
        )
        path = Path(variant.plan.root) / "artifacts" / "evaluation" / "comparison.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        heldout_score, milliseconds = scores[
            (variant.arm_id, variant.replicate_id)
        ]
        path.write_text(
            json.dumps(_comparison(heldout_score, milliseconds)),
            encoding="utf-8",
        )
        for split, split_score in (
            ("train", heldout_score + 0.1),
            ("heldout", heldout_score),
        ):
            split_root = path.parent / split
            split_root.mkdir()
            eval_row = {
                "sample_id": "sample-1",
                "score": split_score,
                "answer": "42",
                "answer_type": "chart",
                "confidence": 0.8,
                "meta": {},
                "reward_components": {},
                "applicable_rewards": [],
            }
            (split_root / "per_sample.jsonl").write_text(
                json.dumps(eval_row) + "\n",
                encoding="utf-8",
            )

    result = aggregate_sweep_results(plan)

    assert result["ranking"] == ["no_sequence_distillation", "baseline"]
    contrast = result["contrasts"]["sequence_distillation_effect"]
    assert contrast["metrics"]["heldout_score"] == pytest.approx(0.125)
    assert contrast["metric_statistics"]["heldout_score"]["ci95"] == pytest.approx(
        [0.1, 0.15]
    )
    assert contrast["heldout_score_conclusion"] == "improved"
    assert contrast["parameter_contrast"]["total"] == 0.0
    assert result["variants"]["no_sequence_distillation"]["delta_vs_baseline"][
        "heldout_score"
    ] == pytest.approx(0.125)
    paired = result["variants"]["no_sequence_distillation"][
        "paired_delta_statistics"
    ]["heldout_score"]
    assert paired["n"] == 2
    assert paired["ci95"] == pytest.approx([0.1, 0.15])
    assert (
        result["variants"]["no_sequence_distillation"]["heldout_score_conclusion"]
        == "improved"
    )
    assert set(result["matched_evaluation_artifacts_by_replicate"]) == {
        "seed_0",
        "seed_1",
    }
    assert result["replicate_count"] == 2
    candidate = result["variants"]["no_sequence_distillation"]
    assert candidate["gate_status"] == "insufficient_evidence"
    assert candidate["gate_statuses"]["parameter_budget"] == "pass"
    assert candidate["gate_statuses"]["generalization"] == "pass"
    assert (
        result["variants"]["baseline"]["gate_statuses"]["generalization"]
        == "insufficient_evidence"
    )
    promotion = result["promotion"]
    assert promotion["status"] == "promote"
    assert promotion["selected_variants"] == [
        "no_sequence_distillation"
    ]
    candidate_promotion = promotion["candidates"][
        "no_sequence_distillation"
    ]
    assert candidate_promotion["decision"] == "promote"
    assert candidate_promotion["evidence"][
        "simultaneous_lower_bound"
    ] == pytest.approx(0.1)
    assert promotion["multiple_comparisons"]["candidate_count"] == 1
    assert promotion["multiple_comparisons"]["comparison_count"] == 1
    assert promotion["multiple_comparisons"][
        "simultaneous_confidence_level"
    ] == pytest.approx(0.95)
    robustness = candidate["heldout_robustness_delta_statistics"][
        "document_family"
    ]["chart"]
    assert robustness["n"] == 2
    assert robustness["mean"] == pytest.approx(0.125)
    assert robustness["ci95"] == pytest.approx([0.1, 0.15])
    efficiency = candidate["pretraining_efficiency_delta_statistics"]
    assert efficiency["dense_visual_tokens_per_sample"]["mean"] == -20.0
    assert efficiency["student_flops"]["mean"] == -40_000.0
    assert all(
        record["pretraining_visual_attention_backend"] == "loop"
        for record in result["runs"].values()
    )
    assert Path(candidate["gate_report"]).is_file()
    assert all(Path(record["gate_report"]).is_file() for record in result["runs"].values())
    assert (Path(plan.root) / "comparison.json").is_file()
    markdown = (Path(plan.root) / "comparison.md").read_text(encoding="utf-8")
    assert "Paired delta [95% CI]" in markdown
    assert "Gates" in markdown
    assert "Promotion decision" in markdown
    assert "Paired linear contrasts" in markdown
    assert "`sequence_distillation_effect`" in markdown
    assert "`promote`" in markdown
    assert "`no_sequence_distillation`" in markdown


def test_sweep_rejects_mismatched_evaluation_artifacts(tmp_path):
    plan = _compile_tiny(tmp_path)
    for index, variant in enumerate(plan.variants):
        samples = Path(variant.plan.root) / "artifacts" / "samples"
        evaluation = Path(variant.plan.root) / "artifacts" / "evaluation"
        samples.mkdir(parents=True, exist_ok=True)
        evaluation.mkdir(parents=True, exist_ok=True)
        image = samples / "document.png"
        image.write_bytes(f"image-{index}".encode())
        row = {
            "sample_id": "sample-1",
            "image_path": str(image),
            "question": "Read.",
            "answers": ["42"],
            "answer_type": "chart",
            "metric": "relaxed_acc",
            "meta": {},
        }
        for split in ("train", "heldout"):
            (samples / f"{split}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
        (evaluation / "comparison.json").write_text(
            json.dumps(_comparison(0.1, 1.0)),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="mismatched evaluation artifacts"):
        aggregate_sweep_results(plan)


def test_sweep_runner_dry_run_does_not_create_the_output_root(tmp_path):
    plan = _compile_tiny(tmp_path)
    output = Path(plan.root)
    assert not output.exists()

    result = SweepRunner(plan, repo_root=ROOT).run(dry_run=True)

    assert result["dry_run"] is True
    assert len(result["variants"]) == 4
    assert not output.exists()


def test_sweep_runner_records_each_variant_once_and_updates_suite_state(
    tmp_path,
    monkeypatch,
):
    import docvlm_eval.student.sweep as sweep_module

    plan = _compile_tiny(tmp_path)
    calls = []

    class FakeExperimentRunner:
        def __init__(self, experiment_plan, *, repo_root):
            del repo_root
            self.plan = experiment_plan

        def run(self, **kwargs):
            calls.append((self.plan.name, kwargs))
            return {"dry_run": False, "outcomes": []}

    monkeypatch.setattr(sweep_module, "ExperimentRunner", FakeExperimentRunner)
    result = SweepRunner(plan, repo_root=ROOT).run(stop="pretrain")

    assert len(calls) == 4
    assert result["status"] == "completed"
    assert [item["status"] for item in result["variants"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    persisted = json.loads(
        (Path(plan.root) / "sweep_run_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"


def test_sweep_runner_filters_by_arm_and_replicate(tmp_path, monkeypatch):
    import docvlm_eval.student.sweep as sweep_module

    plan = _compile_tiny(tmp_path)
    calls = []

    class FakeExperimentRunner:
        def __init__(self, experiment_plan, *, repo_root):
            del repo_root
            self.plan = experiment_plan

        def run(self, **kwargs):
            calls.append((self.plan.name, kwargs))
            return {"dry_run": True, "outcomes": []}

    monkeypatch.setattr(sweep_module, "ExperimentRunner", FakeExperimentRunner)
    result = SweepRunner(plan, repo_root=ROOT).run(
        dry_run=True,
        variant_ids={"baseline"},
        replicate_ids={"seed_1"},
    )

    assert len(calls) == 1
    assert result["variants"][0]["run"] == "baseline--seed_1"
