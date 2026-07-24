from copy import deepcopy
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sub1b_architecture.yaml"


def test_default_blueprint_is_valid_and_sub1b():
    blueprint = load_blueprint(CONFIG)
    estimates, errors = validate_blueprint(blueprint)
    assert errors == []
    assert 700_000_000 < estimates["total"] < 1_000_000_000
    assert estimates["language"] > estimates["vision"]
    assert blueprint["training"]["posttraining"]["rlvr"][
        "supervised_replay"
    ] == {"every_steps": 20, "loss_coefficient": 0.10}


def test_blueprint_language_mixer_fields_are_backward_compatible():
    blueprint = deepcopy(load_blueprint(CONFIG))
    language = blueprint["student"]["language"]
    language.pop("full_attention_layers")
    language.pop("conv_kernel_size")
    language.pop("conv_bias")

    estimates, errors = validate_blueprint(blueprint)

    assert errors == []
    assert estimates["total"] == 799_919_884


def test_blueprint_rejects_non_integer_attention_layer_indices():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["student"]["language"]["full_attention_layers"] = [
        2,
        "5",
    ]

    _, errors = validate_blueprint(blueprint)

    assert any("must contain integers" in error for error in errors)


def test_blueprint_rejects_invalid_mixture_and_transfer_fraction():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["posttraining"]["rlvr"]["reward_mix"]["box_iou"] = 0.5
    blueprint["initialization_arms"][0]["vision_transfer"] = 1.5

    _, errors = validate_blueprint(blueprint)

    assert any("reward_mix weights sum" in error for error in errors)
    assert any("vision_transfer must be between" in error for error in errors)


def test_blueprint_rejects_invalid_input_pipeline_controls():
    blueprint = deepcopy(load_blueprint(CONFIG))
    pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
    pipeline["rotation_probability"] = 2.0
    pipeline["balance_by"] = "anything"
    pipeline["visual_canvas_mode"] = "implicit-crop"
    pipeline["visual_sequence_mode"] = "flat"
    pipeline["packed_attention_backend"] = "flash"
    pipeline["aspect_ratio_bucketing"] = "yes"
    pipeline["aspect_ratio_bucket_log2_step"] = 0.0
    blueprint["student"]["vision"]["max_position_tokens"] = 4095
    language = blueprint["student"]["language"]
    language["full_attention_layers"] = [5, 5, 99]
    language["conv_kernel_size"] = 1
    language["conv_bias"] = "no"
    sequence_targets = blueprint["training"]["pretraining"]["distillation"]["sequence_targets"]
    sequence_targets["probability"] = 1.1
    sequence_targets["min_score"] = -0.1
    sequence_targets["seed"] = -1
    blueprint["tokenizer"]["vocab_size"] = 32000
    blueprint["student"]["task_heads"]["contrastive_objective"] = "cosine"
    blueprint["student"]["task_heads"]["contrastive_temperature"] = 0.0
    blueprint["student"]["task_heads"]["contrastive_bias_init"] = float("nan")
    blueprint["student"]["connector"]["family"] = "unknown"

    _, errors = validate_blueprint(blueprint)

    assert any("rotation_probability must be between" in error for error in errors)
    assert any("balance_by must be task, source, language, or component" in error for error in errors)
    assert any("visual_canvas_mode" in error for error in errors)
    assert any("visual_sequence_mode" in error for error in errors)
    assert any("packed_attention_backend" in error for error in errors)
    assert any("aspect_ratio_bucketing must be boolean" in error for error in errors)
    assert any("aspect_ratio_bucket_log2_step" in error for error in errors)
    assert any("max_position_tokens" in error for error in errors)
    assert any("full_attention_layers must be unique" in error for error in errors)
    assert any("out-of-range index" in error for error in errors)
    assert any("conv_kernel_size" in error for error in errors)
    assert any("conv_bias must be a boolean" in error for error in errors)
    assert any("sequence_targets.probability" in error for error in errors)
    assert any("sequence_targets.min_score" in error for error in errors)
    assert any("sequence_targets.seed" in error for error in errors)
    assert any("tokenizer.vocab_size must match" in error for error in errors)
    assert any("contrastive_objective" in error for error in errors)
    assert any("contrastive_temperature" in error for error in errors)
    assert any("contrastive_bias_init" in error for error in errors)
    assert any("connector.family" in error for error in errors)


def test_packed_visual_sequences_reject_redundant_aspect_bucketing():
    from docvlm_eval.architecture import load_blueprint, validate_blueprint

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    blueprint["training"]["pretraining"]["input_pipeline"][
        "aspect_ratio_bucketing"
    ] = True

    _, errors = validate_blueprint(blueprint)

    assert any("false for packed visual sequences" in error for error in errors)


def test_blueprint_rejects_invalid_adaptive_mixture_contract():
    blueprint = deepcopy(load_blueprint(CONFIG))
    pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
    pipeline["adaptive_mixture"] = {
        "enabled": True,
        "step_size": float("nan"),
        "ema_decay": 1.0,
        "min_probability": -0.1,
        "warmup_evaluations": 1.5,
    }
    blueprint["training"]["pretraining"]["optimizer"]["eval_every_steps"] = 0
    blueprint["training"]["pretraining"]["curriculum"]["stages"][0][
        "group_weights"
    ] = {"vqa": 1.0}

    _, errors = validate_blueprint(blueprint)

    assert any("step_size must be a finite number" in error for error in errors)
    assert any("ema_decay must be within" in error for error in errors)
    assert any("min_probability must be within" in error for error in errors)
    assert any("warmup_evaluations" in error for error in errors)
    assert any("positive pretraining eval_every_steps" in error for error in errors)
    assert any(
        "adaptive mixture cannot be combined" in error for error in errors
    )


def test_blueprint_rejects_invalid_gradient_probe_contract():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["pretraining"]["gradient_conflict_probe"] = {
        "enabled": "yes",
        "every_steps": 0,
        "components": ["vision", "vision"],
    }

    _, errors = validate_blueprint(blueprint)

    assert any("gradient_conflict_probe.enabled" in error for error in errors)
    assert any("gradient_conflict_probe.every_steps" in error for error in errors)
    assert any("gradient_conflict_probe.components" in error for error in errors)


def test_blueprint_rejects_invalid_contrastive_memory_contract():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["pretraining"]["contrastive_memory"] = {
        "enabled": True,
        "size": 0,
        "min_negatives": 2,
        "scope": "global",
    }
    blueprint["student"]["task_heads"]["region_text_contrastive"] = False
    blueprint["training"]["pretraining"]["input_pipeline"][
        "contrastive"
    ] = False

    _, errors = validate_blueprint(blueprint)

    assert any("contrastive_memory.size" in error for error in errors)
    assert any("contrastive_memory.min_negatives" in error for error in errors)
    assert any("contrastive_memory.scope" in error for error in errors)
    assert any(
        "requires the region-text contrastive head" in error
        for error in errors
    )
    assert any(
        "requires contrastive input batches" in error
        for error in errors
    )


def test_blueprint_rejects_an_unimplemented_pretraining_loss():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["pretraining"]["losses"]["future_objective"] = 0.1
    blueprint["training"]["pretraining"]["box_iou_loss"] = "plain_iou"

    _, errors = validate_blueprint(blueprint)

    assert any("future_objective is not implemented" in error for error in errors)
    assert any("box_iou_loss" in error for error in errors)


def test_blueprint_rejects_invalid_curriculum_contracts():
    blueprint = deepcopy(load_blueprint(CONFIG))
    stages = blueprint["training"]["pretraining"]["curriculum"]["stages"]
    stages[1]["id"] = stages[0]["id"]
    stages[1]["until_fraction"] = 0.1
    stages[-1]["until_fraction"] = 0.9
    stages[-1]["loss_weights"]["future_objective"] = 0.1

    _, errors = validate_blueprint(blueprint)

    assert any("id must be non-empty and unique" in error for error in errors)
    assert any("until_fraction must increase" in error for error in errors)
    assert any("unsupported losses" in error for error in errors)
    assert any("final stage must end at 1.0" in error for error in errors)


def test_blueprint_rejects_inconsistent_token_budget_contract():
    blueprint = deepcopy(load_blueprint(CONFIG))
    optimizer = blueprint["training"]["pretraining"]["optimizer"]
    optimizer["stop_at_total_tokens"] = False
    optimizer["token_unit"] = "pixels"
    blueprint["training"]["pretraining"]["curriculum"]["stages"][0][
        "group_weights"
    ] = {"vqa": 1.0}

    _, errors = validate_blueprint(blueprint)

    assert any("token_unit must be supervised, text, or effective" in error for error in errors)
    assert any(
        "training_token_fraction curriculum requires stop_at_total_tokens" in error
        for error in errors
    )
    assert any("cannot override sampler group weights" in error for error in errors)


def test_blueprint_rejects_invalid_posttraining_contracts():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["posttraining"]["sft"]["target_mode"] = "hidden_reasoning"
    preference = blueprint["training"]["posttraining"]["preference"]
    preference["group_size"] = 1
    preference["dpo_beta"] = 0.0
    preference["ipo_tau"] = 0.0
    preference["sequence_reduction"] = "median"
    preference["rollout"]["use_kv_cache"] = "yes"
    rlvr = blueprint["training"]["posttraining"]["rlvr"]
    rlvr["group_size"] = 1
    rlvr["advantage_estimator"] = "critic"
    rlvr["rollout"]["top_p"] = 2.0
    rlvr["rollout"]["use_kv_cache"] = "yes"
    rlvr["supervised_replay"]["every_steps"] = 0
    rlvr["reward_mix"]["unsupported_reward"] = 0.0

    _, errors = validate_blueprint(blueprint)

    assert any("sft.target_mode is invalid" in error for error in errors)
    assert any("group_size must be at least two" in error for error in errors)
    assert any("preference.dpo_beta must be positive" in error for error in errors)
    assert any("preference.ipo_tau must be positive" in error for error in errors)
    assert any("preference.sequence_reduction" in error for error in errors)
    assert any(
        "preference.rollout.use_kv_cache must be a boolean" in error
        for error in errors
    )
    assert any("advantage_estimator" in error for error in errors)
    assert any("rollout.top_p must be within" in error for error in errors)
    assert any("rollout.use_kv_cache must be a boolean" in error for error in errors)
    assert any("interval and coefficient" in error for error in errors)
    assert any("unsupported_reward" in error for error in errors)


def test_blueprint_rejects_invalid_visual_efficiency_gate():
    blueprint = deepcopy(load_blueprint(CONFIG))
    gate = next(
        gate
        for gate in blueprint["evaluation_gates"]
        if gate["id"] == "visual_efficiency"
    )
    gate["candidate_requested_backend"] = "loop"
    gate["required_device_type"] = "cpu"
    gate["dense_control_requested_backend"] = "dense_fixed_square"
    gate["min_measured_iterations"] = 0
    gate["min_rounds"] = 0
    gate["min_median_speedup_vs_loop"] = 0.0

    _, errors = validate_blueprint(blueprint)

    assert any("candidate_requested_backend" in error for error in errors)
    assert any("required_device_type" in error for error in errors)
    assert any("dense_control_requested_backend" in error for error in errors)
    assert any("min_measured_iterations" in error for error in errors)
    assert any("min_rounds" in error for error in errors)
    assert any("min_median_speedup_vs_loop" in error for error in errors)
