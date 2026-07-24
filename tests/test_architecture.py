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
    sequence_targets = blueprint["training"]["pretraining"]["distillation"]["sequence_targets"]
    sequence_targets["probability"] = 1.1
    sequence_targets["min_score"] = -0.1
    sequence_targets["seed"] = -1
    blueprint["tokenizer"]["vocab_size"] = 32000

    _, errors = validate_blueprint(blueprint)

    assert any("rotation_probability must be between" in error for error in errors)
    assert any("balance_by must be task, source, language, or component" in error for error in errors)
    assert any("visual_canvas_mode" in error for error in errors)
    assert any("visual_sequence_mode" in error for error in errors)
    assert any("packed_attention_backend" in error for error in errors)
    assert any("aspect_ratio_bucketing must be boolean" in error for error in errors)
    assert any("aspect_ratio_bucket_log2_step" in error for error in errors)
    assert any("max_position_tokens" in error for error in errors)
    assert any("sequence_targets.probability" in error for error in errors)
    assert any("sequence_targets.min_score" in error for error in errors)
    assert any("sequence_targets.seed" in error for error in errors)
    assert any("tokenizer.vocab_size must match" in error for error in errors)


def test_packed_visual_sequences_reject_redundant_aspect_bucketing():
    from docvlm_eval.architecture import load_blueprint, validate_blueprint

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    blueprint["training"]["pretraining"]["input_pipeline"][
        "aspect_ratio_bucketing"
    ] = True

    _, errors = validate_blueprint(blueprint)

    assert any("false for packed visual sequences" in error for error in errors)


def test_blueprint_rejects_an_unimplemented_pretraining_loss():
    blueprint = deepcopy(load_blueprint(CONFIG))
    blueprint["training"]["pretraining"]["losses"]["future_objective"] = 0.1

    _, errors = validate_blueprint(blueprint)

    assert any("future_objective is not implemented" in error for error in errors)


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
    rlvr = blueprint["training"]["posttraining"]["rlvr"]
    rlvr["group_size"] = 1
    rlvr["rollout"]["top_p"] = 2.0
    rlvr["supervised_replay"]["every_steps"] = 0
    rlvr["reward_mix"]["unsupported_reward"] = 0.0

    _, errors = validate_blueprint(blueprint)

    assert any("sft.target_mode is invalid" in error for error in errors)
    assert any("group_size must be at least two" in error for error in errors)
    assert any("rollout.top_p must be within" in error for error in errors)
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
