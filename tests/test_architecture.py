from copy import deepcopy
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sub1b_architecture.yaml"


def test_default_blueprint_is_valid_and_sub1b():
    estimates, errors = validate_blueprint(load_blueprint(CONFIG))
    assert errors == []
    assert 700_000_000 < estimates["total"] < 1_000_000_000
    assert estimates["language"] > estimates["vision"]


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
    blueprint["tokenizer"]["vocab_size"] = 32000

    _, errors = validate_blueprint(blueprint)

    assert any("rotation_probability must be between" in error for error in errors)
    assert any("balance_by must be task, source, or language" in error for error in errors)
    assert any("tokenizer.vocab_size must match" in error for error in errors)
