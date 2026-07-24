import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_lora_placement_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_lora_placement_sweep",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
aggregate_results = MODULE.aggregate_results


def _config():
    return yaml.safe_load(
        (ROOT / "configs" / "lora_vision_connector_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )


def _summary(grounding, locate, *, guard=0.8, score=0.5):
    return {
        "score": score,
        "by_answer_type": {
            "grounding": {"score": grounding},
            "L1-locate": {"score": locate},
            "kie": {"score": guard},
            "ocr-full": {"score": guard},
            "multilingual": {"score": guard},
            "reading-order": {"score": guard},
        },
    }


def _results():
    config = _config()
    rows = {}
    for index, replicate in enumerate(config["replicates"]):
        replicate_id = replicate["id"]
        for placement in config["placements"]:
            combined = placement == "vision_connector"
            heldout = _summary(
                0.20 + index * 0.01 + (0.03 if combined else 0),
                0.15 + index * 0.01 + (0.02 if combined else 0),
                guard=0.80 - (0.005 if combined else 0),
                score=0.50 + (0.01 if combined else 0),
            )
            rows[f"{config['name']}:{placement}:{replicate_id}"] = {
                "control": {
                    "lora_budget": {
                        "realized_relative_budget_error": (
                            0.02 if combined else 0.0
                        )
                    }
                },
                "probes": {"heldout": heldout},
                "training_eval": {
                    "train": {"score": 0.60 + (0.005 if combined else 0)},
                    "heldout": {"score": 0.50 + (0.01 if combined else 0)},
                },
            }
    return {"models": {config["model"]: rows}}


def test_analysis_promotes_consistent_budget_matched_gain():
    result = aggregate_results(_config(), _results())

    assert result["decision"] == "promote"
    assert result["metric_statistics"]["grounding"]["mean"] == pytest.approx(
        0.03
    )
    assert result["metric_statistics"]["L1-locate"]["mean"] == pytest.approx(
        0.02
    )
    assert result["gates"] == {
        "primary_direction": True,
        "guard_noninferiority": True,
        "adapter_budget": True,
        "generalization_gap": True,
    }


def test_analysis_fails_closed_on_missing_pair():
    results = _results()
    config = _config()
    key = f"{config['name']}:vision_connector:seed_2"
    del results["models"][config["model"]][key]

    with pytest.raises(ValueError, match="missing completed result"):
        aggregate_results(config, results)
