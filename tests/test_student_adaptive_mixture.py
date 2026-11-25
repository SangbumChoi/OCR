import copy
import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="adaptive mixture tests require torch",
)


def _config(**overrides):
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureConfig

    values = {
        "enabled": True,
        "step_size": 0.5,
        "ema_decay": 0.0,
        "min_probability": 0.02,
        "warmup_evaluations": 0,
    }
    values.update(overrides)
    return AdaptiveMixtureConfig(**values)


def test_harder_heldout_group_receives_more_probability():
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureController

    controller = AdaptiveMixtureController(
        _config(),
        {"easy": 1.0, "hard": 1.0},
    )
    controller.observe({"easy": 1.0, "hard": 2.0})

    assert controller.apply_pending() is True
    assert controller.weights["hard"] > controller.weights["easy"]
    assert sum(controller.weights.values()) == pytest.approx(1.0)
    assert controller.updates == 1


def test_probability_floor_and_warmup_are_enforced():
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureController

    controller = AdaptiveMixtureController(
        _config(
            step_size=100.0,
            min_probability=0.1,
            warmup_evaluations=1,
        ),
        {"easy": 1.0, "medium": 1.0, "hard": 1.0},
    )
    controller.observe({"easy": 0.0, "medium": 0.0, "hard": 10.0})

    assert controller.apply_pending() is False
    assert controller.weights == pytest.approx(
        {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}
    )

    controller.observe({"easy": 0.0, "medium": 0.0, "hard": 10.0})

    assert controller.apply_pending() is True
    assert min(controller.weights.values()) >= 0.1
    assert sum(controller.weights.values()) == pytest.approx(1.0)


def test_controller_rejects_invalid_groups_losses_and_floor():
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureController

    with pytest.raises(ValueError, match="probability mass"):
        AdaptiveMixtureController(
            _config(min_probability=0.5),
            {"easy": 1.0, "hard": 1.0},
        )

    controller = AdaptiveMixtureController(
        _config(),
        {"easy": 1.0, "hard": 1.0},
    )
    with pytest.raises(ValueError, match="do not match"):
        controller.observe({"easy": 1.0})
    with pytest.raises(ValueError, match="finite"):
        controller.observe({"easy": 1.0, "hard": float("nan")})


def test_controller_checkpoint_round_trip_preserves_pending_update():
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureController

    original = AdaptiveMixtureController(
        _config(ema_decay=0.5),
        {"easy": 3.0, "hard": 1.0},
    )
    original.observe({"easy": 1.0, "hard": 3.0})
    restored = AdaptiveMixtureController(
        _config(ema_decay=0.5),
        {"easy": 3.0, "hard": 1.0},
    )

    restored.load_state_dict(copy.deepcopy(original.state_dict()))

    assert restored.state_dict() == original.state_dict()
    assert restored.apply_pending() is True
    assert original.apply_pending() is True
    assert restored.weights == pytest.approx(original.weights)


def test_controller_rejects_corrupt_checkpoint_state():
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureController

    controller = AdaptiveMixtureController(
        _config(),
        {"easy": 1.0, "hard": 1.0},
    )
    state = controller.state_dict()
    state["ema_losses"] = {"easy": 1.0, "hard": float("inf")}
    state["evaluations"] = 1

    with pytest.raises(ValueError, match="EMA losses"):
        controller.load_state_dict(state)
