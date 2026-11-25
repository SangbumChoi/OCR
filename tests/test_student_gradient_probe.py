import pytest

from docvlm_eval.student.gradient_probe import GradientConflictProbeConfig


def test_gradient_probe_config_rejects_invalid_schedule_and_components():
    with pytest.raises(ValueError, match="every_steps"):
        GradientConflictProbeConfig(every_steps=0).validate()
    with pytest.raises(ValueError, match="unique"):
        GradientConflictProbeConfig(
            components=("vision", "vision")
        ).validate()
    with pytest.raises(ValueError, match="subset"):
        GradientConflictProbeConfig(
            components=("vision", "unsupported")
        ).validate()


def test_gradient_probe_config_round_trips_through_mapping():
    config = GradientConflictProbeConfig(
        enabled=True,
        every_steps=25,
        components=("vision", "language"),
    )

    assert config.to_dict() == {
        "enabled": True,
        "every_steps": 25,
        "components": ["vision", "language"],
    }
