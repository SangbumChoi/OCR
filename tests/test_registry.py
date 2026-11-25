"""Model registry: every adapter registers, builds, and exposes a sane profile.

Adapters import torch/transformers only inside load()/generate(), so building them (and
reading metadata) works on CPU with no heavy deps installed.
"""

import pytest

from docvlm_eval.models import build_model, list_models

EXPECTED = {
    "internvl2_5-1b", "internvl3-1b", "internvl2-1b", "smolvlm-256m", "smolvlm-500m",
    "smoldocling-256m", "llava-ov-0.5b", "got-ocr2", "florence2-large", "florence2-base",
    "paddleocr-vl", "paddleocr-vl-1.5", "ovis2-1b", "h2ovl-0.8b",
}


def test_expected_models_registered():
    keys = set(list_models())
    missing = EXPECTED - keys
    assert not missing, f"missing adapters: {missing}"


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_each_model_builds_with_profile(key):
    m = build_model(key, device="cpu", dtype="float32")
    prof = m.profile()
    assert prof["key"] == key
    assert prof["hf_id"]  # non-empty HF id
    assert prof["param_count_m"] > 0
    assert m.key == key


def test_param_counts_sub_1b_ish():
    # all registered candidates should be at/under ~1B (1000M), allowing the ~1.0B borderline
    for key in EXPECTED:
        assert build_model(key, device="cpu").param_count_m <= 1000.0


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("does-not-exist")


def test_adapter_profile_records_pinned_revision():
    revision = "a" * 40
    model = build_model(
        "lfm2_5-vl-1.6b",
        revision=revision,
        device="cpu",
    )

    assert model.profile()["revision"] == revision


def test_dummy_model_generates(tiny_image):
    import docvlm_eval.models.dummy  # noqa: F401
    m = build_model("dummy-echo", device="cpu")
    m.load()
    text, conf = m.generate(tiny_image, "What is the total?")
    assert isinstance(text, str) and text
    assert 0.0 <= conf < 1.0
