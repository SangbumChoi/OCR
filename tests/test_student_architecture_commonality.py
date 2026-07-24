from pathlib import Path

import yaml

from docvlm_eval.student.architecture_commonality import (
    architecture_commonality,
    load_architecture_catalog,
    profile_from_blueprint,
    transfer_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]


def _profiles():
    return load_architecture_catalog(
        ROOT / "configs" / "small_vlm_architectures.yaml"
    )


def _target():
    blueprint = yaml.safe_load(
        (ROOT / "configs" / "sub1b_architecture.yaml").read_text(
            encoding="utf-8"
        )
    )
    return profile_from_blueprint(blueprint)


def test_catalog_covers_diverse_small_vlm_architecture_families():
    profiles = _profiles()

    assert len(profiles) == 5
    assert {profile["vision"]["family"] for profile in profiles} >= {
        "vit",
        "fastvithd",
        "davit",
    }
    assert {profile["language"]["mode"] for profile in profiles} == {
        "decoder_only",
        "encoder_decoder",
    }
    assert any(
        profile["language"]["mixer"]
        == "hybrid_short_convolution_attention"
        for profile in profiles
    )


def test_commonality_finds_recurrent_small_vlm_building_blocks():
    result = architecture_commonality(_profiles(), _target())
    common = {
        item["feature"]: item for item in result["common_features"]
    }

    assert common["vision.norm"]["value"] == "layer_norm"
    assert common["language.mode"]["value"] == "decoder_only"
    assert common["language.norm"]["value"] == "rms_norm"
    assert common["language.activation"]["value"] == "swiglu"
    assert common["language.position"]["value"] == "rope"
    assert len(result["compatibility"]) == 5


def test_transfer_preflight_rejects_shape_only_semantic_mismatches():
    profiles = {profile["id"]: profile for profile in _profiles()}
    target = _target()

    florence = transfer_compatibility(
        profiles["florence-2-base"],
        target,
    )
    internvl = transfer_compatibility(
        profiles["internvl3-1b"],
        target,
    )

    assert (
        florence["decisions"]["language.attention"]["mode"]
        == "distill_only"
    )
    assert (
        florence["decisions"]["language.token_embeddings"]["mode"]
        == "distill_only"
    )
    assert (
        internvl["decisions"]["vision.patch_embedding"]["mode"]
        == "distill_only"
    )
    assert internvl["decisions"]["vision.patch_embedding"]["checks"][
        "patch_size"
    ]
    assert not internvl["decisions"]["vision.patch_embedding"]["checks"][
        "width"
    ]
