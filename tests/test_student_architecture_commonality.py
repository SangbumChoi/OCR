from pathlib import Path

import yaml

from docvlm_eval.student.architecture_commonality import (
    architecture_commonality,
    load_architecture_catalog,
    profile_from_blueprint,
    transfer_compatibility,
)
from docvlm_eval.student.sweep import apply_json_patch


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


def _lfm_aligned_blueprint():
    blueprint = yaml.safe_load(
        (ROOT / "configs" / "sub1b_architecture.yaml").read_text(
            encoding="utf-8"
        )
    )
    sweep = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_lfm_language_transfer_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    patches = next(
        variant["blueprint_patches"]
        for variant in sweep["variants"]
        if variant["id"] == "lfm_random"
    )
    return apply_json_patch(blueprint, patches)


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


def test_lfm_aligned_profile_is_sub1b_and_operator_compatible():
    from docvlm_eval.architecture import estimate_parameters
    from docvlm_eval.student.compute import (
        estimate_forward_flops,
        estimate_language_kv_cache_bytes,
    )
    from docvlm_eval.student.config import StudentConfig

    profiles = {profile["id"]: profile for profile in _profiles()}
    blueprint = _lfm_aligned_blueprint()
    native_blueprint = yaml.safe_load(
        (ROOT / "configs" / "sub1b_architecture.yaml").read_text(
            encoding="utf-8"
        )
    )
    target = profile_from_blueprint(blueprint)
    compatibility = transfer_compatibility(
        profiles["lfm2.5-vl-1.6b"],
        target,
    )

    assert estimate_parameters(blueprint)["total"] == 814_207_243
    assert (
        compatibility["decisions"]["language.attention"]["mode"]
        == "exact"
    )
    assert (
        compatibility["decisions"]["language.short_convolution"]["mode"]
        == "exact"
    )
    assert (
        compatibility["decisions"]["language.mlp"]["mode"]
        == "structured_mlp"
    )
    native = StudentConfig.from_blueprint(native_blueprint)
    aligned = StudentConfig.from_blueprint(blueprint)
    assert estimate_forward_flops(
        aligned,
        text_tokens=2048,
        vision_tokens=2520,
    ).total < estimate_forward_flops(
        native,
        text_tokens=2048,
        vision_tokens=2520,
    ).total
    assert estimate_language_kv_cache_bytes(
        aligned,
        sequence_tokens=2176,
    ) < estimate_language_kv_cache_bytes(
        native,
        sequence_tokens=2176,
    )


def test_lfm_meta_preflight_copies_most_language_parameters():
    from docvlm_eval.student.architecture_commonality import (
        lfm_meta_transfer_preflight,
    )

    profile = next(
        item for item in _profiles()
        if item["id"] == "lfm2.5-vl-1.6b"
    )
    result = lfm_meta_transfer_preflight(
        _lfm_aligned_blueprint(),
        profile,
    )

    assert result["copied_parameters"] == 553_748_992
    assert result["copied_language_fraction"] > 0.80
    assert result["structured_groups"] == 12
    assert result["shape_skips"] == 0
    assert result["semantic_skips"] == 0
    assert result["missing_source_keys"] == 0
