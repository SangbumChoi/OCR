"""A5 LoRA-placement resolver — introspects a model's modules into capability buckets.

Pure (no torch/GPU): a fake module tree drives resolve_lora_targets via a custom is_linear.
"""

from dataclasses import dataclass

import pytest

from docvlm_eval.finetune.lora_vlm import (
    PLACEMENT_GROUPS,
    resolve_lora_budget,
    resolve_lora_targets,
)

# a representative VLM module tree (Qwen-VL-ish + LFM-ish names); "L" marks an adaptable Linear
TREE = [
    ("model.visual.blocks.0.attn.qkv", "L"),                 # vision attn
    ("model.visual.blocks.0.attn.proj", "L"),                # vision proj
    ("model.visual.merger.mlp.0", "L"),                      # connector (merger)
    ("vision_tower.encoder.layers.0.self_attn.k_proj", "L"),  # vision (lfm/siglip)
    ("multi_modal_projector.linear_1", "L"),                 # connector (lfm)
    ("model.language_model.layers.0.self_attn.q_proj", "L"),  # llm attn
    ("model.language_model.layers.0.self_attn.o_proj", "L"),  # llm attn
    ("model.language_model.layers.0.mlp.gate_proj", "L"),     # llm mlp
    ("model.language_model.layers.0.mlp.down_proj", "L"),     # llm mlp
    ("model.language_model.layers.1.conv.in_proj", "L"),       # LFM short convolution
    ("model.language_model.layers.1.conv.out_proj", "L"),      # not attention
    ("model.language_model.layers.1.feed_forward.w1", "L"),    # LFM MLP
    ("model.language_model.layers.1.feed_forward.w2", "L"),
    ("model.language_model.layers.1.feed_forward.w3", "L"),
    ("lm_head", "x"),                                         # not Linear here -> ignored
]

LINEAR = lambda m: m == "L"  # noqa: E731


def R(group):
    return resolve_lora_targets(TREE, group, is_linear=LINEAR)


def test_vision_group_excludes_connector_and_llm():
    v = R("vision")
    assert "model.visual.blocks.0.attn.qkv" in v
    assert "vision_tower.encoder.layers.0.self_attn.k_proj" in v
    assert all("merger" not in n and "language_model" not in n for n in v)


def test_connector_group():
    c = R("connector")
    assert set(c) == {"model.visual.merger.mlp.0", "multi_modal_projector.linear_1"}


def test_vision_connector_is_the_exact_union():
    assert set(R("vision_connector")) == set(R("vision")) | set(R("connector"))


def test_llm_attn_group_is_attn_leaves_on_llm_side_only():
    a = R("llm_attn")
    assert set(a) == {"model.language_model.layers.0.self_attn.q_proj",
                      "model.language_model.layers.0.self_attn.o_proj"}


def test_llm_mlp_group():
    m = R("llm_mlp")
    assert set(m) == {"model.language_model.layers.0.mlp.gate_proj",
                      "model.language_model.layers.0.mlp.down_proj",
                      "model.language_model.layers.1.feed_forward.w1",
                      "model.language_model.layers.1.feed_forward.w2",
                      "model.language_model.layers.1.feed_forward.w3"}


def test_lfm_short_conv_projections_are_not_mislabeled_as_attention():
    assert "model.language_model.layers.1.conv.out_proj" not in R("llm_attn")


def test_all_group_is_every_linear():
    assert len(R("all")) == sum(1 for _, m in TREE if m == "L")


def test_groups_are_disjoint_and_cover_llm_split():
    with pytest.raises(ValueError):
        R("bogus")
    # vision/connector/llm_attn/llm_mlp don't overlap
    seen = []
    for g in ("vision", "connector", "llm_attn", "llm_mlp"):
        seen += R(g)
    assert len(seen) == len(set(seen))
    assert set(PLACEMENT_GROUPS) == {
        "vision",
        "connector",
        "vision_connector",
        "llm_attn",
        "llm_mlp",
        "all",
    }


@dataclass
class FakeLinear:
    in_features: int
    out_features: int


def test_combined_placement_matches_reference_adapter_budget():
    modules = [
        ("vision_tower.block.q_proj", FakeLinear(100, 100)),
        ("vision_tower.block.v_proj", FakeLinear(100, 100)),
        ("multi_modal_projector.linear_1", FakeLinear(100, 100)),
    ]

    targets, report = resolve_lora_budget(
        modules,
        "vision_connector",
        requested_rank=16,
        requested_alpha=32,
        reference_placement="vision",
        is_linear=lambda module: isinstance(module, FakeLinear),
    )

    assert len(targets) == 3
    assert report["target_trainable_parameters"] == 6_400
    assert report["effective_rank"] == 11
    assert report["effective_alpha"] == 22
    assert report["actual_trainable_parameters"] == 6_600
    assert report["relative_budget_error"] == pytest.approx(0.03125)


def test_lora_budget_rejects_modules_without_linear_dimensions():
    with pytest.raises(ValueError, match="in_features/out_features"):
        resolve_lora_budget(
            [("vision_tower.block.q_proj", object())],
            "vision",
            requested_rank=16,
            requested_alpha=32,
            is_linear=lambda module: True,
        )


def test_lora_budget_rejects_nonpositive_rank():
    with pytest.raises(ValueError, match="rank and alpha"):
        resolve_lora_budget(
            [],
            "vision",
            requested_rank=0,
            requested_alpha=32,
        )
