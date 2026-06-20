"""A5 LoRA-placement resolver — introspects a model's modules into capability buckets.

Pure (no torch/GPU): a fake module tree drives resolve_lora_targets via a custom is_linear.
"""

from docvlm_eval.finetune.lora_vlm import PLACEMENT_GROUPS, resolve_lora_targets

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


def test_llm_attn_group_is_attn_leaves_on_llm_side_only():
    a = R("llm_attn")
    assert set(a) == {"model.language_model.layers.0.self_attn.q_proj",
                      "model.language_model.layers.0.self_attn.o_proj"}


def test_llm_mlp_group():
    m = R("llm_mlp")
    assert set(m) == {"model.language_model.layers.0.mlp.gate_proj",
                      "model.language_model.layers.0.mlp.down_proj"}


def test_all_group_is_every_linear():
    assert len(R("all")) == sum(1 for _, m in TREE if m == "L")


def test_groups_are_disjoint_and_cover_llm_split():
    import pytest
    with pytest.raises(ValueError):
        R("bogus")
    # vision/connector/llm_attn/llm_mlp don't overlap
    seen = []
    for g in ("vision", "connector", "llm_attn", "llm_mlp"):
        seen += R(g)
    assert len(seen) == len(set(seen))
    assert set(PLACEMENT_GROUPS) == {"vision", "connector", "llm_attn", "llm_mlp", "all"}
