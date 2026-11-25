import json
from types import SimpleNamespace

import torch

from docvlm_eval.student.weight_commonality import (
    build_weight_commonality_report,
    cross_architecture_weight_commonality,
    refresh_weight_commonality_report,
    semantic_weight_role,
    sketch_remote_safetensors,
    sketch_state_dict,
    validate_weight_commonality_report,
)


REVISION = "a" * 40


def _state(seed: int):
    generator = torch.Generator().manual_seed(seed)

    def weight(*shape):
        return torch.randn(*shape, generator=generator) / shape[-1] ** 0.5

    return {
        "vision.patch_embedding.weight": weight(8, 3, 2, 2),
        "vision.blocks.0.attn.q_proj.weight": weight(8, 8),
        "vision.blocks.0.attn.k_proj.weight": weight(8, 8),
        "vision.blocks.0.attn.v_proj.weight": weight(8, 8),
        "vision.blocks.0.attn.o_proj.weight": weight(8, 8),
        "vision.blocks.0.norm1.weight": torch.ones(8),
        "connector.projection.weight": weight(8, 8),
        "language.token_embedding.weight": weight(16, 8),
        "language.blocks.0.attn.q_proj.weight": weight(8, 8),
        "language.blocks.0.attn.k_proj.weight": weight(4, 8),
        "language.blocks.0.attn.v_proj.weight": weight(4, 8),
        "language.blocks.0.attn.o_proj.weight": weight(8, 8),
        "language.blocks.0.mlp.gate_proj.weight": weight(12, 8),
        "language.blocks.0.mlp.up_proj.weight": weight(12, 8),
        "language.blocks.0.mlp.down_proj.weight": weight(8, 12),
        "language.blocks.0.norm1.weight": torch.ones(8),
    }


def test_semantic_roles_normalize_different_checkpoint_names():
    assert (
        semantic_weight_role(
            "model.text_model.layers.0.self_attn.q_proj.weight",
            [8, 8],
        )
        == "language.attention.q"
    )
    assert (
        semantic_weight_role(
            "model.vision_model.encoder.layers.0.mlp.fc1.weight",
            [16, 8],
        )
        == "vision.mlp.in"
    )
    assert (
        semantic_weight_role(
            "model.connector.modality_projection.proj.weight",
            [8, 8],
        )
        == "connector.projection"
    )


def test_state_sketch_is_bounded_and_does_not_persist_raw_values():
    profile = sketch_state_dict(
        _state(7),
        model_id="test/model",
        max_tensors_per_role=1,
        max_values_per_tensor=16,
    )

    assert profile["sampled_tensors"] == profile["sampled_roles"]
    assert profile["sampled_values"] <= profile["sampled_tensors"] * 16
    assert profile["bytes_read"] == profile["sampled_values"] * 4
    encoded = json.dumps(profile)
    assert "vision.patch_embedding.weight" not in encoded
    assert "raw_values" not in encoded
    assert profile["sample_digest"].startswith("sha256:")


def test_role_health_rejects_one_degenerate_tensor_among_healthy_layers():
    state = _state(7)
    state["language.blocks.1.attn.q_proj.weight"] = torch.zeros(8, 8)

    profile = sketch_state_dict(
        state,
        model_id="test/model",
        max_tensors_per_role=2,
        max_values_per_tensor=16,
    )

    role = profile["roles"]["language.attention.q"]
    assert role["min_rms"] == 0
    assert role["max_zero_fraction"] == 1
    assert role["sample_healthy"] is False


def test_cross_model_statistics_never_claim_basis_alignment():
    models = [
        sketch_state_dict(_state(seed), model_id=f"test/model-{seed}")
        for seed in (1, 2, 3)
    ]

    result = cross_architecture_weight_commonality(models)
    roles = {item["role"]: item for item in result["common_roles"]}

    assert roles["language.attention.q"]["stable_across_models"]
    assert roles["language.attention.q"]["transfer_rule"] == (
        "exact_only_with_semantic_and_geometry_match"
    )
    assert roles["language.mlp.gate"]["transfer_rule"] == (
        "exact_or_joint_structured_channel_selection"
    )
    assert result["decision_contract"]["raw_basis_alignment"] == (
        "never_assumed"
    )


def test_remote_sketch_reads_only_bounded_safetensor_ranges(monkeypatch):
    import huggingface_hub

    values = torch.linspace(-1, 1, 96).numpy().astype("<f4").tobytes()
    header = json.dumps(
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [12, 8],
                "data_offsets": [0, len(values)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    padding = b" " * ((8 - len(header) % 8) % 8)
    header += padding
    file_bytes = len(header).to_bytes(8, "little") + header + values
    tensor_info = SimpleNamespace(
        dtype="F32",
        shape=[12, 8],
        data_offsets=(0, len(values)),
        parameter_count=96,
    )
    metadata = SimpleNamespace(
        files_metadata={
            "model.safetensors": SimpleNamespace(
                tensors={
                    "model.layers.0.self_attn.q_proj.weight": tensor_info
                }
            )
        }
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(sha=REVISION)
        ),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "get_safetensors_metadata",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_url",
        lambda *_args, **_kwargs: "memory://model.safetensors",
    )

    def range_get(_url, start, end):
        return file_bytes[start : end + 1]

    profile = sketch_remote_safetensors(
        model_id="test/model",
        revision=REVISION,
        max_tensors_per_role=1,
        max_values_per_tensor=12,
        range_get=range_get,
    )

    assert profile["sampled_tensors"] == 1
    assert profile["sampled_values"] == 12
    assert profile["range_requests"] == 4
    assert profile["bytes_read"] == 8 + 12 * 4
    assert profile["roles"]["language.attention.q"]["finite_fraction"] == 1.0


def test_report_validation_binds_pinned_sources_and_sampling_limits():
    profiles = [
        {"model_id": f"test/model-{index}", "revision": REVISION}
        for index in range(3)
    ]

    def sketcher(*, model_id, revision, **_kwargs):
        seed = int(model_id.rsplit("-", 1)[-1])
        return sketch_state_dict(
            _state(seed),
            model_id=model_id,
            revision=revision,
            max_tensors_per_role=1,
            max_values_per_tensor=16,
        )

    report = build_weight_commonality_report(
        profiles,
        sketcher=sketcher,
        max_tensors_per_role=1,
        max_values_per_tensor=16,
    )
    audit = validate_weight_commonality_report(
        report,
        profiles,
        require_remote=False,
    )
    assert audit["status"] == "pass"
    assert audit["stable_role_count"] > 0

    report["commonality"] = {}
    report["report_fingerprint"] = "stale"
    refreshed = refresh_weight_commonality_report(report)
    assert validate_weight_commonality_report(
        refreshed,
        profiles,
        require_remote=False,
    )["status"] == "pass"

    refreshed["models"][0]["revision"] = "b" * 40
    failed = validate_weight_commonality_report(
        refreshed,
        profiles,
        require_remote=False,
    )
    assert failed["status"] == "fail"
    assert any("source identities" in error for error in failed["errors"])
