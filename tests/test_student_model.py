import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="native student tests require torch",
)
ROOT = Path(__file__).resolve().parents[1]


def test_tiny_student_multimodal_forward_and_auxiliary_losses():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(3)
    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config)
    input_ids = torch.randint(0, config.language.vocab_size, (2, 6))
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    labels = input_ids.clone()
    labels[attention_mask == 0] = config.ignore_index
    output = model(
        input_ids=input_ids,
        pixel_values=torch.randn(2, 3, 30, 31),
        attention_mask=attention_mask,
        labels=labels,
        box_targets=torch.rand(2, 4),
        orientation_labels=torch.tensor([0, 3]),
        contrastive=True,
    )

    assert output.logits.shape == (2, config.connector.latent_tokens + 6, 256)
    assert output.box_predictions.shape == (2, 4)
    assert torch.all(output.box_predictions[:, 2:] >= output.box_predictions[:, :2])
    assert torch.all((0.0 <= output.box_predictions) & (output.box_predictions <= 1.0))
    assert output.orientation_logits.shape == (2, 4)
    assert set(output.losses) == {
        "autoregressive",
        "box_regression",
        "orientation",
        "region_text_contrastive",
    }
    assert output.loss is not None and torch.isfinite(output.loss)


def test_siglip_objective_trains_logit_scale_and_bias():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(13)
    config = StudentConfig.tiny()
    config = replace(
        config,
        task_heads=replace(
            config.task_heads,
            contrastive_objective="siglip",
        ),
    )
    model = DocumentVLMStudent(config)
    output = model(
        input_ids=torch.randint(0, config.language.vocab_size, (2, 6)),
        pixel_values=torch.randn(2, 3, 32, 32),
        contrastive=True,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.contrastive_logit_scale.grad is not None
    assert model.contrastive_logit_bias.grad is not None


def test_softmax_contrastive_memory_supplies_batch_one_negatives():
    import torch
    import torch.nn.functional as F

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(17)
    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config)
    input_ids = torch.randint(0, config.language.vocab_size, (1, 6))
    pixels = torch.randn(1, 3, 32, 32)
    width = config.task_heads.contrastive_width
    memory_vision = F.normalize(torch.randn(1, width), dim=-1)
    memory_text = F.normalize(torch.randn(1, width), dim=-1)

    same_only = model(
        input_ids=input_ids,
        pixel_values=pixels,
        contrastive=True,
        contrastive_ids=torch.tensor([11]),
        contrastive_vision_keys=memory_vision,
        contrastive_text_keys=memory_text,
        contrastive_key_ids=torch.tensor([11]),
    )
    with_negative = model(
        input_ids=input_ids,
        pixel_values=pixels,
        contrastive=True,
        contrastive_ids=torch.tensor([11]),
        contrastive_vision_keys=memory_vision,
        contrastive_text_keys=memory_text,
        contrastive_key_ids=torch.tensor([29]),
    )

    assert float(
        same_only.losses["region_text_contrastive"].detach()
    ) == pytest.approx(0.0)
    assert with_negative.losses["region_text_contrastive"] > 0
    with_negative.loss.backward()
    assert model.vision_projection.weight.grad is not None
    assert model.text_projection.weight.grad is not None


def test_average_pool_connector_matches_dense_and_packed_sequences():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import AveragePoolProjector

    config = replace(
        StudentConfig.tiny().connector,
        family="average_pool_projector",
        latent_tokens=3,
    )
    connector = AveragePoolProjector(config)
    first = torch.randn(5, config.input_width)
    second = torch.randn(3, config.input_width)
    dense = torch.zeros(2, 5, config.input_width)
    dense[0] = first
    dense[1, :3] = second
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    packed = torch.cat((first, second))
    cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

    dense_output = connector(dense, mask)
    packed_output = connector.forward_packed(packed, cu_seqlens)

    assert dense_output.shape == (2, 3, config.output_width)
    assert torch.allclose(dense_output, packed_output)
    assert connector.last_packed_attention_backend == "pool"


def test_average_pool_student_forward_trains_connector_projection():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    base = StudentConfig.tiny()
    config = replace(
        base,
        connector=replace(
            base.connector,
            family="average_pool_projector",
        ),
    )
    model = DocumentVLMStudent(config)
    model.configure_gradient_checkpointing(
        enabled=True,
        components=("connector",),
        use_reentrant=False,
    )
    input_ids = torch.randint(0, config.language.vocab_size, (2, 6))
    output = model(
        input_ids=input_ids,
        pixel_values=torch.randn(2, 3, 32, 32),
        labels=input_ids,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.connector.projection.weight.grad is not None
    assert (
        model.connector.gradient_probe_anchor
        is model.connector.projection.weight
    )
    assert model.connector.gradient_checkpointing is True


def test_gradient_checkpointing_preserves_forward_and_backward(monkeypatch):
    import torch

    import docvlm_eval.student.model as student_model
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(37)
    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config).train()
    input_ids = torch.randint(0, config.language.vocab_size, (1, 8))
    pixels = torch.randn(1, 3, 24, 32)
    baseline = model(
        input_ids=input_ids,
        pixel_values=pixels,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    expected_logits = baseline.logits.detach()
    expected_loss = baseline.loss.detach()
    model.zero_grad(set_to_none=True)

    calls = 0
    original = student_model.torch_checkpoint

    def counted_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        student_model,
        "torch_checkpoint",
        counted_checkpoint,
    )
    model.configure_gradient_checkpointing(
        enabled=True,
        components=("vision", "connector", "language"),
        use_reentrant=False,
    )
    checkpointed = model(
        input_ids=input_ids,
        pixel_values=pixels,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
    )
    checkpointed.loss.backward()

    assert calls == (
        config.vision.layers
        + config.connector.layers
        + config.language.layers
    )
    assert torch.equal(checkpointed.logits, expected_logits)
    assert torch.equal(checkpointed.loss.detach(), expected_loss)
    assert model.vision.patch_embed.weight.grad is not None
    assert model.connector.latents.grad is not None
    assert model.language.token_embedding.weight.grad is not None
    assert model.gradient_checkpointing_state == {
        "enabled": True,
        "components": ["vision", "connector", "language"],
        "use_reentrant": False,
    }


def test_gradient_checkpointing_state_preserves_disabled_contract():
    import pytest

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny())
    model.configure_gradient_checkpointing(
        enabled=False,
        components=("language", "vision"),
        use_reentrant=True,
    )

    assert model.gradient_checkpointing_state == {
        "enabled": False,
        "components": ["language", "vision"],
        "use_reentrant": True,
    }
    assert model.language.gradient_checkpointing is False
    assert model.vision.gradient_checkpointing is False
    with pytest.raises(ValueError, match="unique"):
        model.configure_gradient_checkpointing(
            enabled=True,
            components=("language", "language"),
        )


def test_visual_position_ids_are_stable_across_batch_canvas_widths():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    tower = DocumentVLMStudent(StudentConfig.tiny()).vision

    assert tower._position_ids(2, 2, torch.device("cpu")).tolist() == [
        0,
        1,
        8,
        9,
    ]
    assert tower._position_ids(2, 3, torch.device("cpu")).tolist() == [
        0,
        1,
        2,
        8,
        9,
        10,
    ]


def test_visual_prefix_is_invariant_to_fully_masked_canvas_extension():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(31)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    pixels = torch.randn(1, 3, 16, 16)
    compact_mask = torch.ones(1, 16, 16, dtype=torch.bool)
    extended = torch.zeros(1, 3, 16, 24)
    extended[:, :, :, :16] = pixels
    extended_mask = torch.zeros(1, 16, 24, dtype=torch.bool)
    extended_mask[:, :, :16] = True

    compact = model.encode_images(pixels, compact_mask)
    padded = model.encode_images(extended, extended_mask)

    assert torch.allclose(compact, padded, atol=1e-6)


def test_tiny_student_supports_text_replay_and_generation():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config).eval()
    input_ids = torch.randint(0, config.language.vocab_size, (1, 4))
    output = model(input_ids=input_ids, labels=input_ids)
    generated = model.generate(input_ids, max_new_tokens=2)

    assert output.logits.shape == (1, 4, config.language.vocab_size)
    assert output.loss is not None
    assert generated.shape == (1, 6)


def test_cached_visual_prefix_matches_direct_multimodal_forward():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(5)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    input_ids = torch.randint(0, 256, (2, 5))
    pixel_values = torch.randn(2, 3, 31, 29)

    direct = model(input_ids, pixel_values=pixel_values).logits
    prefix = model.encode_images(pixel_values)
    cached = model(input_ids, visual_prefix=prefix).logits

    assert torch.equal(direct, cached)


def test_multimodal_generation_encodes_the_image_once(monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    input_ids = torch.randint(0, 256, (1, 4))
    calls = 0
    original = model.vision.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.vision, "forward", counted_forward)
    generated = model.generate(
        input_ids,
        pixel_values=torch.randn(1, 3, 32, 32),
        max_new_tokens=3,
    )

    assert generated.shape == (1, 7)
    assert calls == 1


def test_language_kv_cache_matches_full_prefix_logits():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.compute import (
        estimate_language_kv_cache_bytes,
    )
    from docvlm_eval.student.model import (
        AttentionLayerCache,
        DocumentVLMStudent,
    )

    torch.manual_seed(17)
    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config).eval()
    input_ids = torch.randint(0, config.language.vocab_size, (2, 5))
    visual_prefix = model.encode_images(torch.randn(2, 3, 32, 32))
    attention_mask = torch.ones_like(input_ids)

    full = model(
        input_ids,
        visual_prefix=visual_prefix,
        attention_mask=attention_mask,
    ).logits[:, -1].float()
    cached, state = model.prefill_generation(
        input_ids,
        visual_prefix=visual_prefix,
        attention_mask=attention_mask,
        max_new_tokens=3,
    )

    assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)
    assert state.cache.sequence_length == (
        config.connector.latent_tokens + input_ids.shape[1]
    )
    for layer_cache in state.cache.layers:
        assert isinstance(layer_cache, AttentionLayerCache)
        assert layer_cache.key.shape == layer_cache.value.shape
        assert layer_cache.key.shape[:2] == (
            2,
            config.language.kv_heads,
        )
        assert layer_cache.key.shape[2] == state.cache.capacity
    assert state.cache.tensor_bytes == estimate_language_kv_cache_bytes(
        config,
        sequence_tokens=state.cache.capacity,
        batch_size=2,
        bytes_per_element=4,
    )
    cache_pointers = [
        (layer_cache.key.data_ptr(), layer_cache.value.data_ptr())
        for layer_cache in state.cache.layers
    ]

    sequence = input_ids
    for token in (
        torch.tensor([[7], [8]]),
        torch.tensor([[9], [10]]),
    ):
        sequence = torch.cat((sequence, token), dim=1)
        cached, state = model.decode_generation(token, state)
        full = model(
            sequence,
            visual_prefix=visual_prefix,
            attention_mask=torch.ones_like(sequence),
        ).logits[:, -1].float()
        assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)
        assert [
            (
                layer_cache.key.data_ptr(),
                layer_cache.value.data_ptr(),
            )
            for layer_cache in state.cache.layers
        ] == cache_pointers
    with pytest.raises(ValueError, match="capacity exceeded"):
        model.decode_generation(torch.tensor([[11], [12]]), state)


def test_hybrid_language_cache_matches_full_prefix_and_backpropagates():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.compute import (
        estimate_language_kv_cache_bytes,
    )
    from docvlm_eval.student.model import (
        AttentionLayerCache,
        DocumentVLMStudent,
        ShortConvLayerCache,
    )

    torch.manual_seed(29)
    base = StudentConfig.tiny()
    config = replace(
        base,
        language=replace(
            base.language,
            full_attention_layers=(1,),
            conv_kernel_size=3,
        ),
    )
    model = DocumentVLMStudent(config)
    input_ids = torch.randint(0, config.language.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    training = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    assert training.loss is not None
    expected_logits = training.logits.detach()
    expected_loss = training.loss.detach()
    training.loss.backward()
    assert model.language.blocks[0].conv.conv.weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.configure_gradient_checkpointing(
        enabled=True,
        components=("language",),
        use_reentrant=False,
    )
    checkpointed = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    assert checkpointed.loss is not None
    assert torch.equal(checkpointed.logits, expected_logits)
    assert torch.equal(checkpointed.loss, expected_loss)
    checkpointed.loss.backward()
    assert model.language.blocks[0].conv.conv.weight.grad is not None

    model.eval()
    with torch.no_grad():
        eval_mask = attention_mask.clone()
        eval_mask[1, :2] = 0
        full = model(
            input_ids,
            attention_mask=eval_mask,
        ).logits[:, -1].float()
        cached, state = model.prefill_generation(
            input_ids,
            attention_mask=eval_mask,
            max_new_tokens=3,
        )
        assert torch.allclose(cached, full, atol=1e-5, rtol=1e-5)
        assert isinstance(state.cache.layers[0], ShortConvLayerCache)
        assert isinstance(state.cache.layers[1], AttentionLayerCache)
        conv_state = state.cache.layers[0].state
        assert conv_state.shape == (
            2,
            config.language.width,
            config.language.conv_kernel_size - 1,
        )
        assert state.cache.tensor_bytes == estimate_language_kv_cache_bytes(
            config,
            sequence_tokens=state.cache.capacity,
            batch_size=2,
            bytes_per_element=4,
        )
        cache_pointers = (
            conv_state.data_ptr(),
            state.cache.layers[1].key.data_ptr(),
            state.cache.layers[1].value.data_ptr(),
        )
        sequence = input_ids
        sequence_mask = eval_mask
        for token in (
            torch.tensor([[13], [14]]),
            torch.tensor([[15], [16]]),
        ):
            sequence = torch.cat((sequence, token), dim=1)
            sequence_mask = torch.cat(
                (
                    sequence_mask,
                    torch.ones(
                        sequence_mask.shape[0],
                        1,
                        dtype=sequence_mask.dtype,
                    ),
                ),
                dim=1,
            )
            cached, state = model.decode_generation(token, state)
            full = model(
                sequence,
                attention_mask=sequence_mask,
            ).logits[:, -1].float()
            assert torch.allclose(
                cached,
                full,
                atol=1e-5,
                rtol=1e-5,
            )
            assert (
                state.cache.layers[0].state.data_ptr(),
                state.cache.layers[1].key.data_ptr(),
                state.cache.layers[1].value.data_ptr(),
            ) == cache_pointers


def test_cached_generation_matches_uncached_and_decodes_one_token(monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(19)
    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config).eval()
    input_ids = torch.randint(0, config.language.vocab_size, (2, 4))
    query_lengths = []
    projection = model.language.blocks[0].attn.k_proj
    original = projection.forward

    def counted_forward(values):
        query_lengths.append(int(values.shape[1]))
        return original(values)

    monkeypatch.setattr(projection, "forward", counted_forward)
    cached, cached_confidence = model.generate_with_confidence(
        input_ids,
        max_new_tokens=3,
        use_kv_cache=True,
    )
    assert query_lengths == [4, 1, 1]

    query_lengths.clear()
    uncached, uncached_confidence = model.generate_with_confidence(
        input_ids,
        max_new_tokens=3,
        use_kv_cache=False,
    )
    assert query_lengths == [4, 5, 6]
    assert torch.equal(cached, uncached)
    assert torch.allclose(
        cached_confidence,
        uncached_confidence,
        atol=1e-6,
        rtol=1e-6,
    )


def test_generation_rejects_a_mask_without_an_image():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    with pytest.raises(ValueError, match="pixel_mask requires pixel_values"):
        model.generate(
            torch.randint(0, 256, (1, 4)),
            pixel_mask=torch.ones(1, 32, 32, dtype=torch.bool),
        )


def test_explicit_flex_packing_rejects_an_unsupported_cpu_backend():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        model(
            torch.ones(1, 2, dtype=torch.long),
            packed_pixel_values=torch.zeros(1, 3, 8, 8),
            packed_position_ids=torch.zeros(1, dtype=torch.long),
            packed_cu_seqlens=torch.tensor([0, 1]),
            packed_attention_backend="flex",
        )


def test_flex_block_mask_prevents_cross_document_attention():
    import torch

    from docvlm_eval.student.model import _create_document_block_mask

    cu_seqlens = torch.tensor([0, 3, 5])
    block_mask = _create_document_block_mask(
        cu_seqlens,
        cu_seqlens,
        q_length=5,
        kv_length=5,
        device=torch.device("cpu"),
        compile_mask=False,
    )
    def scalar(value):
        return torch.tensor(value, dtype=torch.long)

    assert bool(
        block_mask.mask_mod(scalar(0), scalar(0), scalar(1), scalar(2))
    )
    assert not bool(
        block_mask.mask_mod(scalar(0), scalar(0), scalar(1), scalar(3))
    )
    assert bool(
        block_mask.mask_mod(scalar(0), scalar(0), scalar(3), scalar(4))
    )


def test_generation_returns_bounded_sequence_confidence():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    input_ids = torch.randint(0, 256, (2, 4))

    generated, confidence = model.generate_with_confidence(
        input_ids,
        max_new_tokens=3,
    )

    assert generated.shape == (2, 7)
    assert confidence.shape == (2,)
    assert torch.all(confidence > 0)
    assert torch.all(confidence <= 1)


def test_random_init_first_step_reaches_the_vision_tower():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    config = StudentConfig.tiny()
    model = DocumentVLMStudent(config)
    ids = torch.randint(0, config.language.vocab_size, (1, 5))
    output = model(ids, pixel_values=torch.randn(1, 3, 32, 32), labels=ids)
    assert output.loss is not None
    output.loss.backward()

    gradient = model.vision.patch_embed.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_student_checkpoint_round_trip(tmp_path):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(9)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    ids = torch.randint(0, 256, (1, 5))
    expected = model(ids).logits
    model.save_pretrained(tmp_path, metadata={"initialization_arm": "I0_random"})

    loaded = DocumentVLMStudent.from_pretrained(tmp_path).eval()

    assert loaded.config == model.config
    assert torch.equal(expected, loaded(ids).logits)
    assert (tmp_path / "metadata.json").exists()


def test_full_meta_model_matches_the_blueprint_estimator():
    import torch

    from docvlm_eval.architecture import estimate_parameters, load_blueprint
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent, count_unique_parameters

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    with torch.device("meta"):
        model = DocumentVLMStudent(StudentConfig.from_blueprint(blueprint))

    actual = count_unique_parameters(model)
    estimated = estimate_parameters(blueprint)
    assert actual == estimated
    assert actual["total"] == 799_919_884
    assert actual["total"] < 1_000_000_000


def test_hybrid_meta_model_matches_the_blueprint_estimator():
    import torch

    from docvlm_eval.architecture import estimate_parameters, load_blueprint
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import (
        DocumentVLMStudent,
        count_unique_parameters,
    )

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    blueprint["student"]["language"]["full_attention_layers"] = [
        2,
        5,
        8,
        11,
        14,
        17,
        20,
        22,
    ]
    with torch.device("meta"):
        model = DocumentVLMStudent(
            StudentConfig.from_blueprint(blueprint)
        )

    actual = count_unique_parameters(model)
    estimated = estimate_parameters(blueprint)
    assert actual == estimated
    assert 800_000_000 < actual["total"] < 1_000_000_000


def test_average_pool_meta_model_matches_the_blueprint_estimator():
    import torch

    from docvlm_eval.architecture import estimate_parameters, load_blueprint
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import (
        DocumentVLMStudent,
        count_unique_parameters,
    )

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    blueprint["student"]["connector"]["family"] = (
        "average_pool_projector"
    )
    with torch.device("meta"):
        model = DocumentVLMStudent(
            StudentConfig.from_blueprint(blueprint)
        )

    actual = count_unique_parameters(model)
    assert actual == estimate_parameters(blueprint)
    assert actual["connector"] == 1_181_184
    assert actual["total"] == 767_942_922


def test_student_config_preserves_invalid_mixer_indices_for_validation():
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.config import StudentConfig

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    blueprint["student"]["language"]["full_attention_layers"] = [
        1.5,
    ]

    config = StudentConfig.from_blueprint(blueprint)

    assert config.language.full_attention_layers == (1.5,)
    assert any(
        "must contain integers" in error
        for error in config.validate()
    )


def test_selective_transfer_depth_maps_exact_shape_blocks_only():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    student_config = StudentConfig.tiny()
    teacher_config = replace(
        student_config,
        language=replace(student_config.language, layers=4),
    )
    student = DocumentVLMStudent(student_config)
    teacher = DocumentVLMStudent(teacher_config)
    with torch.no_grad():
        for index, block in enumerate(teacher.language.blocks):
            block.attn.q_proj.weight.fill_(index + 1)
    before_block_zero = student.language.blocks[0].attn.q_proj.weight.detach().clone()

    report = selective_transfer(
        student,
        teacher.state_dict(),
        {"language": 0.5},
        family="student",
    )

    assert torch.equal(
        student.language.blocks[0].attn.q_proj.weight,
        before_block_zero,
    )
    assert torch.all(student.language.blocks[1].attn.q_proj.weight == 4)
    assert "language.blocks.1.attn.q_proj.weight" in report.copied_keys
    assert report.copied_parameters > 0


def test_siglip_transfer_maps_attention_output_projection():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    student = DocumentVLMStudent(StudentConfig.tiny())
    target = student.state_dict()
    source = {
        "vision_model.encoder.layers.0.self_attn.out_proj.weight": (
            torch.full_like(
                target["vision.blocks.0.attn.o_proj.weight"],
                3.0,
            )
        ),
    }

    report = selective_transfer(
        student,
        source,
        {"vision": 1.0},
        family="siglip",
    )

    assert torch.all(student.vision.blocks[0].attn.o_proj.weight == 3)
    assert "vision.blocks.0.attn.o_proj.weight" in report.copied_keys


def test_lfm2_hybrid_transfer_maps_attention_convolution_and_mlp():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    base = StudentConfig.tiny()
    config = replace(
        base,
        language=replace(
            base.language,
            full_attention_layers=(1,),
        ),
    )
    student = DocumentVLMStudent(config)
    target = student.state_dict()
    source = {
        "model.layers.0.conv.in_proj.weight": torch.full_like(
            target["language.blocks.0.conv.in_proj.weight"],
            1.0,
        ),
        "model.layers.0.feed_forward.w1.weight": torch.full_like(
            target["language.blocks.0.mlp.gate_proj.weight"],
            2.0,
        ),
        "model.language_model.layers.1.self_attn.out_proj.weight": torch.full_like(
            target["language.blocks.1.attn.o_proj.weight"],
            3.0,
        ),
        "model.layers.1.operator_norm.weight": torch.full_like(
            target["language.blocks.1.norm1.weight"],
            4.0,
        ),
    }

    report = selective_transfer(
        student,
        source,
        {"language": 1.0},
        family="lfm2",
    )

    assert torch.all(
        student.language.blocks[0].conv.in_proj.weight == 1
    )
    assert torch.all(
        student.language.blocks[0].mlp.gate_proj.weight == 2
    )
    assert torch.all(
        student.language.blocks[1].attn.o_proj.weight == 3
    )
    assert torch.all(student.language.blocks[1].norm1.weight == 4)
    assert {
        "language.blocks.0.conv.in_proj.weight",
        "language.blocks.0.mlp.gate_proj.weight",
        "language.blocks.1.attn.o_proj.weight",
        "language.blocks.1.norm1.weight",
    } <= set(report.copied_keys)


def test_selective_transfer_reports_shape_mismatch_without_cropping():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    student = DocumentVLMStudent(StudentConfig.tiny())
    source = {key: value.clone() for key, value in student.state_dict().items()}
    key = "vision.patch_embed.weight"
    source[key] = torch.randn(1)
    original = student.state_dict()[key].clone()

    report = selective_transfer(student, source, {"vision": 1.0})

    assert torch.equal(student.state_dict()[key], original)
    assert any(item["target"] == key for item in report.skipped_shape)


def test_structured_mlp_transfer_uses_one_joint_channel_selection():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    base = StudentConfig.tiny()
    config = replace(
        base,
        language=replace(base.language, layers=1, mlp_width=2),
    )
    student = DocumentVLMStudent(config)
    hidden = config.language.width
    gate = torch.zeros(4, hidden)
    up = torch.zeros(4, hidden)
    down = torch.zeros(hidden, 4)
    gate[1].fill_(5.0)
    up[3].fill_(7.0)
    down[:, 2].fill_(0.1)
    source = {
        "language.blocks.0.mlp.gate_proj.weight": gate,
        "language.blocks.0.mlp.up_proj.weight": up,
        "language.blocks.0.mlp.down_proj.weight": down,
    }

    report = selective_transfer(
        student,
        source,
        {"language": 1.0},
        shape_policy="structured_mlp",
    )

    assert torch.equal(
        student.language.blocks[0].mlp.gate_proj.weight,
        gate[[1, 3]],
    )
    assert torch.equal(
        student.language.blocks[0].mlp.up_proj.weight,
        up[[1, 3]],
    )
    assert torch.equal(
        student.language.blocks[0].mlp.down_proj.weight,
        down[:, [1, 3]],
    )
    assert report.structured_tensors == 3
    assert report.structured_parameters == 3 * hidden * 2
    assert len(report.structured_groups) == 1
    group = report.structured_groups[0]
    assert group["selection"] == "joint_l2_salience"
    assert group["source_channels"] == 4
    assert group["target_channels"] == 2
    assert group["channel_index_fingerprint"].startswith("sha256:")


def test_structured_mlp_transfer_fails_closed_for_incomplete_group():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    base = StudentConfig.tiny()
    config = replace(
        base,
        language=replace(base.language, mlp_width=2),
    )
    student = DocumentVLMStudent(config)
    original = student.language.blocks[0].mlp.gate_proj.weight.detach().clone()
    source = {
        "language.blocks.0.mlp.gate_proj.weight": torch.ones(
            4,
            config.language.width,
        ),
    }

    report = selective_transfer(
        student,
        source,
        {"language": 1.0},
        shape_policy="structured_mlp",
    )

    assert torch.equal(
        student.language.blocks[0].mlp.gate_proj.weight,
        original,
    )
    assert report.structured_tensors == 0
    assert report.structured_groups == []
    assert any(
        item["target"] == "language.blocks.0.mlp.gate_proj.weight"
        for item in report.skipped_shape
    )


def test_selective_transfer_rejects_unknown_shape_policy():
    import pytest

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    student = DocumentVLMStudent(StudentConfig.tiny())

    with pytest.raises(ValueError, match="shape_policy"):
        selective_transfer(
            student,
            {},
            {"language": 1.0},
            shape_policy="crop",
        )


def test_meta_selective_transfer_counts_copied_target_shapes_exactly():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    with torch.device("meta"):
        student = DocumentVLMStudent(StudentConfig.tiny())
        source = {
            key: torch.empty_like(value)
            for key, value in student.state_dict().items()
        }

    report = selective_transfer(
        student,
        source,
        {"vision": 1.0},
        family="student",
    )
    target = student.state_dict()

    assert report.copied_parameters == sum(
        target[key].numel() for key in report.copied_keys
    )


def test_student_builder_rejects_a_required_zero_parameter_transfer(tmp_path):
    import torch

    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"unrelated.weight": torch.ones(1)}, checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sub1b_student.py"),
            "--tiny",
            "--tiny-vocab-size",
            "260",
            "--device",
            "cpu",
            "--init-arm",
            "I1_vision",
            "--vision-source",
            str(checkpoint),
            "--vision-family",
            "student",
            "--save",
            str(tmp_path / "student"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "copied zero parameters" in result.stderr


def test_student_builder_rejects_a_nonzero_but_underdosed_transfer(tmp_path):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny(vocab_size=260))
    checkpoint = tmp_path / "underdosed.pt"
    torch.save(
        {
            "vision.norm.weight": model.state_dict()[
                "vision.norm.weight"
            ].clone()
        },
        checkpoint,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sub1b_student.py"),
            "--tiny",
            "--tiny-vocab-size",
            "260",
            "--device",
            "cpu",
            "--init-arm",
            "I1_vision",
            "--vision-source",
            str(checkpoint),
            "--vision-family",
            "student",
            "--save",
            str(tmp_path / "student"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "parameter dose is below" in result.stderr
    assert "'component': 'vision'" in result.stderr


def test_student_builder_records_the_realized_component_transfer_dose(tmp_path):
    import json

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny(vocab_size=260))
    checkpoint = tmp_path / "compatible.pt"
    torch.save(model.state_dict(), checkpoint)
    output = tmp_path / "student"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sub1b_student.py"),
            "--tiny",
            "--tiny-vocab-size",
            "260",
            "--device",
            "cpu",
            "--init-arm",
            "I1_vision",
            "--vision-source",
            str(checkpoint),
            "--vision-family",
            "student",
            "--save",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    report = metadata["transfer_reports"][0]
    assert report["component"] == "vision"
    assert report["target_component_parameters"] > 0
    assert report["realized_component_parameter_fraction"] >= 0.8
    assert report["minimum_component_parameter_fraction"] == 0.8


def test_student_builder_routes_structured_mlp_arm_and_records_groups(tmp_path):
    import json

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny(vocab_size=260))
    state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }
    for index in range(model.config.language.layers):
        prefix = f"language.blocks.{index}.mlp"
        for projection in ("gate_proj", "up_proj"):
            key = f"{prefix}.{projection}.weight"
            state[key] = torch.cat(
                [state[key], torch.ones(4, state[key].shape[1])],
                dim=0,
            )
        key = f"{prefix}.down_proj.weight"
        state[key] = torch.cat(
            [state[key], torch.ones(state[key].shape[0], 4)],
            dim=1,
        )
    checkpoint = tmp_path / "structured.pt"
    torch.save(state, checkpoint)
    output = tmp_path / "student"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sub1b_student.py"),
            "--tiny",
            "--tiny-vocab-size",
            "260",
            "--device",
            "cpu",
            "--init-arm",
            "I5_structured_mlp",
            "--vision-source",
            str(checkpoint),
            "--vision-family",
            "student",
            "--language-source",
            str(checkpoint),
            "--language-family",
            "student",
            "--save",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    language_report = next(
        report
        for report in metadata["transfer_reports"]
        if report["component"] == "language"
    )
    assert language_report["shape_policy"] == "structured_mlp"
    assert language_report["structured_tensors"] == 3
    assert language_report["structured_parameters"] > 0
    assert len(language_report["structured_groups"]) == 1


def test_student_builder_skips_sources_for_inactive_components(tmp_path):
    import json

    import torch

    checkpoint = tmp_path / "unused.pt"
    torch.save({"incompatible.weight": torch.ones(1)}, checkpoint)
    output = tmp_path / "student"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sub1b_student.py"),
            "--tiny",
            "--tiny-vocab-size",
            "260",
            "--device",
            "cpu",
            "--init-arm",
            "I0_random",
            "--vision-source",
            str(checkpoint),
            "--language-source",
            str(checkpoint),
            "--save",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["transfer_reports"] == []
    assert "Skipping vision source" in result.stdout
    assert "Skipping language source" in result.stdout


def test_box_iou_losses_are_zero_for_an_exact_match():
    import torch

    from docvlm_eval.student.losses import (
        complete_box_iou_loss,
        distance_box_iou_loss,
        generalized_box_iou_loss,
    )

    boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9], [0.0, 0.0, 1.0, 1.0]])

    assert torch.allclose(generalized_box_iou_loss(boxes, boxes), torch.tensor(0.0))
    assert torch.allclose(distance_box_iou_loss(boxes, boxes), torch.tensor(0.0))
    assert torch.allclose(complete_box_iou_loss(boxes, boxes), torch.tensor(0.0))


def test_complete_iou_penalizes_aspect_ratio_beyond_distance_iou():
    import torch

    from docvlm_eval.student.losses import (
        box_iou_loss,
        complete_box_iou_loss,
        distance_box_iou_loss,
    )

    horizontal = torch.tensor([[0.1, 0.3, 0.9, 0.7]])
    vertical = torch.tensor([[0.3, 0.1, 0.7, 0.9]])

    diou = distance_box_iou_loss(horizontal, vertical)
    ciou = complete_box_iou_loss(horizontal, vertical)

    assert ciou > diou
    assert torch.equal(box_iou_loss(horizontal, vertical, kind="ciou"), ciou)
    with pytest.raises(ValueError, match="unsupported box IoU loss"):
        box_iou_loss(horizontal, vertical, kind="plain_iou")


@pytest.mark.parametrize("kind", ["giou", "diou", "ciou"])
def test_box_iou_losses_have_finite_coordinate_gradients(kind):
    import torch

    from docvlm_eval.student.losses import box_iou_loss

    predicted = torch.tensor(
        [[0.05, 0.20, 0.65, 0.55]],
        requires_grad=True,
    )
    target = torch.tensor([[0.30, 0.10, 0.75, 0.85]])

    loss = box_iou_loss(predicted, target, kind=kind)
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.all(torch.isfinite(predicted.grad))


def test_external_embedding_transfer_requires_identity_map():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    student = DocumentVLMStudent(StudentConfig.tiny())
    original = student.language.token_embedding.weight.detach().clone()
    external_embedding = torch.arange(256 * 128, dtype=torch.float32).view(256, 128)
    source = {"model.embed_tokens.weight": external_embedding}

    no_map = selective_transfer(student, source, {"language": 1.0}, family="llama")
    assert torch.equal(student.language.token_embedding.weight, original)
    assert no_map.token_rows_copied == 0

    report = selective_transfer(
        student,
        source,
        {"language": 1.0},
        family="llama",
        token_map={7: 11},
    )
    assert torch.equal(student.language.token_embedding.weight[7], external_embedding[11])
    assert torch.equal(student.language.token_embedding.weight[6], original[6])
    assert report.token_rows_copied == 1
    assert report.copied_parameters == 128


def test_auxiliary_heads_are_real_architecture_switches():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent, count_unique_parameters

    config = StudentConfig.tiny()
    disabled = replace(
        config,
        task_heads=replace(
            config.task_heads,
            region_text_contrastive=False,
            orientation=False,
            box_regression=False,
        ),
    )
    model = DocumentVLMStudent(disabled)
    output = model(torch.randint(0, 256, (1, 3)), pixel_values=torch.randn(1, 3, 32, 32))

    assert output.box_predictions is None
    assert output.orientation_logits is None
    assert output.vision_embeddings is None
    assert count_unique_parameters(model)["task_heads"] == 0


def test_student_captures_only_requested_distillation_layers():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny())
    output = model(
        torch.randint(0, 256, (1, 4)),
        pixel_values=torch.randn(1, 3, 32, 32),
        feature_layers={"vision": [0, -1], "language": [1, -1]},
    )

    assert set(output.vision_features) == {0, -1}
    assert set(output.language_features) == {1, -1}
    assert output.vision_features[0].shape == (1, 16, 64)
    assert output.language_features[1].shape == (1, 12, 128)
    assert output.vision_mask.shape == (1, 16)


def test_student_rejects_unknown_distillation_layer():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    model = DocumentVLMStudent(StudentConfig.tiny())

    with pytest.raises(ValueError, match="unknown language feature layers"):
        model(
            torch.randint(0, 256, (1, 4)),
            feature_layers={"language": [99]},
        )
