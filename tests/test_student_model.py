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
    assert actual["total"] == 799_919_882
    assert actual["total"] < 1_000_000_000


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


def test_generalized_box_iou_is_zero_for_an_exact_match():
    import torch

    from docvlm_eval.student.losses import generalized_box_iou_loss

    boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9], [0.0, 0.0, 1.0, 1.0]])

    assert torch.allclose(generalized_box_iou_loss(boxes, boxes), torch.tensor(0.0))


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
