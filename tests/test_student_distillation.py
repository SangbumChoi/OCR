import copy
import importlib.util
from dataclasses import replace

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="student distillation tests require torch",
)


def _batch():
    import torch

    input_ids = torch.randint(0, 256, (2, 6))
    labels = input_ids.clone()
    labels[:, :3] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "labels": labels,
        "pixel_values": torch.randn(2, 3, 32, 32),
        "pixel_mask": torch.ones(2, 32, 32, dtype=torch.bool),
    }


def test_identical_teacher_has_zero_topk_and_feature_distillation_loss():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.distillation import (
        DistillationConfig,
        DistillationLoss,
        NativeStudentTeacher,
    )
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(13)
    config = StudentConfig.tiny()
    student = DocumentVLMStudent(config).eval()
    teacher_model = copy.deepcopy(student)
    distill_config = DistillationConfig(
        temperature=2.0,
        logit_top_k=16,
        vision_layer_pairs=((0, 0), (-1, -1)),
        language_layer_pairs=((1, 1), (-1, -1)),
    )
    batch = _batch()
    teacher = NativeStudentTeacher(teacher_model, distill_config)
    signals = teacher(batch)
    output = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        feature_layers=distill_config.student_feature_layers,
    )
    loss_module = DistillationLoss(config, config, distill_config)

    losses = loss_module(output, signals, batch["attention_mask"])

    assert signals.topk_indices.shape == (6, 16)
    assert signals.bucket_logits.shape == (6, 17)
    assert torch.allclose(losses["teacher_kl"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        losses["hidden_feature_distillation"],
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_distillation_projects_incompatible_teacher_widths_and_backpropagates():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.distillation import (
        DistillationConfig,
        DistillationLoss,
        NativeStudentTeacher,
    )
    from docvlm_eval.student.model import DocumentVLMStudent

    student_config = StudentConfig.tiny()
    teacher_config = replace(
        student_config,
        vision=replace(student_config.vision, width=96),
        connector=replace(
            student_config.connector,
            input_width=96,
            output_width=192,
            mlp_width=384,
        ),
        language=replace(
            student_config.language,
            width=192,
            mlp_width=384,
        ),
    )
    student = DocumentVLMStudent(student_config)
    teacher_model = DocumentVLMStudent(teacher_config)
    distill_config = DistillationConfig(
        temperature=1.5,
        logit_top_k=8,
        vision_layer_pairs=((0, 0),),
        language_layer_pairs=((0, 0),),
    )
    batch = _batch()
    signals = NativeStudentTeacher(teacher_model, distill_config)(batch)
    output = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        feature_layers=distill_config.student_feature_layers,
    )
    loss_module = DistillationLoss(
        student_config,
        teacher_config,
        distill_config,
    )

    losses = loss_module(output, signals, batch["attention_mask"])
    total = sum(losses.values())
    total.backward()

    assert all(torch.isfinite(loss) for loss in losses.values())
    language_projection = loss_module.language_projections["s0_t0"]
    vision_projection = loss_module.vision_projections["s0_t0"]
    assert language_projection.weight.grad is not None
    assert vision_projection.weight.grad is not None
    assert student.language.blocks[0].attn.q_proj.weight.grad is not None


def test_online_logit_distillation_rejects_a_different_vocabulary():
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.distillation import DistillationConfig, DistillationLoss

    student = StudentConfig.tiny()
    teacher = replace(
        student,
        language=replace(student.language, vocab_size=512),
    )

    with pytest.raises(ValueError, match="identical tokenizer vocabulary"):
        DistillationLoss(student, teacher, DistillationConfig())


def test_text_only_distillation_skips_unavailable_vision_features():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.distillation import (
        DistillationConfig,
        DistillationLoss,
        NativeStudentTeacher,
    )
    from docvlm_eval.student.model import DocumentVLMStudent

    config = StudentConfig.tiny()
    student = DocumentVLMStudent(config)
    teacher_model = DocumentVLMStudent(config)
    teacher_model.load_state_dict(student.state_dict())
    distill_config = DistillationConfig(
        logit_top_k=8,
        vision_layer_pairs=((0, 0),),
        language_layer_pairs=((0, 0),),
    )
    batch = _batch()
    batch.pop("pixel_values")
    batch.pop("pixel_mask")
    signals = NativeStudentTeacher(teacher_model, distill_config)(batch)
    output = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        feature_layers=distill_config.student_feature_layers,
    )

    losses = DistillationLoss(config, config, distill_config)(
        output,
        signals,
        batch["attention_mask"],
    )

    assert signals.vision_features == {}
    assert set(losses) == {"teacher_kl", "hidden_feature_distillation"}
    assert torch.allclose(
        losses["hidden_feature_distillation"],
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_distillation_config_is_read_from_the_blueprint():
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.distillation import DistillationConfig

    config = DistillationConfig.from_blueprint(
        load_blueprint("configs/sub1b_architecture.yaml")
    )

    assert config.temperature == 2.0
    assert config.logit_top_k == 128
    assert config.vision_layer_pairs[-1] == (11, 11)
    assert config.language_layer_pairs[-1] == (22, 22)
