import copy
import importlib.util
import json

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="student pretraining tests require torch",
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + ord(character) % 240 for character in text]


def _loader():
    from PIL import Image
    from torch.utils.data import DataLoader

    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        StudentExample,
    )

    examples = [
        StudentExample(
            sample_id=f"sample-{index}",
            source="smoke",
            task="recognition" if index % 2 else "vqa",
            prompt="Read the value.",
            answer=str(40 + index),
            image=Image.new("RGB", (16 + index, 12 + index), "white"),
            image_key=f"image-{index}",
        )
        for index in range(4)
    ]
    collator = StudentCollator(
        _Tokenizer(),
        StudentCollatorConfig(
            max_length=64,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            vocab_size=256,
            rotation_probability=1.0,
            augmentation_seed=19,
            contrastive=False,
        ),
    )
    return DataLoader(
        examples,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def _config(output, max_steps, resume=None):
    from docvlm_eval.student.pretrain import PretrainConfig

    return PretrainConfig(
        output_dir=str(output),
        epochs=2,
        max_steps=max_steps,
        learning_rate=1e-3,
        min_lr_ratio=0.2,
        weight_decay=0.01,
        warmup_tokens=0,
        total_tokens=1000,
        grad_accum_steps=1,
        checkpoint_every_steps=1,
        eval_every_steps=0,
        log_every_steps=1,
        precision="float32",
        device="cpu",
        resume_from=resume,
        loss_weights={
            "autoregressive": 1.0,
            "orientation": 0.1,
            "box_regression": 0.2,
            "region_text_contrastive": 0.0,
        },
    )


def test_pretraining_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    torch.manual_seed(23)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    uninterrupted = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)

    full_result = train_student(
        uninterrupted,
        _loader(),
        _config(tmp_path / "full", max_steps=4),
    )
    first_result = train_student(
        resumed,
        _loader(),
        _config(tmp_path / "resume", max_steps=2),
    )
    resumed_result = train_student(
        resumed,
        _loader(),
        _config(tmp_path / "resume", max_steps=4, resume="latest"),
    )

    assert full_result.global_step == 4
    assert first_result.global_step == 2
    assert resumed_result.global_step == 4
    assert full_result.tokens_seen == resumed_result.tokens_seen
    assert (tmp_path / "resume" / "latest_checkpoint.txt").exists()
    assert (
        tmp_path
        / "resume"
        / "checkpoints"
        / "step-00000004"
        / "training_state.pt"
    ).exists()
    state = json.loads(
        (
            tmp_path
            / "resume"
            / "checkpoints"
            / "step-00000004"
            / "trainer_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["epoch"] == 1
    assert state["batch_in_epoch"] == 0
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name


def test_distilled_pretraining_saves_projection_state(tmp_path):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.distillation import (
        DistillationConfig,
        DistillationLoss,
        NativeStudentTeacher,
    )
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

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
    distill_config = DistillationConfig(
        logit_top_k=8,
        vision_layer_pairs=((0, 0),),
        language_layer_pairs=((0, 0),),
    )
    student = DocumentVLMStudent(student_config)
    teacher_model = DocumentVLMStudent(teacher_config)
    loss_module = DistillationLoss(
        student_config,
        teacher_config,
        distill_config,
    )

    result = train_student(
        student,
        _loader(),
        _config(tmp_path / "distilled", max_steps=1),
        teacher=NativeStudentTeacher(teacher_model, distill_config),
        distillation_loss=loss_module,
    )
    payload = torch.load(
        f"{result.last_checkpoint}/training_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result.global_step == 1
    assert "language_projections.s0_t0.weight" in payload["distillation_loss"]
    assert "vision_projections.s0_t0.weight" in payload["distillation_loss"]


def test_final_partial_accumulation_window_matches_a_full_step(tmp_path):
    from dataclasses import replace

    import torch
    from torch.utils.data import DataLoader

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    torch.manual_seed(29)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    one_microbatch = copy.deepcopy(initial)
    partial_window = copy.deepcopy(initial)

    train_student(
        one_microbatch,
        _loader(),
        _config(tmp_path / "one", max_steps=1),
    )
    train_student(
        partial_window,
        DataLoader([next(iter(_loader()))], batch_size=None),
        replace(
            _config(tmp_path / "partial", max_steps=1),
            grad_accum_steps=4,
        ),
    )

    for name, expected in one_microbatch.state_dict().items():
        assert torch.equal(expected, partial_window.state_dict()[name]), name


def test_resume_rejects_a_different_tokenizer_fingerprint(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    model = DocumentVLMStudent(StudentConfig.tiny())
    first = replace(
        _config(tmp_path / "fingerprint", max_steps=1),
        tokenizer_fingerprint="sha256:first",
    )
    train_student(model, _loader(), first)
    resumed = replace(
        _config(
            tmp_path / "fingerprint",
            max_steps=2,
            resume="latest",
        ),
        tokenizer_fingerprint="sha256:second",
    )

    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        train_student(model, _loader(), resumed)


def test_resume_rejects_a_different_training_stage(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    model = DocumentVLMStudent(StudentConfig.tiny())
    train_student(
        model,
        _loader(),
        replace(
            _config(tmp_path / "stage", max_steps=1),
            run_stage="sft:evidence_linked",
        ),
    )
    resumed = replace(
        _config(tmp_path / "stage", max_steps=2, resume="latest"),
        run_stage="sft:answer_only",
    )

    with pytest.raises(ValueError, match="run stage"):
        train_student(model, _loader(), resumed)


def test_token_cosine_scheduler_is_driven_by_tokens_not_step_count():
    import torch

    from docvlm_eval.student.pretrain import TokenCosineScheduler

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = TokenCosineScheduler(
        optimizer,
        base_lr=1.0,
        warmup_tokens=100,
        total_tokens=1000,
        min_lr_ratio=0.1,
    )

    assert scheduler.step(50) == pytest.approx(0.5)
    assert scheduler.step(100) == pytest.approx(1.0)
    assert scheduler.step(1000) == pytest.approx(0.1)


def test_evaluation_weights_losses_by_sample_count():
    import torch

    from docvlm_eval.student.model import StudentOutput
    from docvlm_eval.student.pretrain import _DistributedContext, _evaluate

    class _EvalStudent(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            value = input_ids[:, 0].float().mean()
            return StudentOutput(
                logits=torch.empty(0),
                loss=value,
                losses={"autoregressive": value},
            )

    loader = [
        {
            "input_ids": torch.tensor([[1], [1], [1]]),
            "attention_mask": torch.ones(3, 1, dtype=torch.long),
            "labels": torch.ones(3, 1, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[9]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "labels": torch.ones(1, 1, dtype=torch.long),
        },
    ]
    context = _DistributedContext(0, 1, 0, torch.device("cpu"))

    metrics = _evaluate(
        _EvalStudent(),
        {"heldout": loader},
        context,
        {"autoregressive": 1.0},
        "float32",
    )

    assert metrics["eval/heldout/autoregressive"] == pytest.approx(3.0)
    assert metrics["eval/heldout/weighted_loss"] == pytest.approx(3.0)


def test_pretrain_config_is_read_from_the_blueprint(tmp_path):
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.pretrain import PretrainConfig

    config = PretrainConfig.from_blueprint(
        load_blueprint("configs/sub1b_architecture.yaml"),
        tmp_path,
    )

    assert config.learning_rate == 3e-4
    assert config.grad_accum_steps == 8
    assert config.warmup_tokens == 100_000_000
    assert config.total_tokens == 20_000_000_000
    assert config.loss_weights["teacher_kl"] == 0.35


def test_pretrain_config_rejects_invalid_logging_and_loss_controls(tmp_path):
    from docvlm_eval.student.pretrain import PretrainConfig

    with pytest.raises(ValueError, match="log_every_steps"):
        PretrainConfig(
            output_dir=str(tmp_path),
            warmup_tokens=0,
            total_tokens=10,
            log_every_steps=0,
        )
    with pytest.raises(ValueError, match="loss weights"):
        PretrainConfig(
            output_dir=str(tmp_path),
            warmup_tokens=0,
            total_tokens=10,
            loss_weights={"autoregressive": -1.0},
        )
