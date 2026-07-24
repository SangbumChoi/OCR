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


def _loader(visual_sequence_mode="dense", batch_size=1):
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
            visual_sequence_mode=visual_sequence_mode,
        ),
    )
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def _adaptive_loader():
    from PIL import Image
    from torch.utils.data import DataLoader

    from docvlm_eval.student.data import (
        BalancedGroupBatchSampler,
        StudentCollator,
        StudentCollatorConfig,
        StudentExample,
    )

    examples = [
        StudentExample(
            sample_id=f"adaptive-{index}",
            source="smoke",
            task="easy" if index % 2 == 0 else "hard",
            prompt="Read the value.",
            answer=str(index),
            image=Image.new("RGB", (16, 16), "white"),
            image_key=f"adaptive-image-{index}",
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
            rotation_probability=0.0,
            contrastive=False,
        ),
    )
    sampler = BalancedGroupBatchSampler(
        [example.task for example in examples],
        batch_size=1,
        num_batches=4,
        seed=29,
    )
    return DataLoader(
        examples,
        batch_sampler=sampler,
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
    assert full_result.text_tokens_seen == resumed_result.text_tokens_seen
    assert full_result.effective_tokens_seen == resumed_result.effective_tokens_seen
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


def test_adaptive_mixture_resume_matches_uninterrupted_training(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    import torch

    import docvlm_eval.student.pretrain as pretrain_module
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureConfig
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    def fixed_eval(*args, **kwargs):
        del args, kwargs
        return {
            "eval/easy/weighted_loss": 1.0,
            "eval/hard/weighted_loss": 3.0,
        }

    monkeypatch.setattr(pretrain_module, "_evaluate", fixed_eval)
    adaptive = AdaptiveMixtureConfig(
        enabled=True,
        step_size=0.5,
        ema_decay=0.0,
        min_probability=0.02,
        warmup_evaluations=0,
    )
    eval_loaders = {"easy": [object()], "hard": [object()]}
    torch.manual_seed(47)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    uninterrupted = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)
    full_config = replace(
        _config(tmp_path / "adaptive-full", max_steps=8),
        eval_every_steps=1,
        adaptive_mixture=adaptive,
    )
    resume_config = replace(
        full_config,
        output_dir=str(tmp_path / "adaptive-resume"),
    )

    train_student(
        uninterrupted,
        _adaptive_loader(),
        full_config,
        eval_loaders=eval_loaders,
    )
    train_student(
        resumed,
        _adaptive_loader(),
        replace(resume_config, max_steps=2),
        eval_loaders=eval_loaders,
    )
    train_student(
        resumed,
        _adaptive_loader(),
        replace(resume_config, resume_from="latest"),
        eval_loaders=eval_loaders,
    )

    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    full_state = torch.load(
        tmp_path
        / "adaptive-full"
        / "checkpoints"
        / "step-00000008"
        / "training_state.pt",
        weights_only=False,
    )
    resumed_state = torch.load(
        tmp_path
        / "adaptive-resume"
        / "checkpoints"
        / "step-00000008"
        / "training_state.pt",
        weights_only=False,
    )
    assert resumed_state["adaptive_mixture"] == full_state["adaptive_mixture"]
    assert full_state["adaptive_mixture"]["weights"]["hard"] > 0.5


def test_token_budget_repeats_epochs_until_the_declared_total(tmp_path):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    first_batch = next(iter(_loader()))
    tokens_per_batch = int((first_batch["labels"] != -100).sum().item())
    torch.manual_seed(37)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    uninterrupted = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)
    full_config = replace(
        _config(tmp_path / "budget-full", max_steps=None),
        epochs=None,
        stop_at_total_tokens=True,
        total_tokens=tokens_per_batch + 1,
    )
    resumed_config = replace(
        full_config,
        output_dir=str(tmp_path / "budget-resume"),
    )
    result = train_student(
        uninterrupted,
        _loader(),
        full_config,
    )
    first = train_student(
        resumed,
        _loader(),
        replace(resumed_config, max_steps=1),
    )
    resumed_result = train_student(
        resumed,
        _loader(),
        replace(resumed_config, resume_from="latest"),
    )

    assert result.global_step == 2
    assert first.global_step == 1
    assert resumed_result.global_step == 2
    assert result.tokens_seen >= full_config.total_tokens
    assert result.budget_tokens_seen == result.tokens_seen
    assert result.tokens_seen == resumed_result.tokens_seen
    assert result.text_tokens_seen == resumed_result.text_tokens_seen
    assert result.effective_tokens_seen == resumed_result.effective_tokens_seen
    assert result.token_unit == "supervised"
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name


def test_compute_budget_repeats_epochs_and_resumes_exactly(tmp_path):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.compute import estimate_batch_training_flops
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    model_config = StudentConfig.tiny()
    first_batch_flops = estimate_batch_training_flops(
        model_config,
        next(iter(_loader())),
    )
    torch.manual_seed(41)
    initial = DocumentVLMStudent(model_config)
    uninterrupted = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)
    config = replace(
        _config(tmp_path / "compute-full", max_steps=None),
        epochs=None,
        stop_at_student_flops=True,
        total_student_flops=first_batch_flops + 1,
        schedule_unit="student_flops",
        gradient_checkpointing=True,
    )
    resumed_config = replace(
        config,
        output_dir=str(tmp_path / "compute-resume"),
    )

    result = train_student(uninterrupted, _loader(), config)
    first = train_student(
        resumed,
        _loader(),
        replace(resumed_config, max_steps=1),
    )
    resumed_result = train_student(
        resumed,
        _loader(),
        replace(resumed_config, resume_from="latest"),
    )

    assert result.global_step == 2
    assert first.global_step == 1
    assert resumed_result.global_step == 2
    assert result.student_flops_seen >= config.total_student_flops
    assert result.student_flops_seen == resumed_result.student_flops_seen
    assert result.checkpoint_recompute_flops_seen > 0
    assert (
        result.checkpoint_recompute_flops_seen
        == resumed_result.checkpoint_recompute_flops_seen
    )
    assert result.executed_student_flops_seen == (
        result.student_flops_seen
        + result.checkpoint_recompute_flops_seen
    )
    assert result.schedule_unit == "student_flops"
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name


def test_effective_token_count_includes_resampled_visual_prefix():
    from docvlm_eval.student.pretrain import _batch_token_counts

    batch = next(iter(_loader()))
    counts = _batch_token_counts(batch, visual_tokens_per_image=16)

    assert counts["supervised"] == int((batch["labels"] != -100).sum())
    assert counts["text"] == int(batch["attention_mask"].sum())
    assert counts["effective"] == counts["text"] + 16


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
    training_config = _config(tmp_path / "distilled", max_steps=1)

    result = train_student(
        student,
        _loader("packed", batch_size=2),
        replace(
            training_config,
            loss_weights={
                **training_config.loss_weights,
                "teacher_kl": 0.3,
                "hidden_feature_distillation": 0.2,
            },
        ),
        teacher=NativeStudentTeacher(teacher_model, distill_config),
        distillation_loss=loss_module,
    )
    payload = torch.load(
        f"{result.last_checkpoint}/training_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert result.global_step == 1
    assert result.executed_visual_tokens_seen > 0
    assert result.executed_visual_tokens_seen == result.valid_visual_tokens_seen
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


def test_planned_optimizer_steps_accounts_for_partial_epoch_windows():
    from docvlm_eval.student.curriculum import planned_optimizer_steps

    assert planned_optimizer_steps(
        num_batches=5,
        grad_accum_steps=2,
        epochs=3,
        max_steps=None,
    ) == 9
    assert planned_optimizer_steps(
        num_batches=5,
        grad_accum_steps=2,
        epochs=3,
        max_steps=4,
    ) == 4


def test_training_token_curriculum_uses_budget_fraction():
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage

    schedule = CurriculumSchedule(
        unit="training_token_fraction",
        stages=(
            CurriculumStage("bootstrap", 0.25, loss_weights={"autoregressive": 0.5}),
            CurriculumStage("refine", 1.0, loss_weights={"autoregressive": 1.0}),
        ),
    )

    schedule.validate()
    assert schedule.stage_for_fraction(0.24).id == "bootstrap"
    assert schedule.stage_for_fraction(0.25).id == "refine"
    assert schedule.loss_weights_for_fraction(
        {"autoregressive": 2.0},
        0.5,
    )["autoregressive"] == 1.0
    with pytest.raises(ValueError, match="stage_for_fraction"):
        schedule.stage_for_step(1, 4)


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
    assert config.epochs is None
    assert config.stop_at_total_tokens is True
    assert config.token_unit == "effective"
    assert config.visual_tokens_per_image == 64
    assert config.gradient_checkpointing is True
    assert config.gradient_checkpointing_components == (
        "vision",
        "connector",
        "language",
    )
    assert config.gradient_checkpointing_use_reentrant is False
    assert config.loss_weights["teacher_kl"] == 0.0
    assert [stage.id for stage in config.curriculum.stages] == [
        "perception_bootstrap",
        "dense_multilingual_alignment",
        "hard_reasoning_refinement",
    ]
    assert config.curriculum.fingerprint.startswith("sha256:")
    assert config.curriculum.unit == "training_token_fraction"
    assert config.adaptive_mixture.enabled is False
    assert config.adaptive_mixture.step_size == 0.5


def test_pretrain_config_rejects_invalid_adaptive_mixture_contract(tmp_path):
    from docvlm_eval.student.adaptive_mixture import AdaptiveMixtureConfig
    from docvlm_eval.student.curriculum import (
        CurriculumSchedule,
        CurriculumStage,
    )
    from docvlm_eval.student.pretrain import PretrainConfig

    enabled = AdaptiveMixtureConfig(enabled=True)
    with pytest.raises(ValueError, match="periodic heldout"):
        PretrainConfig(
            output_dir=str(tmp_path),
            warmup_tokens=0,
            total_tokens=10,
            eval_every_steps=0,
            adaptive_mixture=enabled,
        )
    with pytest.raises(ValueError, match="curriculum group-weight"):
        PretrainConfig(
            output_dir=str(tmp_path),
            warmup_tokens=0,
            total_tokens=10,
            eval_every_steps=1,
            adaptive_mixture=enabled,
            curriculum=CurriculumSchedule(
                stages=(
                    CurriculumStage(
                        id="weighted",
                        until_fraction=1.0,
                        group_weights={"hard": 1.0},
                    ),
                )
            ),
        )


def test_supervision_contract_rejects_silent_online_teacher_mismatch(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.pretrain import (
        pretraining_supervision_contract,
    )

    base = _config(tmp_path, max_steps=1)
    with pytest.raises(ValueError, match="require a native teacher"):
        pretraining_supervision_contract(
            replace(
                base,
                loss_weights={
                    **base.loss_weights,
                    "teacher_kl": 0.2,
                },
            ),
            has_online_teacher=False,
        )
    with pytest.raises(ValueError, match="provided but"):
        pretraining_supervision_contract(
            base,
            has_online_teacher=True,
        )


def test_checkpoint_records_resolved_supervision_contract(tmp_path):
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    train_student(
        DocumentVLMStudent(StudentConfig.tiny()),
        _loader(),
        _config(tmp_path / "contract", max_steps=1),
    )
    metadata = json.loads(
        (
            tmp_path
            / "contract"
            / "checkpoints"
            / "step-00000001"
            / "student"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )

    contract = metadata["supervision_contract"]
    assert contract["has_online_teacher"] is False
    assert contract["online_teacher_losses"] == []
    assert contract["stages"][0]["active_losses"] == [
        "autoregressive",
        "box_regression",
        "orientation",
    ]


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
    with pytest.raises(ValueError, match="epochs can be null"):
        PretrainConfig(
            output_dir=str(tmp_path),
            epochs=None,
            warmup_tokens=0,
            total_tokens=10,
        )


def test_curriculum_changes_logged_loss_weights_at_optimizer_boundaries(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(
                id="bootstrap",
                until_fraction=0.5,
                loss_weights={"autoregressive": 0.25},
            ),
            CurriculumStage(
                id="refine",
                until_fraction=1.0,
                loss_weights={"autoregressive": 1.5},
            ),
        )
    )
    output = tmp_path / "curriculum"
    train_student(
        DocumentVLMStudent(StudentConfig.tiny()),
        _loader(),
        replace(
            _config(output, max_steps=4),
            curriculum=schedule,
        ),
    )
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [row["train/curriculum_stage"] for row in metrics] == [
        "bootstrap",
        "bootstrap",
        "refine",
        "refine",
    ]
    assert [row["train/loss_weight/autoregressive"] for row in metrics] == [
        0.25,
        0.25,
        1.5,
        1.5,
    ]


def test_resume_rejects_a_changed_curriculum(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    first_schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(
                id="all",
                until_fraction=1.0,
                loss_weights={"autoregressive": 1.0},
            ),
        )
    )
    changed_schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(
                id="all",
                until_fraction=1.0,
                loss_weights={"autoregressive": 0.5},
            ),
        )
    )
    output = tmp_path / "curriculum-resume"
    model = DocumentVLMStudent(StudentConfig.tiny())
    train_student(
        model,
        _loader(),
        replace(
            _config(output, max_steps=1),
            curriculum=first_schedule,
        ),
    )

    with pytest.raises(ValueError, match="curriculum fingerprint"):
        train_student(
            model,
            _loader(),
            replace(
                _config(output, max_steps=2, resume="latest"),
                curriculum=changed_schedule,
            ),
        )


def test_resume_rejects_a_changed_token_budget_contract(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    output = tmp_path / "token-budget-resume"
    model = DocumentVLMStudent(StudentConfig.tiny())
    train_student(
        model,
        _loader(),
        _config(output, max_steps=1),
    )

    with pytest.raises(ValueError, match="token-budget contract"):
        train_student(
            model,
            _loader(),
            replace(
                _config(output, max_steps=2, resume="latest"),
                token_unit="text",
            ),
        )


def test_resume_rejects_a_changed_curriculum_horizon(tmp_path):
    from dataclasses import replace

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import train_student

    schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(id="all", until_fraction=1.0),
        )
    )
    output = tmp_path / "curriculum-horizon"
    model = DocumentVLMStudent(StudentConfig.tiny())
    train_student(
        model,
        _loader(),
        replace(
            _config(output, max_steps=1),
            curriculum=schedule,
        ),
    )

    with pytest.raises(ValueError, match="curriculum horizon"):
        train_student(
            model,
            _loader(),
            replace(
                _config(output, max_steps=2, resume="latest"),
                curriculum=schedule,
            ),
        )
