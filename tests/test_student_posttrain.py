import copy
import importlib.util
import json

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="student post-training tests require torch",
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 256
    fingerprint = "sha256:posttrain-test"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + ord(character) % 240 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr((int(token) - 3) % 240) for token in ids if token >= 3)


def _dataset(target_mode="evidence_linked"):
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.posttrain import StructuredPostTrainingDataset

    return StructuredPostTrainingDataset(
        [
            Sample(
                sample_id="post-1",
                image_path="",
                question="What is the total?",
                answers=["42"],
                answer_type="chart-numeric",
                metric="relaxed_acc",
                meta={"rationale": "Read the total cell."},
            )
        ],
        target_mode=target_mode,
    )


def _formula_dataset():
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.posttrain import StructuredPostTrainingDataset

    return StructuredPostTrainingDataset(
        [
            Sample(
                sample_id="formula-1",
                image_path="",
                question="Transcribe an equivalent formula.",
                answers=["a^2+2ab+b^2"],
                answer_type="formula",
                metric="ned",
            )
        ],
        target_mode="evidence_linked",
    )


def _collator():
    from docvlm_eval.student.data import StudentCollator, StudentCollatorConfig

    return StudentCollator(
        _Tokenizer(),
        StudentCollatorConfig(
            max_length=256,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            vocab_size=256,
            rotation_probability=0.0,
            contrastive=False,
        ),
    )


def _rl_config(
    output,
    max_steps,
    resume=None,
    *,
    replay_every=0,
    replay_coefficient=0.0,
):
    from docvlm_eval.student.posttrain import RLVRConfig

    return RLVRConfig(
        output_dir=str(output),
        max_steps=max_steps,
        group_size=2,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        kl_coefficient=0.04,
        supervised_replay_every_steps=replay_every,
        supervised_replay_loss_coefficient=replay_coefficient,
        max_grad_norm=1.0,
        precision="float32",
        checkpoint_every_steps=1,
        log_every_steps=1,
        seed=31,
        device="cpu",
        resume_from=resume,
        tokenizer_fingerprint=_Tokenizer.fingerprint,
        reference_id="sft-reference-test",
    )


def test_structured_posttraining_dataset_exposes_ablation_targets():
    import json

    answer_only = json.loads(_dataset("answer_only").target(0))
    free_rationale = json.loads(_dataset("free_rationale").target(0))
    evidence_linked = json.loads(_dataset("evidence_linked").target(0))

    assert answer_only == {"answer": "42", "evidence": [], "rationale": ""}
    assert free_rationale["rationale"] == "Read the total cell."
    assert free_rationale["evidence"] == []
    assert evidence_linked["rationale"] == "Read the total cell."
    assert "Return exactly one JSON object" in _dataset().prompt(0)


def test_group_rollout_reuses_one_visual_encoding(tmp_path, monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import sample_completion_group

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    calls = 0
    original = model.vision.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.vision, "forward", counted_forward)
    completion_ids, completion_mask, texts = sample_completion_group(
        model,
        {
            "input_ids": torch.tensor([[1, 7, 8]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 32, 32),
            "pixel_mask": torch.ones(1, 32, 32, dtype=torch.bool),
        },
        _Tokenizer(),
        _rl_config(tmp_path, 1),
    )

    assert completion_ids.shape[0] == 2
    assert completion_mask.shape == completion_ids.shape
    assert len(texts) == 2
    assert calls == 1


def test_group_relative_policy_loss_uses_reward_rank_and_reference_kl():
    import torch

    from docvlm_eval.student.posttrain import group_relative_policy_loss

    policy = torch.tensor(
        [[-1.0, -1.2], [-1.4, -1.6]],
        requires_grad=True,
    )
    reference = torch.tensor([[-1.1, -1.1], [-1.2, -1.7]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    rewards = torch.tensor([1.0, 0.0])

    loss, metrics = group_relative_policy_loss(
        policy,
        reference,
        mask,
        rewards,
        kl_coefficient=0.04,
        advantage_epsilon=1e-4,
    )
    loss.backward()

    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["reward_std"] == pytest.approx(0.5)
    assert metrics["reference_kl"] >= 0
    assert policy.grad is not None
    assert policy.grad[0].mean() < policy.grad[1].mean()


def test_supervised_replay_updates_zero_advantage_group(tmp_path, monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import RewardConfig, build_structured_target

    tied = build_structured_target("17")

    def tied_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [5, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [tied, tied],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        tied_group,
    )
    torch.manual_seed(41)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    policy = copy.deepcopy(initial)
    reference = copy.deepcopy(initial)
    result = train_grpo(
        policy,
        reference,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(
            tmp_path / "replay",
            1,
            replay_every=1,
            replay_coefficient=0.5,
        ),
        RewardConfig(
            weights={
                "answer_correctness": 0.8,
                "normalized_text_similarity": 0.2,
            }
        ),
    )

    assert result.final_metrics["rlvr/advantage_abs_mean"] == 0
    assert result.final_metrics["rlvr/supervised_replay_applied"] == 1
    assert result.final_metrics["rlvr/supervised_replay_loss"] > 0
    assert result.final_metrics["rlvr/supervised_replay_tokens"] > 0
    assert any(
        not torch.equal(initial.state_dict()[name], value)
        for name, value in policy.state_dict().items()
    )
    metric = json.loads(
        (tmp_path / "replay" / "metrics.jsonl").read_text(encoding="utf-8")
    )
    assert metric["supervised_replay_sample_id"] == "post-1"


def test_symbolic_formula_reward_drives_group_advantage(tmp_path, monkeypatch):
    import torch

    pytest.importorskip("sympy")
    pytest.importorskip("antlr4")
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import RewardConfig, build_structured_target

    equivalent = build_structured_target("(a+b)^2")
    wrong = build_structured_target("a^2+b^2")

    def formula_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [6, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [equivalent, wrong],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        formula_group,
    )
    torch.manual_seed(43)
    policy = DocumentVLMStudent(StudentConfig.tiny())
    reference = copy.deepcopy(policy)
    result = train_grpo(
        policy,
        reference,
        _formula_dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(tmp_path / "formula", 1),
        RewardConfig(
            weights={
                "answer_correctness": 0.25,
                "normalized_text_similarity": 0.25,
                "formula_equivalence": 0.50,
            }
        ),
    )

    assert result.final_metrics["reward/formula_equivalence"] == 0.5
    assert result.final_metrics["rlvr/reward_std"] > 0
    assert result.final_metrics["rlvr/advantage_abs_mean"] > 0


def test_structured_sft_runs_and_marks_the_checkpoint_stage(tmp_path):
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import SFTConfig, train_sft

    result = train_sft(
        DocumentVLMStudent(StudentConfig.tiny()),
        _dataset(),
        _collator(),
        SFTConfig(
            output_dir=str(tmp_path / "sft"),
            epochs=1,
            max_steps=1,
            batch_size=1,
            grad_accum_steps=1,
            learning_rate=1e-3,
            warmup_tokens=0,
            total_tokens=1000,
            precision="float32",
            checkpoint_every_steps=1,
            eval_every_steps=0,
            log_every_steps=1,
            num_workers=0,
            device="cpu",
            tokenizer_fingerprint=_Tokenizer.fingerprint,
        ),
    )
    metadata = json.loads(
        (
            tmp_path
            / "sft"
            / "checkpoints"
            / "step-00000001"
            / "student"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert result.global_step == 1
    assert metadata["run_stage"] == "sft:evidence_linked"


def test_rlvr_checkpoint_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import RewardConfig, build_structured_target

    correct = build_structured_target("42")
    wrong = build_structured_target("17")

    def fixed_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [6, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [correct, wrong],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        fixed_group,
    )
    rewards = RewardConfig(
        weights={
            "answer_correctness": 0.8,
            "normalized_text_similarity": 0.2,
        }
    )
    torch.manual_seed(37)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    full = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)
    reference_full = copy.deepcopy(initial)
    reference_resumed = copy.deepcopy(initial)

    full_result = train_grpo(
        full,
        reference_full,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(
            tmp_path / "full",
            2,
            replay_every=1,
            replay_coefficient=0.5,
        ),
        rewards,
    )
    first_result = train_grpo(
        resumed,
        reference_resumed,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(
            tmp_path / "resumed",
            1,
            replay_every=1,
            replay_coefficient=0.5,
        ),
        rewards,
    )
    resumed_result = train_grpo(
        resumed,
        reference_resumed,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(
            tmp_path / "resumed",
            2,
            "latest",
            replay_every=1,
            replay_coefficient=0.5,
        ),
        rewards,
    )

    assert full_result.optimizer_step == 2
    assert first_result.optimizer_step == 1
    assert resumed_result.optimizer_step == 2
    assert any(
        not torch.equal(initial.state_dict()[name], value)
        for name, value in resumed.state_dict().items()
    )
    for name, expected in full.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    for name, expected in initial.state_dict().items():
        assert torch.equal(expected, reference_resumed.state_dict()[name]), name


def test_rlvr_resume_rejects_changed_supervised_replay_contract(
    tmp_path,
    monkeypatch,
):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import RewardConfig, build_structured_target

    target = build_structured_target("42")

    def fixed_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [6, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [target, target],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        fixed_group,
    )
    policy = DocumentVLMStudent(StudentConfig.tiny())
    reference = copy.deepcopy(policy)
    reward = RewardConfig(
        weights={
            "answer_correctness": 0.8,
            "normalized_text_similarity": 0.2,
        }
    )
    train_grpo(
        policy,
        reference,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(tmp_path / "run", 1),
        reward,
    )

    with pytest.raises(ValueError, match="supervised replay contract mismatch"):
        train_grpo(
            policy,
            reference,
            _dataset(),
            _collator(),
            _Tokenizer(),
            _rl_config(
                tmp_path / "run",
                2,
                "latest",
                replay_every=1,
                replay_coefficient=0.5,
            ),
            reward,
        )
