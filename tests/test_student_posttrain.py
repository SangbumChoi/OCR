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


def _collator(*, max_length=256):
    from docvlm_eval.student.data import StudentCollator, StudentCollatorConfig

    return StudentCollator(
        _Tokenizer(),
        StudentCollatorConfig(
            max_length=max_length,
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
    advantage_estimator="group_standardized",
    policy_start_id=None,
    policy_start_stage="sft",
):
    from docvlm_eval.student.posttrain import RLVRConfig

    return RLVRConfig(
        output_dir=str(output),
        max_steps=max_steps,
        group_size=2,
        advantage_estimator=advantage_estimator,
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
        policy_start_id=policy_start_id,
        policy_start_stage=policy_start_stage,
    )


def _preference_config(
    output,
    max_steps,
    resume=None,
    *,
    objective="dpo",
    dpo_beta=0.1,
    ipo_tau=0.1,
    margin=0.05,
    source="reference_verifier_ranked",
):
    from docvlm_eval.student.posttrain import PreferenceConfig

    return PreferenceConfig(
        output_dir=str(output),
        max_steps=max_steps,
        objective=objective,
        preference_source=source,
        group_size=2,
        minimum_reward_margin=margin,
        dpo_beta=dpo_beta,
        ipo_tau=ipo_tau,
        sequence_reduction="sum",
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        learning_rate=1e-3,
        weight_decay=0.0,
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


def test_posttraining_configs_share_blueprint_checkpointing(tmp_path):
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.posttrain import PreferenceConfig, RLVRConfig, SFTConfig

    blueprint = load_blueprint("configs/sub1b_architecture.yaml")
    sft = SFTConfig.from_blueprint(blueprint, tmp_path / "sft")
    preference = PreferenceConfig.from_blueprint(
        blueprint,
        tmp_path / "preference",
        reference_id="reference",
    )
    rlvr = RLVRConfig.from_blueprint(
        blueprint,
        tmp_path / "rlvr",
        reference_id="reference",
    )

    for config in (sft, preference, rlvr):
        assert config.optimizer.name == "adamw_8bit"
        assert config.gradient_checkpointing is True
        assert config.gradient_checkpointing_components == (
            "vision",
            "connector",
            "language",
        )
        assert config.gradient_checkpointing_use_reentrant is False
    assert preference.objective == "dpo"
    assert (
        preference.preference_source
        == "gold_anchored_verifier_ranked"
    )
    assert preference.sequence_reduction == "sum"
    assert preference.repetition_guard_min_tokens == 24
    assert preference.repetition_guard_max_period == 16
    assert preference.repetition_guard_repetitions == 3
    assert preference.max_new_tokens_hard_cap == 512
    assert dict(preference.max_new_tokens_by_answer_type) == {
        "ocr-full": 512,
        "reading-order": 384,
        "table*": 512,
        "chart*": 256,
        "H-comprehension": 256,
        "H-accounting": 256,
    }
    assert rlvr.use_kv_cache is True
    assert rlvr.repetition_guard_min_tokens == 24
    assert rlvr.repetition_guard_max_period == 16
    assert rlvr.repetition_guard_repetitions == 3
    assert rlvr.max_new_tokens_hard_cap == 512
    assert (
        rlvr.max_new_tokens_by_answer_type
        == preference.max_new_tokens_by_answer_type
    )
    assert rlvr.advantage_estimator == "group_standardized"
    assert sft.as_pretrain_config().optimizer == sft.optimizer


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


def test_group_rollout_cache_matches_full_prefix_sampling(tmp_path):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import sample_completion_group

    torch.manual_seed(23)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    prompt = {
        "input_ids": torch.tensor([[1, 7, 8]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    config = _rl_config(tmp_path, 1)

    torch.manual_seed(101)
    cached = sample_completion_group(
        model,
        prompt,
        _Tokenizer(),
        config,
    )
    torch.manual_seed(101)
    uncached = sample_completion_group(
        model,
        prompt,
        _Tokenizer(),
        replace(config, use_kv_cache=False),
    )

    assert torch.equal(cached[0], uncached[0])
    assert torch.equal(cached[1], uncached[1])
    assert cached[2] == uncached[2]


def test_group_rollout_ends_exact_suffix_cycle(tmp_path, monkeypatch):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import sample_completion_group

    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    config = replace(
        _rl_config(tmp_path, 1),
        max_new_tokens=12,
        repetition_guard_min_tokens=6,
        repetition_guard_max_period=2,
        repetition_guard_repetitions=3,
    )
    monkeypatch.setattr(
        "docvlm_eval.student.posttrain._top_p_sample",
        lambda logits, top_p: torch.full(
            (logits.shape[0], 1),
            7,
            dtype=torch.long,
            device=logits.device,
        ),
    )

    completion_ids, completion_mask, _ = sample_completion_group(
        model,
        {
            "input_ids": torch.tensor([[1, 8]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        _Tokenizer(),
        config,
    )

    assert completion_ids.shape == (2, 6)
    assert torch.all(completion_ids[:, :5] == 7)
    assert torch.all(completion_ids[:, -1] == _Tokenizer.eos_token_id)
    assert torch.all(completion_mask)


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


def test_group_relative_policy_loss_supports_leave_one_out_advantages():
    import torch

    from docvlm_eval.student.posttrain import group_relative_policy_loss

    policy = torch.tensor(
        [[-1.0], [-1.0], [-1.0]],
        requires_grad=True,
    )
    reference = policy.detach().clone()
    rewards = torch.tensor([1.0, 0.5, 0.0])

    _, standardized = group_relative_policy_loss(
        policy,
        reference,
        torch.ones(3, 1, dtype=torch.bool),
        rewards,
        kl_coefficient=0.0,
        advantage_epsilon=1e-4,
        advantage_estimator="group_standardized",
    )
    _, leave_one_out = group_relative_policy_loss(
        policy,
        reference,
        torch.ones(3, 1, dtype=torch.bool),
        rewards,
        kl_coefficient=0.0,
        advantage_epsilon=1e-4,
        advantage_estimator="leave_one_out",
    )

    assert standardized["advantage_std"] == pytest.approx(1.0)
    assert leave_one_out["advantage_std"] == pytest.approx(
        rewards.std(unbiased=False) * 1.5
    )
    assert leave_one_out["advantage_abs_mean"] < standardized[
        "advantage_abs_mean"
    ]


def test_direct_preference_loss_and_pair_selection_follow_verifier_rank():
    import math

    import torch

    from docvlm_eval.student.posttrain import (
        direct_preference_loss,
        select_preference_pair,
    )

    policy = torch.tensor(
        [[-1.0, -1.2], [-1.4, -1.6]],
        requires_grad=True,
    )
    reference = policy.detach().clone()
    mask = torch.ones(2, 2, dtype=torch.bool)

    loss, metrics = direct_preference_loss(
        policy,
        reference,
        mask,
        beta=0.1,
        sequence_reduction="sum",
    )
    loss.backward()
    pair = select_preference_pair(
        torch.tensor([0.5, 1.0, 0.0]),
        minimum_reward_margin=0.05,
    )

    assert float(loss.detach()) == pytest.approx(math.log(2))
    assert float(metrics["preference_logit"].detach()) == pytest.approx(0)
    assert policy.grad is not None
    assert policy.grad[0].mean() < 0
    assert policy.grad[1].mean() > 0
    assert pair is not None
    assert pair[:2] == (1, 2)
    assert pair[2] == pytest.approx(1.0)
    assert (
        select_preference_pair(
            torch.ones(3),
            minimum_reward_margin=0.05,
        )
        is None
    )


def test_ipo_regresses_log_ratio_margin_to_finite_target():
    import torch

    from docvlm_eval.student.posttrain import preference_optimization_loss

    policy = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    loss, metrics = preference_optimization_loss(
        policy,
        reference,
        torch.ones(2, 2, dtype=torch.bool),
        objective="ipo",
        dpo_beta=0.1,
        ipo_tau=0.1,
        sequence_reduction="sum",
    )
    loss.backward()

    assert float(metrics["target_log_ratio_margin"]) == pytest.approx(5.0)
    assert float(loss.detach()) == pytest.approx(25.0)
    assert policy.grad is not None
    assert policy.grad[0].mean() < 0
    assert policy.grad[1].mean() > 0


def test_dpo_tied_verifier_group_skips_optimizer_but_counts_rollout(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_preference
    from docvlm_eval.student.rewards import RewardConfig, build_structured_target

    target = build_structured_target("42")
    observed_budgets = []

    def tied_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer
        observed_budgets.append(config.max_new_tokens)
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [5, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [target, target],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        tied_group,
    )
    torch.manual_seed(47)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    policy = copy.deepcopy(initial)
    result = train_preference(
        policy,
        copy.deepcopy(initial),
        _dataset(),
        _collator(),
        _Tokenizer(),
        replace(
            _preference_config(tmp_path / "tied", 1),
            max_new_tokens_hard_cap=4,
            max_new_tokens_by_answer_type=(("chart*", 4),),
        ),
        RewardConfig(weights={"answer_correctness": 1.0}),
    )

    assert result.preference_step == 1
    assert result.optimizer_step == 0
    assert result.accepted_pairs == 0
    assert result.skipped_pairs == 1
    assert result.student_flops_seen > 0
    assert result.final_metrics["preference/accepted_pair"] == 0
    assert observed_budgets == [4]
    assert (
        result.final_metrics["preference/generation_token_budget"]
        == 4.0
    )
    assert (
        result.final_metrics["preference/generation_budget_escalated"]
        == 1.0
    )
    for name, expected in initial.state_dict().items():
        assert torch.equal(expected, policy.state_dict()[name]), name


def test_gold_anchored_dpo_bootstraps_from_tied_malformed_candidates(
    tmp_path,
    monkeypatch,
):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_preference
    from docvlm_eval.student.rewards import RewardConfig

    def malformed_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [5, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            ["not-json", "not-json"],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        malformed_group,
    )
    torch.manual_seed(47)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    policy = copy.deepcopy(initial)
    result = train_preference(
        policy,
        copy.deepcopy(initial),
        _dataset(),
        _collator(max_length=512),
        _Tokenizer(),
        _preference_config(
            tmp_path / "gold-anchored",
            1,
            source="gold_anchored_verifier_ranked",
        ),
        RewardConfig(weights={"answer_correctness": 1.0}),
    )

    assert result.preference_step == 1
    assert result.optimizer_step == 1
    assert result.accepted_pairs == 1
    assert result.skipped_pairs == 0
    assert result.final_metrics["preference/gold_anchor_applied"] == 1
    assert result.final_metrics["preference/gold_anchor_reward"] == 1
    assert result.final_metrics["preference/sampled_reward_mean"] == 0
    assert (
        result.final_metrics[
            "preference/sampled_valid_structure_fraction"
        ]
        == 0
    )
    assert result.final_metrics["preference/verifier_reward_margin"] == 1
    assert any(
        not torch.equal(initial.state_dict()[name], value)
        for name, value in policy.state_dict().items()
    )


@pytest.mark.parametrize("objective", ["dpo", "ipo"])
def test_preference_checkpoint_resume_matches_uninterrupted_updates(
    tmp_path,
    monkeypatch,
    objective,
):
    from dataclasses import replace

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_preference
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
    torch.manual_seed(53)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    full = copy.deepcopy(initial)
    resumed = copy.deepcopy(initial)
    reference_full = copy.deepcopy(initial)
    reference_resumed = copy.deepcopy(initial)

    full_result = train_preference(
        full,
        reference_full,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _preference_config(
            tmp_path / f"full-{objective}",
            2,
            objective=objective,
        ),
        rewards,
    )
    first_result = train_preference(
        resumed,
        reference_resumed,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _preference_config(
            tmp_path / f"resumed-{objective}",
            1,
            objective=objective,
        ),
        rewards,
    )
    with pytest.raises(ValueError, match="objective contract mismatch"):
        train_preference(
            copy.deepcopy(resumed),
            copy.deepcopy(reference_resumed),
            _dataset(),
            _collator(),
            _Tokenizer(),
            replace(
                _preference_config(
                    tmp_path / f"resumed-{objective}",
                    2,
                    "latest",
                    objective=objective,
                ),
                **(
                    {"dpo_beta": 0.2}
                    if objective == "dpo"
                    else {"ipo_tau": 0.2}
                ),
            ),
            rewards,
        )
    resumed_result = train_preference(
        resumed,
        reference_resumed,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _preference_config(
            tmp_path / f"resumed-{objective}",
            2,
            "latest",
            objective=objective,
        ),
        rewards,
    )

    assert full_result.optimizer_step == 2
    assert first_result.optimizer_step == 1
    assert resumed_result.optimizer_step == 2
    for name, expected in full.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    for name, expected in initial.state_dict().items():
        assert torch.equal(expected, reference_resumed.state_dict()[name]), name


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
    assert result.policy_signal_steps == 0
    assert result.replay_only_steps == 1
    assert result.final_metrics["rlvr/policy_signal_step"] == 0
    assert result.final_metrics["rlvr/policy_signal_steps"] == 0
    assert result.final_metrics["rlvr/replay_only_step"] == 1
    assert result.final_metrics["rlvr/replay_only_steps"] == 1
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


def test_malformed_recovery_drives_an_on_policy_signal(tmp_path, monkeypatch):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import (
        RewardConfig,
        build_structured_target,
    )

    target = build_structured_target("42")

    def malformed_group(model, prompt_batch, tokenizer, config):
        del model, tokenizer, config
        device = prompt_batch["input_ids"].device
        return (
            torch.tensor([[5, 2], [6, 2]], device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            [target[:-1], "not-json"],
        )

    monkeypatch.setattr(
        "docvlm_eval.student.posttrain.sample_completion_group",
        malformed_group,
    )
    torch.manual_seed(43)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    policy = copy.deepcopy(initial)
    result = train_grpo(
        policy,
        copy.deepcopy(initial),
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(tmp_path / "malformed-recovery", 1),
        RewardConfig(
            weights={"answer_correctness": 1.0},
            malformed_recovery_max=0.1,
        ),
    )

    assert result.policy_signal_steps == 1
    assert result.replay_only_steps == 0
    assert result.final_metrics["rlvr/advantage_abs_mean"] > 0
    assert result.final_metrics["rlvr/policy_signal_step"] == 1
    assert result.final_metrics["rlvr/policy_signal_steps"] == 1
    assert result.final_metrics["rlvr/malformed_fraction"] == 1
    assert (
        result.final_metrics[
            "reward_diagnostic/malformed_recovery_similarity"
        ]
        > 0
    )
    assert any(
        not torch.equal(initial.state_dict()[name], value)
        for name, value in policy.state_dict().items()
    )


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
    from dataclasses import replace

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
    with pytest.raises(ValueError, match="rollout contract mismatch"):
        train_grpo(
            copy.deepcopy(resumed),
            copy.deepcopy(reference_resumed),
            _dataset(),
            _collator(),
            _Tokenizer(),
            replace(
                _rl_config(
                    tmp_path / "resumed",
                    2,
                    "latest",
                    replay_every=1,
                    replay_coefficient=0.5,
                ),
                use_kv_cache=False,
            ),
            rewards,
        )
    with pytest.raises(ValueError, match="objective contract mismatch"):
        train_grpo(
            copy.deepcopy(resumed),
            copy.deepcopy(reference_resumed),
            _dataset(),
            _collator(),
            _Tokenizer(),
            _rl_config(
                tmp_path / "resumed",
                2,
                "latest",
                replay_every=1,
                replay_coefficient=0.5,
                advantage_estimator="leave_one_out",
            ),
            rewards,
        )
    changed_rewards = RewardConfig(
        weights={"answer_correctness": 1.0},
    )
    with pytest.raises(ValueError, match="objective contract mismatch"):
        train_grpo(
            copy.deepcopy(resumed),
            copy.deepcopy(reference_resumed),
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
            changed_rewards,
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


def test_rlvr_resume_pins_preference_warm_start_identity(
    tmp_path,
    monkeypatch,
):
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import (
        RewardConfig,
        build_structured_target,
    )

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
    reward = RewardConfig(weights={"answer_correctness": 1.0})
    train_grpo(
        policy,
        reference,
        _dataset(),
        _collator(),
        _Tokenizer(),
        _rl_config(
            tmp_path / "run",
            1,
            policy_start_id="preference-checkpoint-a",
            policy_start_stage="preference:dpo",
        ),
        reward,
    )

    metadata = json.loads(
        (
            tmp_path
            / "run"
            / "checkpoints"
            / "step-00000001"
            / "student"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["policy_start"] == {
        "content_id": "preference-checkpoint-a",
        "run_stage": "preference:dpo",
    }
    with pytest.raises(ValueError, match="policy-start checkpoint mismatch"):
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
                policy_start_id="preference-checkpoint-b",
                policy_start_stage="preference:dpo",
            ),
            reward,
        )


def test_rlvr_student_flop_budget_stops_after_crossing_target(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace
    from pathlib import Path

    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.posttrain import train_grpo
    from docvlm_eval.student.rewards import (
        RewardConfig,
        build_structured_target,
    )

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
    rewards = RewardConfig(weights={"answer_correctness": 1.0})
    torch.manual_seed(43)
    initial = DocumentVLMStudent(StudentConfig.tiny())
    probe = train_grpo(
        copy.deepcopy(initial),
        copy.deepcopy(initial),
        _dataset(),
        _collator(),
        _Tokenizer(),
        replace(
            _rl_config(tmp_path / "probe", 1),
            gradient_checkpointing=True,
        ),
        rewards,
    )
    result = train_grpo(
        copy.deepcopy(initial),
        copy.deepcopy(initial),
        _dataset(),
        _collator(),
        _Tokenizer(),
        replace(
            _rl_config(tmp_path / "budget", 3),
            max_steps=None,
            total_student_flops=probe.student_flops_seen + 1,
            stop_at_student_flops=True,
            gradient_checkpointing=True,
        ),
        rewards,
    )

    assert result.rollout_step == 2
    assert result.student_flops_seen >= probe.student_flops_seen + 1
    assert result.checkpoint_recompute_flops_seen > 0
    assert result.executed_student_flops_seen == (
        result.student_flops_seen
        + result.checkpoint_recompute_flops_seen
    )
    state = json.loads(
        (
            Path(result.last_checkpoint) / "trainer_state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["student_flops_seen"] == result.student_flops_seen
    assert state["checkpoint_recompute_flops_seen"] == (
        result.checkpoint_recompute_flops_seen
    )
