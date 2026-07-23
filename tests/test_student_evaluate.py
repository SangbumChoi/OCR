import importlib.util
import json

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="student evaluation tests require torch",
)


class _Tokenizer:
    eos_token_id = 2
    responses = {
        20: '{"answer":"42","evidence":[],"rationale":""}',
        21: "not-json",
    }

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        content = [int(token) for token in ids if int(token) != self.eos_token_id]
        return self.responses[content[0]] if content else ""


class _Collator:
    def __call__(self, examples):
        import torch

        marker = 5 if examples[0].sample_id == "eval-1" else 6
        return {
            "input_ids": torch.tensor([[marker, 99]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.tensor([[-100, 99]]),
        }


def _dataset():
    from docvlm_eval.schema import Sample
    from docvlm_eval.student.posttrain import StructuredPostTrainingDataset

    return StructuredPostTrainingDataset(
        [
            Sample(
                sample_id="eval-1",
                image_path="",
                question="What is the total?",
                answers=["42"],
                answer_type="chart-numeric",
                metric="exact",
                meta={"source": "synthetic", "language": "en"},
            ),
            Sample(
                sample_id="eval-2",
                image_path="",
                question="What is the vendor?",
                answers=["Acme"],
                answer_type="kie",
                metric="anls",
                meta={"source": "public", "language": "ko"},
            ),
        ]
    )


def _reward_config():
    from docvlm_eval.student.rewards import RewardConfig

    return RewardConfig(
        weights={
            "answer_correctness": 0.5,
            "normalized_text_similarity": 0.4,
            "calibrated_abstention": 0.1,
        }
    )


def test_structured_evaluation_writes_scores_rewards_and_slices(tmp_path):
    import torch

    from docvlm_eval.student.evaluate import (
        StructuredEvalConfig,
        evaluate_structured_student,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.calls = 0

        def generate(
            self,
            input_ids,
            *,
            pixel_values=None,
            pixel_mask=None,
            max_new_tokens,
            eos_token_id,
        ):
            del pixel_values, pixel_mask, max_new_tokens
            self.calls += 1
            assert input_ids.shape == (1, 1)
            token = 20 if int(input_ids[0, 0]) == 5 else 21
            completion = torch.tensor(
                [[token, eos_token_id]],
                device=input_ids.device,
            )
            return torch.cat((input_ids, completion), dim=1)

    model = Model().train()
    result = evaluate_structured_student(
        model,
        _dataset(),
        _Collator(),
        _Tokenizer(),
        StructuredEvalConfig(
            output_dir=str(tmp_path),
            max_new_tokens=8,
            precision="float32",
            device="cpu",
        ),
        _reward_config(),
        split_name="heldout",
    )

    assert model.calls == 2
    assert model.training
    assert result.summary["dataset_size"] == 2
    assert result.summary["n_samples"] == 2
    assert result.summary["score"] == pytest.approx(0.5)
    assert result.summary["reward"] == pytest.approx(0.5)
    assert result.summary["valid_structure_fraction"] == pytest.approx(0.5)
    assert result.summary["answer_rate"] == pytest.approx(0.5)
    assert result.summary["by_answer_type"]["chart-numeric"]["score"] == 1.0
    assert result.summary["by_answer_type"]["kie"]["score"] == 0.0
    assert result.summary["by_source"]["synthetic"]["score"] == 1.0
    assert result.summary["by_language"]["ko"]["score"] == 0.0
    assert result.summary["reward_components"]["answer_correctness"] == {
        "n": 1,
        "score": 1.0,
    }
    assert result.per_sample[1]["structure_error"]
    assert result.per_sample[0]["meta"] == {
        "source": "synthetic",
        "language": "en",
    }
    assert result.per_sample[0]["confidence"] is None
    assert (tmp_path / "summary.json").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "per_sample.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["eval-1", "eval-2"]


def test_split_comparison_and_wandb_metrics_pair_matching_axes(tmp_path):
    from docvlm_eval.student.evaluate import (
        compare_split_summaries,
        wandb_metrics_for_split,
        write_split_comparison,
    )

    train = {
        "score": 0.8,
        "reward": 0.7,
        "valid_structure_fraction": 0.9,
        "answer_rate": 1.0,
        "by_answer_type": {
            "kie": {
                "score": 0.9,
                "reward": 0.8,
                "valid_structure_fraction": 1.0,
            }
        },
        "by_source": {"synthetic": {"score": 0.8}},
        "by_language": {"en": {"score": 0.8}},
        "reward_components": {"box_iou": {"score": 0.6}},
    }
    heldout = {
        "score": 0.5,
        "reward": 0.4,
        "valid_structure_fraction": 0.7,
        "answer_rate": 0.8,
        "by_answer_type": {
            "kie": {
                "score": 0.6,
                "reward": 0.5,
                "valid_structure_fraction": 0.75,
            }
        },
        "by_source": {"public": {"score": 0.5}},
        "by_language": {"en": {"score": 0.5}},
        "reward_components": {"box_iou": {"score": 0.3}},
    }

    comparison = compare_split_summaries(
        {"train": train, "heldout": heldout}
    )
    assert comparison["train_minus_heldout"]["headline"]["score"] == 0.3
    assert (
        comparison["train_minus_heldout"]["by_answer_type"]["kie"]["reward"]
        == 0.3
    )
    path = write_split_comparison(
        tmp_path,
        {"train": train, "heldout": heldout},
    )
    assert json.loads(path.read_text(encoding="utf-8")) == comparison

    train_metrics = wandb_metrics_for_split(train, "train")
    heldout_metrics = wandb_metrics_for_split(heldout, "heldout")
    assert train_metrics["eval/train_kie"] == 0.9
    assert heldout_metrics["eval/heldout_kie"] == 0.6
    assert train_metrics["eval_by_axis/kie/train"] == 0.9
    assert heldout_metrics["eval_by_axis/kie/heldout"] == 0.6
    assert train_metrics["eval_reward/box_iou/train"] == 0.6
    assert heldout_metrics["eval_reward/box_iou/heldout"] == 0.3
