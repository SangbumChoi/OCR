"""Held-out generation evaluation for the native structured document VLM."""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ..metrics.text import score_sample
from .data import StudentCollator
from .model import DocumentVLMStudent
from .posttrain import StructuredPostTrainingDataset, posttraining_prompt_batch
from .pretrain import _autocast_context
from .rewards import (
    RewardConfig,
    parse_structured_response,
    score_structured_response,
)


@dataclass(frozen=True)
class StructuredEvalConfig:
    output_dir: str
    max_new_tokens: int = 128
    max_samples: int | None = None
    precision: str = "bfloat16"
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.precision not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("invalid evaluation precision")


@dataclass(frozen=True)
class StructuredEvalResult:
    output_dir: str
    summary: dict[str, Any]
    per_sample: list[dict[str, Any]]


def _evaluation_device(name: str) -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("native structured evaluation currently requires one process")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _slice_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "score": _round(_mean([float(row["score"]) for row in rows])),
        "reward": _round(_mean([float(row["reward"]) for row in rows])),
        "valid_structure_fraction": _round(
            _mean([float(row["structurally_valid"]) for row in rows])
        ),
        "answer_rate": _round(
            _mean([float(bool(row["answer"])) for row in rows])
        ),
    }


def _group_summaries(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: _slice_summary(items)
        for name, items in sorted(grouped.items())
    }


def _reward_component_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        applicable = set(row["applicable_rewards"])
        for name, value in row["reward_components"].items():
            if name in applicable:
                values[name].append(float(value))
    return {
        name: {"n": len(scores), "score": _round(_mean(scores))}
        for name, scores in sorted(values.items())
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def evaluate_structured_student(
    model: DocumentVLMStudent,
    dataset: StructuredPostTrainingDataset,
    collator: StudentCollator,
    tokenizer: Any,
    config: StructuredEvalConfig,
    reward_config: RewardConfig,
    *,
    split_name: str = "eval",
) -> StructuredEvalResult:
    """Generate and score one split without exposing gold targets to the model."""

    if not split_name.strip():
        raise ValueError("split_name cannot be empty")
    device = _evaluation_device(config.device)
    indices = list(range(len(dataset)))
    if config.max_samples is not None and len(indices) > config.max_samples:
        indices = sorted(
            random.Random(config.seed).sample(indices, config.max_samples)
        )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.to(device).eval()
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    try:
        for position, index in enumerate(indices, start=1):
            sample = dataset.samples[index]
            raw_batch = collator([dataset[index]])
            prompt_batch = posttraining_prompt_batch(raw_batch, device)
            prompt_length = int(prompt_batch["input_ids"].shape[1])
            with torch.no_grad(), _autocast_context(device, config.precision):
                generated = model.generate(
                    prompt_batch["input_ids"],
                    pixel_values=prompt_batch.get("pixel_values"),
                    pixel_mask=prompt_batch.get("pixel_mask"),
                    max_new_tokens=config.max_new_tokens,
                    eos_token_id=int(tokenizer.eos_token_id),
                )
            if generated.ndim != 2 or generated.shape[0] != 1:
                raise ValueError("student.generate must return one rank-two sequence")
            if generated.shape[1] < prompt_length:
                raise ValueError("student.generate returned fewer tokens than the prompt")
            completion_ids = generated[0, prompt_length:].tolist()
            raw_prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip()
            reward = score_structured_response(
                raw_prediction,
                dataset.contexts[index],
                reward_config,
            )
            if reward.structurally_valid:
                response = parse_structured_response(raw_prediction)
                answer = response.answer
                evidence = [list(box) for box in response.evidence]
                rationale = response.rationale
                standard_score = score_sample(
                    sample.metric,
                    answer,
                    sample.answers,
                )
            else:
                answer = ""
                evidence = []
                rationale = ""
                standard_score = 0.0
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": split_name,
                    "source": dataset.sources[index],
                    "language": dataset.languages[index],
                    "answer_type": sample.answer_type,
                    "metric": sample.metric,
                    "question": sample.question,
                    "answers": sample.answers,
                    "image_path": sample.image_path,
                    "prediction": raw_prediction,
                    "answer": answer,
                    "evidence": evidence,
                    "rationale": rationale,
                    "score": _round(standard_score),
                    "reward": _round(reward.total),
                    "reward_components": {
                        name: _round(value)
                        for name, value in sorted(reward.components.items())
                    },
                    "applicable_rewards": list(reward.applicable),
                    "structurally_valid": reward.structurally_valid,
                    "structure_error": reward.error,
                }
            )
            if position == 1 or position == len(indices) or position % 50 == 0:
                print(
                    f"[student-eval:{split_name}] {position}/{len(indices)}",
                    flush=True,
                )
    finally:
        model.train(was_training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    overall = _slice_summary(rows)
    summary = {
        "split": split_name,
        "dataset_size": len(dataset),
        "n_samples": len(rows),
        "score": overall["score"],
        "reward": overall["reward"],
        "valid_structure_fraction": overall["valid_structure_fraction"],
        "answer_rate": overall["answer_rate"],
        "elapsed_seconds": _round(elapsed),
        "milliseconds_per_sample": _round(
            elapsed * 1000 / len(rows) if rows else 0.0
        ),
        "by_answer_type": _group_summaries(rows, "answer_type"),
        "by_source": _group_summaries(rows, "source"),
        "by_language": _group_summaries(rows, "language"),
        "reward_components": _reward_component_summary(rows),
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_jsonl(output_dir / "per_sample.jsonl", rows)
    return StructuredEvalResult(
        output_dir=str(output_dir),
        summary=summary,
        per_sample=rows,
    )


def compare_split_summaries(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the train-minus-heldout generalization diagnostic when both exist."""

    comparison: dict[str, Any] = {"splits": summaries}
    if "train" not in summaries or "heldout" not in summaries:
        return comparison
    train = summaries["train"]
    heldout = summaries["heldout"]
    headline = {}
    for name in ("score", "reward", "valid_structure_fraction", "answer_rate"):
        headline[name] = _round(float(train[name]) - float(heldout[name]))
    train_axes = train.get("by_answer_type", {})
    heldout_axes = heldout.get("by_answer_type", {})
    axes = {}
    for name in sorted(set(train_axes) & set(heldout_axes)):
        axes[name] = {
            "score": _round(
                float(train_axes[name]["score"])
                - float(heldout_axes[name]["score"])
            ),
            "reward": _round(
                float(train_axes[name]["reward"])
                - float(heldout_axes[name]["reward"])
            ),
            "valid_structure_fraction": _round(
                float(train_axes[name]["valid_structure_fraction"])
                - float(heldout_axes[name]["valid_structure_fraction"])
            ),
        }
    comparison["train_minus_heldout"] = {
        "headline": headline,
        "by_answer_type": axes,
    }
    return comparison


def wandb_metrics_for_split(
    summary: dict[str, Any],
    split_name: str,
) -> dict[str, float]:
    """Flatten one summary with both split-first and axis-first W&B keys."""

    metrics = {
        f"eval/{split_name}_score": float(summary["score"]),
        f"eval/{split_name}_reward": float(summary["reward"]),
        f"eval/{split_name}_valid_structure": float(
            summary["valid_structure_fraction"]
        ),
        f"eval/{split_name}_answer_rate": float(summary["answer_rate"]),
        f"eval_by_axis/score/{split_name}": float(summary["score"]),
        f"eval_by_axis/reward/{split_name}": float(summary["reward"]),
        f"eval_by_axis/valid_structure/{split_name}": float(
            summary["valid_structure_fraction"]
        ),
        f"eval_by_axis/answer_rate/{split_name}": float(
            summary["answer_rate"]
        ),
    }
    for axis, values in summary.get("by_answer_type", {}).items():
        metrics[f"eval/{split_name}_{axis}"] = float(values["score"])
        metrics[f"eval_by_axis/{axis}/{split_name}"] = float(values["score"])
    for source, values in summary.get("by_source", {}).items():
        metrics[f"eval_by_source/{source}/{split_name}"] = float(
            values["score"]
        )
    for language, values in summary.get("by_language", {}).items():
        metrics[f"eval_by_language/{language}/{split_name}"] = float(
            values["score"]
        )
    for name, values in summary.get("reward_components", {}).items():
        metrics[f"eval_reward/{name}/{split_name}"] = float(values["score"])
    return metrics


def write_split_comparison(
    output_dir: str | Path,
    summaries: dict[str, dict[str, Any]],
) -> Path:
    path = Path(output_dir) / "comparison.json"
    _atomic_write_json(path, compare_split_summaries(summaries))
    return path
