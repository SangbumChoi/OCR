"""Held-out generation evaluation for the native structured document VLM."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import torch

from ..metrics.calibration import (
    expected_calibration_error,
    fit_temperature_scaling,
    temperature_scale_confidence,
)
from ..metrics.text import score_sample
from ..schema import Sample
from ..generation_policy import (
    resolve_generation_token_budget,
    validate_generation_token_budget_policy,
)
from .data import StudentCollator, visual_model_inputs
from .generation import has_repeated_suffix_cycle
from .model import DocumentVLMStudent
from .posttrain import StructuredPostTrainingDataset, posttraining_prompt_batch
from .pretrain import _autocast_context
from .rewards import (
    RewardConfig,
    parse_structured_response,
    score_structured_response,
)


def _supported_generation_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop optional generation controls unsupported by custom model wrappers."""

    signature = inspect.signature(method)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

ROBUSTNESS_AXES = (
    "document_family",
    "language",
    "evidence_count",
    "degradation",
    "overlay",
    "page_count",
    "document_count",
)


@dataclass(frozen=True)
class StructuredEvalConfig:
    output_dir: str
    max_new_tokens: int = 128
    max_new_tokens_hard_cap: int = 128
    max_new_tokens_by_answer_type: tuple[tuple[str, int], ...] = ()
    max_samples: int | None = None
    sample_selection: str = "random"
    use_kv_cache: bool = True
    precision: str = "bfloat16"
    device: str = "auto"
    seed: int = 0
    repetition_guard_min_tokens: int = 24
    repetition_guard_max_period: int = 16
    repetition_guard_repetitions: int = 3

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        validate_generation_token_budget_policy(
            base_tokens=self.max_new_tokens,
            hard_cap=self.max_new_tokens_hard_cap,
            by_answer_type=dict(self.max_new_tokens_by_answer_type),
        )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.sample_selection not in {
            "random",
            "answer_type_round_robin",
        }:
            raise ValueError("unsupported evaluation sample selection")
        if not isinstance(self.use_kv_cache, bool):
            raise ValueError("use_kv_cache must be a boolean")
        if self.precision not in {"auto", "float32", "bfloat16", "float16"}:
            raise ValueError("invalid evaluation precision")
        if (
            self.repetition_guard_min_tokens < 1
            or self.repetition_guard_max_period < 1
            or self.repetition_guard_repetitions < 2
        ):
            raise ValueError("invalid repetition guard controls")


@dataclass(frozen=True)
class StructuredEvalResult:
    output_dir: str
    summary: dict[str, Any]
    per_sample: list[dict[str, Any]]


def select_evaluation_indices(
    dataset: StructuredPostTrainingDataset,
    *,
    max_samples: int | None,
    seed: int,
    strategy: str,
) -> list[int]:
    """Select a deterministic bounded subset without reading target content."""

    indices = list(range(len(dataset)))
    if max_samples is None or len(indices) <= max_samples:
        return indices
    if strategy == "random":
        return sorted(random.Random(seed).sample(indices, max_samples))
    if strategy != "answer_type_round_robin":
        raise ValueError("unsupported evaluation sample selection")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[str(sample.answer_type).strip().lower()].append(index)
    rng = random.Random(seed)
    labels = sorted(groups)
    rng.shuffle(labels)
    for label in labels:
        rng.shuffle(groups[label])

    selected: list[int] = []
    depth = 0
    while len(selected) < max_samples:
        added = False
        for label in labels:
            group = groups[label]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected)


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
        "max_token_rate": _round(
            _mean(
                [
                    float(row["reached_max_new_tokens"])
                    for row in rows
                ]
            )
        ),
        "degenerate_repetition_rate": _round(
            _mean(
                [
                    float(row["degenerate_repetition"])
                    for row in rows
                ]
            )
        ),
        "mean_generation_token_budget": _round(
            _mean(
                [
                    float(row["generation_token_budget"])
                    for row in rows
                ]
            )
        ),
        "budget_escalation_rate": _round(
            _mean(
                [
                    float(row["generation_token_budget_source"] != "default")
                    for row in rows
                ]
            )
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


def _canonical_robustness_slices(
    meta: dict[str, Any],
    *,
    source: str,
    language: str,
) -> dict[str, str]:
    boxes = meta.get("boxes")
    evidence_count = meta.get("evidence_count")
    if evidence_count is None:
        evidence_count = len(boxes) if isinstance(boxes, list) else 0
    degradation = meta.get("degradation")
    if isinstance(degradation, dict):
        degradation = degradation.get("preset")
    overlay_types = meta.get("overlay_types")
    if isinstance(overlay_types, (list, tuple, set)):
        overlay = "+".join(
            sorted({str(value) for value in overlay_types if value})
        ) or "none"
    else:
        overlay = "unknown"
    return {
        "document_family": str(
            meta.get("document_family")
            or meta.get("template_family")
            or meta.get("doc_type")
            or source
            or "unknown"
        ),
        "language": str(language or meta.get("language") or "und"),
        "evidence_count": str(evidence_count),
        "degradation": str(
            degradation
            or meta.get("degraded_preset")
            or meta.get("render_variant")
            or "unknown"
        ),
        "overlay": overlay,
        "page_count": str(meta.get("page_count") or "unknown"),
        "document_count": str(meta.get("document_count") or "unknown"),
    }


def _robustness_summaries(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in ROBUSTNESS_AXES:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["robustness_slices"][axis]].append(row)
        result[axis] = {
            value: _slice_summary(items)
            for value, items in sorted(grouped.items())
        }
    return result


def _robustness_coverage(
    slices: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    axes = {}
    for axis in ROBUSTNESS_AXES:
        counts = {
            value: int(summary["n"])
            for value, summary in slices.get(axis, {}).items()
        }
        unknown = counts.get("unknown", 0) + counts.get("und", 0)
        axes[axis] = {
            "counts": counts,
            "known_samples": sum(counts.values()) - unknown,
            "unknown_samples": unknown,
        }
    return {
        "required_axes": list(ROBUSTNESS_AXES),
        "complete": all(values["unknown_samples"] == 0 for values in axes.values()),
        "axes": axes,
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


def partition_calibration_samples(
    samples: Sequence[Sample],
    *,
    fraction: float,
    min_samples: int,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Deterministically carve calibration rows out of an evaluation split."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be within (0, 1)")
    if min_samples <= 0:
        raise ValueError("calibration min_samples must be positive")
    if len(samples) <= min_samples:
        raise ValueError(
            "source split must contain more rows than calibration min_samples"
        )
    ranked = sorted(
        samples,
        key=lambda sample: hashlib.sha256(
            f"{seed}:{sample.sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    fit_count = max(min_samples, round(len(ranked) * fraction))
    fit_count = min(fit_count, len(ranked) - 1)
    fit_ids = {sample.sample_id for sample in ranked[:fit_count]}
    calibration = [
        sample for sample in samples if sample.sample_id in fit_ids
    ]
    evaluation = [
        sample for sample in samples if sample.sample_id not in fit_ids
    ]
    return calibration, evaluation


def apply_temperature_calibration(
    rows_by_split: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    *,
    fit_split: str,
    output_dir: str | Path,
    correct_threshold: float,
    min_samples: int,
    min_temperature: float,
    max_temperature: float,
    partition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on one split, apply to every split, and rewrite evaluation artifacts."""

    if fit_split not in rows_by_split:
        raise ValueError(f"calibration fit split {fit_split!r} is missing")
    fit_rows = rows_by_split[fit_split]
    fit_confidences = [
        row.get("confidence")
        for row in fit_rows
        if row.get("confidence") is not None
    ]
    fit_correctness = [
        float(float(row["score"]) >= correct_threshold)
        for row in fit_rows
        if row.get("confidence") is not None
    ]
    fitted = fit_temperature_scaling(
        fit_confidences,
        fit_correctness,
        min_samples=min_samples,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    temperature = fitted.temperature
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name, rows in rows_by_split.items():
        raw_confidences: list[float] = []
        calibrated_confidences: list[float] = []
        correctness: list[float] = []
        for row in rows:
            confidence = row.get("confidence")
            target = float(float(row["score"]) >= correct_threshold)
            if confidence is None:
                row["calibrated_confidence"] = None
                continue
            raw_confidences.append(float(confidence))
            correctness.append(target)
            calibrated = (
                temperature_scale_confidence(float(confidence), temperature)
                if temperature is not None
                else None
            )
            row["calibrated_confidence"] = (
                _round(calibrated) if calibrated is not None else None
            )
            if calibrated is not None:
                calibrated_confidences.append(calibrated)
        raw_ece = expected_calibration_error(
            raw_confidences,
            correctness,
        )
        calibrated_ece = (
            expected_calibration_error(
                calibrated_confidences,
                correctness,
            )
            if temperature is not None
            else None
        )
        metrics = {
            "status": fitted.status,
            "fit_split": fit_split,
            "temperature": (
                _round(temperature) if temperature is not None else None
            ),
            "n_confidence": len(raw_confidences),
            "correct_threshold": correct_threshold,
            "raw_ece": _round(raw_ece) if raw_ece is not None else None,
            "calibrated_ece": (
                _round(calibrated_ece)
                if calibrated_ece is not None
                else None
            ),
        }
        summaries[split_name]["calibration"] = metrics
        split_metrics[split_name] = metrics
        split_dir = Path(output_dir) / split_name
        _atomic_write_json(split_dir / "summary.json", summaries[split_name])
        _atomic_write_jsonl(split_dir / "per_sample.jsonl", rows)
    sample_ids = sorted(str(row["sample_id"]) for row in fit_rows)
    artifact = {
        "schema_version": 1,
        "method": "scalar_temperature_scaling",
        "confidence_source": "mean_generated_token_probability",
        "fit_split": fit_split,
        "fit_sample_fingerprint": "sha256:"
        + hashlib.sha256(
            json.dumps(
                sample_ids,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "correct_threshold": correct_threshold,
        "temperature_bounds": [min_temperature, max_temperature],
        "partition": dict(partition or {}),
        "fit": asdict(fitted),
        "splits": split_metrics,
    }
    _atomic_write_json(Path(output_dir) / "calibration.json", artifact)
    return artifact


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
    indices = select_evaluation_indices(
        dataset,
        max_samples=config.max_samples,
        seed=config.seed,
        strategy=config.sample_selection,
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
            generation_token_budget, budget_source = (
                resolve_generation_token_budget(
                    sample.answer_type,
                    base_tokens=config.max_new_tokens,
                    hard_cap=config.max_new_tokens_hard_cap,
                    by_answer_type=config.max_new_tokens_by_answer_type,
                )
            )
            raw_batch = collator([dataset[index]])
            prompt_batch = posttraining_prompt_batch(raw_batch, device)
            prompt_length = int(prompt_batch["input_ids"].shape[1])
            with torch.no_grad(), _autocast_context(device, config.precision):
                generate_with_confidence = getattr(
                    model,
                    "generate_with_confidence",
                    None,
                )
                if generate_with_confidence is None:
                    generation_kwargs = _supported_generation_kwargs(
                        model.generate,
                        {
                            **visual_model_inputs(prompt_batch),
                            "attention_mask": prompt_batch["attention_mask"],
                            "max_new_tokens": generation_token_budget,
                            "eos_token_id": int(tokenizer.eos_token_id),
                            "use_kv_cache": config.use_kv_cache,
                            "repetition_guard_min_tokens": (
                                config.repetition_guard_min_tokens
                            ),
                            "repetition_guard_max_period": (
                                config.repetition_guard_max_period
                            ),
                            "repetition_guard_repetitions": (
                                config.repetition_guard_repetitions
                            ),
                        },
                    )
                    generated = model.generate(
                        prompt_batch["input_ids"],
                        **generation_kwargs,
                    )
                    confidence = None
                else:
                    generated, confidence_tensor = generate_with_confidence(
                        prompt_batch["input_ids"],
                        **visual_model_inputs(prompt_batch),
                        attention_mask=prompt_batch["attention_mask"],
                        max_new_tokens=generation_token_budget,
                        eos_token_id=int(tokenizer.eos_token_id),
                        use_kv_cache=config.use_kv_cache,
                        repetition_guard_min_tokens=(
                            config.repetition_guard_min_tokens
                        ),
                        repetition_guard_max_period=(
                            config.repetition_guard_max_period
                        ),
                        repetition_guard_repetitions=(
                            config.repetition_guard_repetitions
                        ),
                    )
                    confidence = float(confidence_tensor[0].item())
            if generated.ndim != 2 or generated.shape[0] != 1:
                raise ValueError("student.generate must return one rank-two sequence")
            if generated.shape[1] < prompt_length:
                raise ValueError("student.generate returned fewer tokens than the prompt")
            completion_ids = generated[0, prompt_length:].tolist()
            raw_prediction = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip()
            reached_max_new_tokens = (
                len(completion_ids) >= generation_token_budget
                and (
                    not completion_ids
                    or completion_ids[-1] != int(tokenizer.eos_token_id)
                )
            )
            degenerate_repetition = has_repeated_suffix_cycle(
                completion_ids,
                min_tokens=config.repetition_guard_min_tokens,
                max_period=config.repetition_guard_max_period,
                repetitions=config.repetition_guard_repetitions,
            )
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
            robustness_slices = _canonical_robustness_slices(
                sample.meta,
                source=dataset.sources[index],
                language=dataset.languages[index],
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": split_name,
                    "source": dataset.sources[index],
                    "language": dataset.languages[index],
                    "answer_type": sample.answer_type,
                    "metric": sample.metric,
                    "meta": sample.meta,
                    "robustness_slices": robustness_slices,
                    "question": sample.question,
                    "answers": sample.answers,
                    "image_path": sample.image_path,
                    "prediction": raw_prediction,
                    "generated_tokens": len(completion_ids),
                    "generation_token_budget": generation_token_budget,
                    "generation_token_budget_source": budget_source,
                    "reached_max_new_tokens": reached_max_new_tokens,
                    "degenerate_repetition": degenerate_repetition,
                    "confidence": (
                        _round(confidence) if confidence is not None else None
                    ),
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
    robustness_slices = _robustness_summaries(rows)
    summary = {
        "split": split_name,
        "dataset_size": len(dataset),
        "n_samples": len(rows),
        "score": overall["score"],
        "reward": overall["reward"],
        "valid_structure_fraction": overall["valid_structure_fraction"],
        "answer_rate": overall["answer_rate"],
        "max_token_rate": overall["max_token_rate"],
        "degenerate_repetition_rate": overall[
            "degenerate_repetition_rate"
        ],
        "mean_generation_token_budget": overall[
            "mean_generation_token_budget"
        ],
        "budget_escalation_rate": overall["budget_escalation_rate"],
        "elapsed_seconds": _round(elapsed),
        "milliseconds_per_sample": _round(
            elapsed * 1000 / len(rows) if rows else 0.0
        ),
        "generation_backend": (
            "kv_cache" if config.use_kv_cache else "full_prefix"
        ),
        "sample_selection": config.sample_selection,
        "generation_token_budget_policy": {
            "base_tokens": config.max_new_tokens,
            "hard_cap": config.max_new_tokens_hard_cap,
            "by_answer_type": dict(
                config.max_new_tokens_by_answer_type
            ),
        },
        "by_answer_type": _group_summaries(rows, "answer_type"),
        "by_source": _group_summaries(rows, "source"),
        "by_language": _group_summaries(rows, "language"),
        "by_robustness_axis": robustness_slices,
        "robustness_coverage": _robustness_coverage(robustness_slices),
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
    robustness_axes = {}
    train_robustness = train.get("by_robustness_axis", {})
    heldout_robustness = heldout.get("by_robustness_axis", {})
    for axis in ROBUSTNESS_AXES:
        train_values = train_robustness.get(axis, {})
        heldout_values = heldout_robustness.get(axis, {})
        robustness_axes[axis] = {}
        for value in sorted(set(train_values) & set(heldout_values)):
            robustness_axes[axis][value] = {
                name: _round(
                    float(train_values[value][name])
                    - float(heldout_values[value][name])
                )
                for name in (
                    "score",
                    "reward",
                    "valid_structure_fraction",
                    "answer_rate",
                )
            }
    comparison["train_minus_heldout"] = {
        "headline": headline,
        "by_answer_type": axes,
        "by_robustness_axis": robustness_axes,
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
    for name in (
        "max_token_rate",
        "degenerate_repetition_rate",
        "mean_generation_token_budget",
        "budget_escalation_rate",
    ):
        if name in summary:
            value = float(summary[name])
            metrics[f"eval/{split_name}_{name}"] = value
            metrics[f"eval_by_axis/{name}/{split_name}"] = value
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
    for axis, slices in summary.get("by_robustness_axis", {}).items():
        for value, values in slices.items():
            segment = quote(str(value), safe="-_.")
            metrics[f"eval_by_slice/{axis}/{segment}/{split_name}"] = float(
                values["score"]
            )
    for name, values in summary.get("reward_components", {}).items():
        metrics[f"eval_reward/{name}/{split_name}"] = float(values["score"])
    calibration = summary.get("calibration", {})
    for metric_name in ("raw_ece", "calibrated_ece"):
        value = calibration.get(metric_name)
        if value is not None:
            short_name = (
                "ece_raw" if metric_name == "raw_ece" else "ece_calibrated"
            )
            metrics[f"eval/{split_name}_{short_name}"] = float(value)
            metrics[f"eval_by_axis/{short_name}/{split_name}"] = float(value)
    temperature = calibration.get("temperature")
    if temperature is not None:
        metrics[f"eval/{split_name}_calibration_temperature"] = float(
            temperature
        )
    return metrics


def write_split_comparison(
    output_dir: str | Path,
    summaries: dict[str, dict[str, Any]],
) -> Path:
    path = Path(output_dir) / "comparison.json"
    _atomic_write_json(path, compare_split_summaries(summaries))
    return path
