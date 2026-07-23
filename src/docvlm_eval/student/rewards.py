"""Verifiable structured rewards for native document-VLM post-training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..metrics.bank import semantic_match
from ..metrics.grounding import iou, parse_gold_box
from ..metrics.tables import teds_score
from ..metrics.text import exact_match, relaxed_accuracy
from ..schema import Sample


@dataclass(frozen=True)
class StructuredResponse:
    answer: str
    evidence: tuple[tuple[float, float, float, float], ...] = ()
    rationale: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "answer": self.answer,
                "evidence": [list(box) for box in self.evidence],
                "rationale": self.rationale,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class RewardContext:
    sample_id: str
    answers: tuple[str, ...]
    metric: str = "anls"
    answer_type: str = "default"
    gold_boxes: tuple[tuple[float, float, float, float], ...] = ()
    gold_rationale: str = ""
    abstain_expected: bool = False
    table_expected: bool = False
    chart_expected: bool = False
    formula_expected: bool = False

    @classmethod
    def from_sample(cls, sample: Sample) -> "RewardContext":
        meta = sample.meta or {}
        boxes: list[tuple[float, float, float, float]] = []
        size = meta.get("size")
        raw_box = meta.get("box")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
            boxes.append(_normalize_box(raw_box[:4], size))
        raw_boxes = meta.get("boxes")
        if isinstance(raw_boxes, (list, tuple)):
            for raw in raw_boxes:
                if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                    boxes.append(_normalize_box(raw[:4], size))
        if sample.metric == "grounding":
            for answer in sample.answers:
                parsed = parse_gold_box(answer)
                if parsed is not None:
                    box, image_size = parsed
                    boxes.append(_normalize_box(box, image_size))
        probe = meta.get("probe") if isinstance(meta.get("probe"), dict) else {}
        answer_type = sample.answer_type.lower()
        return cls(
            sample_id=sample.sample_id,
            answers=tuple(str(answer) for answer in sample.answers),
            metric=sample.metric,
            answer_type=sample.answer_type,
            gold_boxes=tuple(dict.fromkeys(boxes)),
            gold_rationale=str(meta.get("rationale") or ""),
            abstain_expected=(
                answer_type == "probe:abstain"
                or str(probe.get("kind") or "").lower() == "abstain"
            ),
            table_expected=sample.metric == "teds" or "table" in answer_type,
            chart_expected=(
                sample.metric == "relaxed_acc"
                or "chart" in answer_type
                or "numeric" in answer_type
            ),
            formula_expected="formula" in answer_type or "latex" in answer_type,
        )


@dataclass(frozen=True)
class RewardConfig:
    weights: dict[str, float]
    malformed_reward: float = 0.0

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("reward weights cannot be empty")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("reward weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one reward weight must be positive")
        if not 0 <= self.malformed_reward <= 1:
            raise ValueError("malformed_reward must be within [0, 1]")

    @classmethod
    def from_blueprint(cls, blueprint: dict[str, Any]) -> "RewardConfig":
        raw = blueprint["training"]["posttraining"]["rlvr"]
        return cls(
            weights={
                str(name): float(weight)
                for name, weight in raw["reward_mix"].items()
            },
            malformed_reward=float(raw.get("malformed_reward", 0.0)),
        )


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    applicable: tuple[str, ...] = ()
    structurally_valid: bool = True
    error: str = ""


_REWARD_NAMES = {
    "answer_correctness",
    "normalized_text_similarity",
    "box_iou",
    "table_tree_similarity",
    "chart_numeric_tolerance",
    "formula_equivalence",
    "grounded_rationale_consistency",
    "calibrated_abstention",
}
_ABSTAIN_FORMS = {
    "[redacted]",
    "redacted",
    "not present",
    "not shown",
    "not legible",
    "n/a",
    "na",
    "none",
    "unknown",
    "cannot determine",
    "can't tell",
    "no",
    "absent",
}
_ABSTAIN_NORMALIZED = {
    re.sub(r"\s+", " ", value.lower()).strip(" .[]")
    for value in _ABSTAIN_FORMS
}


def parse_structured_response(text: str) -> StructuredResponse:
    """Parse the strict answer/evidence/rationale contract or raise ``ValueError``."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("response must be one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    expected_fields = {"answer", "evidence", "rationale"}
    if set(payload) - expected_fields:
        raise ValueError("response JSON contains unsupported fields")
    if set(payload) != expected_fields:
        raise ValueError("response JSON must contain answer, evidence, and rationale")
    answer = payload["answer"]
    evidence = payload["evidence"]
    rationale = payload["rationale"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("response.answer must be a non-empty string")
    if not isinstance(rationale, str):
        raise ValueError("response.rationale must be a string")
    if not isinstance(evidence, list) or len(evidence) > 32:
        raise ValueError("response.evidence must be a list of at most 32 boxes")
    boxes: list[tuple[float, float, float, float]] = []
    for raw_box in evidence:
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("each evidence box must contain four numbers")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_box):
            raise ValueError("evidence coordinates must be numeric")
        box = tuple(float(value) for value in raw_box)
        if not all(0 <= value <= 1 for value in box):
            raise ValueError("evidence coordinates must be normalized to [0, 1]")
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError("evidence boxes must use ordered xyxy coordinates")
        boxes.append(box)
    return StructuredResponse(
        answer=answer.strip(),
        evidence=tuple(boxes),
        rationale=rationale.strip(),
    )


def build_structured_target(
    answer: str,
    *,
    evidence: tuple[tuple[float, float, float, float], ...] = (),
    rationale: str = "",
) -> str:
    return StructuredResponse(answer.strip(), evidence, rationale.strip()).to_json()


def _normalize_box(
    box: Any,
    size: Any,
) -> tuple[float, float, float, float]:
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError("gold evidence box must contain four coordinates")
    if max(values) > 1:
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            raise ValueError("pixel evidence boxes require image size metadata")
        width, height = float(size[0]), float(size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image size metadata must be positive")
        values = [
            values[0] / width,
            values[1] / height,
            values[2] / width,
            values[3] / height,
        ]
    x1, x2 = sorted((max(0.0, min(1.0, values[0])), max(0.0, min(1.0, values[2]))))
    y1, y2 = sorted((max(0.0, min(1.0, values[1])), max(0.0, min(1.0, values[3]))))
    return x1, y1, x2, y2


def _box_reward(
    predicted: tuple[tuple[float, float, float, float], ...],
    gold: tuple[tuple[float, float, float, float], ...],
) -> float:
    if not gold or not predicted:
        return 0.0
    gold_recall = sum(
        max(iou(list(gold_box), list(predicted_box)) for predicted_box in predicted)
        for gold_box in gold
    ) / len(gold)
    predicted_precision = sum(
        max(iou(list(predicted_box), list(gold_box)) for gold_box in gold)
        for predicted_box in predicted
    ) / len(predicted)
    if gold_recall + predicted_precision == 0:
        return 0.0
    overlap_f1 = (
        2 * gold_recall * predicted_precision
        / (gold_recall + predicted_precision)
    )
    count_agreement = min(len(gold), len(predicted)) / max(
        len(gold),
        len(predicted),
    )
    return overlap_f1 * count_agreement


def _formula_normalize(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"^\$+|\$+$", "", normalized)
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = normalized.replace(r"\tfrac", r"\frac")
    normalized = normalized.replace(r"\cdot", "*").replace(r"\times", "*")
    normalized = re.sub(r"\s+", "", normalized)
    while normalized.startswith("{") and normalized.endswith("}"):
        normalized = normalized[1:-1]
    return normalized


def _is_abstention(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.lower()).strip(" .[]")
    return normalized in _ABSTAIN_NORMALIZED


def score_structured_response(
    response_text: str,
    context: RewardContext,
    config: RewardConfig,
) -> RewardResult:
    unknown = set(config.weights) - _REWARD_NAMES
    if unknown:
        raise ValueError(f"unknown reward components: {sorted(unknown)}")
    try:
        response = parse_structured_response(response_text)
    except ValueError as exc:
        return RewardResult(
            total=config.malformed_reward,
            structurally_valid=False,
            error=str(exc),
        )

    components: dict[str, float] = {
        "answer_correctness": exact_match(response.answer, list(context.answers)),
        "normalized_text_similarity": semantic_match(
            response.answer,
            list(context.answers),
        ),
        "calibrated_abstention": float(
            _is_abstention(response.answer) == context.abstain_expected
        ),
    }
    applicable = {
        "answer_correctness",
        "normalized_text_similarity",
        "calibrated_abstention",
    }
    if context.gold_boxes:
        box_score = _box_reward(response.evidence, context.gold_boxes)
        components["box_iou"] = box_score
        applicable.add("box_iou")
        if context.gold_rationale:
            components["grounded_rationale_consistency"] = (
                box_score if response.rationale else 0.0
            )
            applicable.add("grounded_rationale_consistency")
    if context.table_expected:
        components["table_tree_similarity"] = teds_score(
            response.answer,
            list(context.answers),
        )
        applicable.add("table_tree_similarity")
    if context.chart_expected:
        components["chart_numeric_tolerance"] = relaxed_accuracy(
            response.answer,
            list(context.answers),
        )
        applicable.add("chart_numeric_tolerance")
    if context.formula_expected:
        components["formula_equivalence"] = float(
            any(
                _formula_normalize(response.answer) == _formula_normalize(gold)
                for gold in context.answers
            )
        )
        applicable.add("formula_equivalence")

    active_weight = sum(
        config.weights.get(name, 0.0)
        for name in applicable
    )
    total = (
        sum(
            components[name] * config.weights.get(name, 0.0)
            for name in applicable
        )
        / active_weight
        if active_weight > 0
        else 0.0
    )
    return RewardResult(
        total=max(0.0, min(1.0, float(total))),
        components=components,
        applicable=tuple(sorted(applicable)),
    )
