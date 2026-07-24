"""Verifiable structured rewards for native document-VLM post-training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
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
    reasoning_trace: dict[str, Any] | None = None
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
        raw_trace = meta.get("reasoning_trace")
        reasoning_trace = (
            _validate_reasoning_trace(raw_trace)
            if raw_trace is not None
            else None
        )
        return cls(
            sample_id=sample.sample_id,
            answers=tuple(str(answer) for answer in sample.answers),
            metric=sample.metric,
            answer_type=sample.answer_type,
            gold_boxes=tuple(dict.fromkeys(boxes)),
            gold_rationale=str(meta.get("rationale") or ""),
            reasoning_trace=reasoning_trace,
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
    rationale_verifier: str = "evidence_semantic"

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("reward weights cannot be empty")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("reward weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one reward weight must be positive")
        if not 0 <= self.malformed_reward <= 1:
            raise ValueError("malformed_reward must be within [0, 1]")
        if self.rationale_verifier not in {
            "evidence_semantic",
            "evidence_program_trace",
        }:
            raise ValueError(
                "rationale_verifier must be evidence_semantic or "
                "evidence_program_trace"
            )

    @classmethod
    def from_blueprint(cls, blueprint: dict[str, Any]) -> "RewardConfig":
        raw = blueprint["training"]["posttraining"]["rlvr"]
        return cls(
            weights={
                str(name): float(weight)
                for name, weight in raw["reward_mix"].items()
            },
            malformed_reward=float(raw.get("malformed_reward", 0.0)),
            rationale_verifier=str(
                raw.get("rationale_verifier", "evidence_semantic")
            ),
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
    "no aparece",
    "no se muestra",
    "no disponible",
    "기재되어 있지 않음",
    "없음",
    "알 수 없음",
    "記載なし",
    "不明",
    "ありません",
    "未提供",
    "未显示",
    "未知",
}
_ABSTAIN_NORMALIZED = {
    re.sub(r"\s+", " ", value.lower()).strip(" .[]")
    for value in _ABSTAIN_FORMS
}
_TRACE_OPERATIONS = {
    "value",
    "sum",
    "mean",
    "difference",
    "ratio",
    "percent_change",
    "relative_reduction",
    "confidence_interval",
    "significance_decision",
    "significance_claim_consistency",
    "argmax",
    "argmin",
    "weighted_sum",
    "path_product",
    "sum_products",
}
_TRACE_FIELDS = {
    "schema_version",
    "operation",
    "inputs",
    "parameters",
    "answer_value",
    "answer",
    "required_numeric_facts",
    "trace_fingerprint",
}
_NUMERIC_LITERAL = re.compile(
    r"(?<![\w.])(?P<currency>[$€£¥₩])?"
    r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<percent>\s*%)?"
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _trace_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _trace_inputs(
    trace: dict[str, Any],
    input_type: str,
) -> list[dict[str, Any]]:
    inputs = trace["inputs"]
    if any(item["input_type"] != input_type for item in inputs):
        raise ValueError(
            f"reasoning trace operation requires {input_type} inputs"
        )
    return inputs


def _evaluate_reasoning_trace(trace: dict[str, Any]) -> Any:
    operation = trace["operation"]
    parameters = trace["parameters"]
    if operation in {"path_product", "sum_products"}:
        inputs = _trace_inputs(trace, "edge")
        weights = {
            item["id"]: _trace_number(
                item.get("weight"),
                f"reasoning trace edge {item['id']} weight",
            )
            for item in inputs
        }
        if operation == "path_product":
            if not weights:
                raise ValueError("path_product reasoning trace requires edges")
            return math.prod(weights.values())
        paths = parameters.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("sum_products reasoning trace requires paths")
        products: list[float] = []
        for path in paths:
            if not isinstance(path, list) or not path:
                raise ValueError(
                    "sum_products reasoning trace paths must be non-empty lists"
                )
            try:
                products.append(
                    math.prod(weights[str(edge_id)] for edge_id in path)
                )
            except KeyError as exc:
                raise ValueError(
                    "sum_products reasoning trace references an unknown edge"
                ) from exc
        return sum(products)

    inputs = _trace_inputs(trace, "node")
    values = [item.get("value") for item in inputs]
    labels = [str(item.get("label") or item["id"]) for item in inputs]
    if operation == "value":
        if len(values) != 1:
            raise ValueError("value reasoning trace requires one input")
        return values[0]
    numeric = [
        _trace_number(value, f"reasoning trace input {item['id']}")
        for item, value in zip(inputs, values)
    ]
    if operation == "sum":
        if not numeric:
            raise ValueError("sum reasoning trace requires inputs")
        return sum(numeric)
    if operation == "mean":
        if not numeric:
            raise ValueError("mean reasoning trace requires inputs")
        return sum(numeric) / len(numeric)
    if operation in {
        "difference",
        "ratio",
        "percent_change",
        "relative_reduction",
    }:
        if len(numeric) != 2:
            raise ValueError(
                f"{operation} reasoning trace requires two inputs"
            )
        if operation == "difference":
            return numeric[0] - numeric[1]
        if numeric[1] == 0:
            raise ValueError(
                f"{operation} reasoning trace denominator cannot be zero"
            )
        if operation == "ratio":
            return numeric[0] / numeric[1]
        if operation == "relative_reduction":
            return (numeric[1] - numeric[0]) / abs(numeric[1]) * 100.0
        return (numeric[0] - numeric[1]) / abs(numeric[1]) * 100.0
    if operation == "confidence_interval":
        if len(numeric) != 2:
            raise ValueError(
                "confidence_interval reasoning trace requires two inputs"
            )
        mean, standard_error = numeric
        if standard_error < 0:
            raise ValueError(
                "confidence_interval standard error cannot be negative"
            )
        critical = _trace_number(
            parameters.get("critical_value"),
            "confidence interval critical value",
        )
        places = parameters.get("decimal_places")
        separator = parameters.get("separator")
        if critical <= 0:
            raise ValueError(
                "confidence interval critical value must be positive"
            )
        if not isinstance(places, int) or not 0 <= places <= 6:
            raise ValueError(
                "confidence interval decimal_places must be within [0, 6]"
            )
        if (
            not isinstance(separator, str)
            or not separator
            or len(separator) > 16
        ):
            raise ValueError(
                "confidence interval separator must be a short string"
            )
        margin = critical * standard_error
        return (
            f"{mean - margin:.{places}f}"
            f"{separator}"
            f"{mean + margin:.{places}f}"
        )
    if operation == "significance_decision":
        if len(numeric) != 4:
            raise ValueError(
                "significance_decision reasoning trace requires four inputs"
            )
        mean_left, mean_right, se_left, se_right = numeric
        if se_left < 0 or se_right < 0:
            raise ValueError(
                "significance_decision standard errors cannot be negative"
            )
        pooled = math.sqrt(se_left**2 + se_right**2)
        if pooled == 0:
            raise ValueError(
                "significance_decision pooled standard error cannot be zero"
            )
        threshold = _trace_number(
            parameters.get("threshold"),
            "significance threshold",
        )
        outputs = parameters.get("outputs")
        if threshold <= 0:
            raise ValueError("significance threshold must be positive")
        if (
            not isinstance(outputs, list)
            or len(outputs) != 2
            or any(not isinstance(item, str) or not item for item in outputs)
        ):
            raise ValueError(
                "significance_decision requires two text outputs"
            )
        z_score = (mean_left - mean_right) / pooled
        return outputs[int(abs(z_score) >= threshold)]
    if operation == "significance_claim_consistency":
        if len(numeric) != 5:
            raise ValueError(
                "significance_claim_consistency reasoning trace requires "
                "five inputs"
            )
        mean_left, mean_right, se_left, se_right, claim_value = numeric
        if se_left < 0 or se_right < 0:
            raise ValueError(
                "significance_claim_consistency standard errors cannot be "
                "negative"
            )
        if claim_value not in {0.0, 1.0}:
            raise ValueError(
                "significance_claim_consistency claim must be 0 or 1"
            )
        pooled = math.sqrt(se_left**2 + se_right**2)
        if pooled == 0:
            raise ValueError(
                "significance_claim_consistency pooled standard error "
                "cannot be zero"
            )
        threshold = _trace_number(
            parameters.get("threshold"),
            "significance threshold",
        )
        claim_labels = parameters.get("claim_labels")
        outputs = parameters.get("outputs")
        if threshold <= 0:
            raise ValueError("significance threshold must be positive")
        if (
            not isinstance(claim_labels, list)
            or len(claim_labels) != 2
            or any(
                not isinstance(item, str) or not item
                for item in claim_labels
            )
        ):
            raise ValueError(
                "significance_claim_consistency requires two claim labels"
            )
        if (
            not isinstance(outputs, list)
            or len(outputs) != 2
            or any(not isinstance(item, str) or not item for item in outputs)
        ):
            raise ValueError(
                "significance_claim_consistency requires two text outputs"
            )
        z_score = (mean_left - mean_right) / pooled
        data_claim = int(abs(z_score) >= threshold)
        return outputs[int(data_claim == int(claim_value))]
    if operation in {"argmax", "argmin"}:
        if not numeric:
            raise ValueError(
                f"{operation} reasoning trace requires inputs"
            )
        index = (max if operation == "argmax" else min)(
            range(len(numeric)),
            key=numeric.__getitem__,
        )
        outputs = parameters.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, list) or len(outputs) != len(numeric):
                raise ValueError(
                    f"{operation} reasoning trace outputs must match inputs"
                )
            return outputs[index]
        return labels[index]
    if operation == "weighted_sum":
        weights = parameters.get("weights")
        if not isinstance(weights, list) or len(weights) != len(numeric):
            raise ValueError(
                "weighted_sum reasoning trace requires one weight per input"
            )
        return sum(
            value * _trace_number(weight, "reasoning trace weight")
            for value, weight in zip(numeric, weights)
        )
    raise ValueError(f"unsupported reasoning trace operation {operation!r}")


def _trace_values_equal(left: Any, right: Any) -> bool:
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    return left == right


def _validate_reasoning_trace(raw: Any) -> dict[str, Any]:
    """Validate trace integrity and independently recompute its program result."""

    if not isinstance(raw, dict) or set(raw) != _TRACE_FIELDS:
        raise ValueError(
            "reasoning_trace must contain the complete schema_version 1 trace"
        )
    if raw["schema_version"] != 1:
        raise ValueError("reasoning_trace.schema_version must be 1")
    operation = raw["operation"]
    if operation not in _TRACE_OPERATIONS:
        raise ValueError(f"unsupported reasoning_trace operation {operation!r}")
    inputs = raw["inputs"]
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 64:
        raise ValueError(
            "reasoning_trace.inputs must contain between 1 and 64 inputs"
        )
    input_ids: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("reasoning_trace inputs must be objects")
        input_id = item.get("id")
        input_type = item.get("input_type")
        if (
            not isinstance(input_id, str)
            or not input_id
            or input_id in input_ids
            or input_type not in {"node", "edge"}
        ):
            raise ValueError(
                "reasoning_trace inputs require unique IDs and valid types"
            )
        input_ids.add(input_id)
    if not isinstance(raw["parameters"], dict):
        raise ValueError("reasoning_trace.parameters must be an object")
    if len(_stable_json(raw["parameters"])) > 16_384:
        raise ValueError("reasoning_trace.parameters exceeds the size limit")
    if not isinstance(raw["answer"], str) or not raw["answer"]:
        raise ValueError("reasoning_trace.answer must be a non-empty string")
    answer_value = raw["answer_value"]
    if isinstance(answer_value, bool) or not isinstance(
        answer_value,
        (str, int, float),
    ):
        raise ValueError(
            "reasoning_trace.answer_value must be text or a finite number"
        )
    if isinstance(answer_value, (int, float)):
        _trace_number(answer_value, "reasoning_trace.answer_value")
    facts = raw["required_numeric_facts"]
    if not isinstance(facts, list) or len(facts) > 64:
        raise ValueError(
            "reasoning_trace.required_numeric_facts must be a bounded list"
        )
    for fact in facts:
        if (
            not isinstance(fact, dict)
            or set(fact) != {"value", "percent"}
            or not isinstance(fact["percent"], bool)
        ):
            raise ValueError(
                "reasoning_trace numeric facts require value and percent"
            )
        _trace_number(fact["value"], "reasoning_trace numeric fact")
    fingerprint = raw["trace_fingerprint"]
    payload = {key: value for key, value in raw.items() if key != "trace_fingerprint"}
    expected_fingerprint = hashlib.sha256(
        _stable_json(payload).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("reasoning_trace fingerprint does not match its payload")
    recomputed = _evaluate_reasoning_trace(raw)
    if not _trace_values_equal(recomputed, answer_value):
        raise ValueError(
            "reasoning_trace answer_value does not match the recomputed program"
        )
    return dict(raw)


def _numeric_facts(text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for match in _NUMERIC_LITERAL.finditer(text):
        facts.append(
            {
                "value": float(match.group("value").replace(",", "")),
                "percent": bool(match.group("percent")),
            }
        )
    return facts


def _facts_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_value = float(left["value"])
    right_value = float(right["value"])
    left_percent = bool(left.get("percent"))
    right_percent = bool(right.get("percent"))
    if left_percent == right_percent:
        candidate = left_value
    elif left_percent:
        candidate = left_value / 100.0
    else:
        candidate = left_value
        right_value /= 100.0
    return math.isclose(
        candidate,
        right_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _deduplicate_facts(
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for fact in facts:
        if not any(_facts_match(fact, existing) for existing in unique):
            unique.append(fact)
    return unique


def _allowed_trace_facts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    facts = list(trace["required_numeric_facts"])
    for item in trace["inputs"]:
        if item["input_type"] == "node":
            value = item.get("value")
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
            ):
                facts.append({"value": float(value), "percent": False})
            facts.extend(_numeric_facts(str(item.get("label") or "")))
        else:
            facts.append(
                {
                    "value": _trace_number(
                        item.get("weight"),
                        "reasoning trace edge weight",
                    ),
                    "percent": False,
                }
            )

    def collect(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            facts.append({"value": float(value), "percent": False})
        elif isinstance(value, str):
            facts.extend(_numeric_facts(value))
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(trace["parameters"])
    collect(trace["answer_value"])
    facts.extend(_numeric_facts(trace["answer"]))
    return _deduplicate_facts(facts)


def _program_fact_score(
    rationale: str,
    trace: dict[str, Any],
) -> float:
    predicted = _deduplicate_facts(_numeric_facts(rationale))
    required = _deduplicate_facts(trace["required_numeric_facts"])
    allowed = _allowed_trace_facts(trace)
    recall = (
        sum(
            any(_facts_match(fact, candidate) for candidate in predicted)
            for fact in required
        )
        / len(required)
        if required
        else 1.0
    )
    precision = (
        sum(
            any(_facts_match(fact, candidate) for candidate in allowed)
            for fact in predicted
        )
        / len(predicted)
        if predicted
        else float(not required)
    )
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


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


_FORMULA_STYLE_COMMAND = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|text|operatorname)"
    r"\s*\{([^{}]*)\}"
)
_FORMULA_COMMAND = re.compile(r"\\([A-Za-z]+)")
_FORMULA_COMMANDS = {
    "abs",
    "alpha",
    "arccos",
    "arcsin",
    "arctan",
    "beta",
    "cdot",
    "chi",
    "cos",
    "cosh",
    "cot",
    "csc",
    "delta",
    "dfrac",
    "epsilon",
    "eta",
    "exp",
    "frac",
    "gamma",
    "infty",
    "kappa",
    "lambda",
    "left",
    "ln",
    "log",
    "max",
    "min",
    "mu",
    "nu",
    "omega",
    "phi",
    "pi",
    "psi",
    "rho",
    "right",
    "sec",
    "sigma",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "tau",
    "tfrac",
    "theta",
    "times",
    "upsilon",
    "varepsilon",
    "varphi",
    "varrho",
    "varsigma",
    "vartheta",
    "xi",
    "zeta",
}
_FORMULA_MAX_CHARACTERS = 512
_FORMULA_MAX_COMMANDS = 64
_FORMULA_MAX_NODES = 192
_FORMULA_MAX_OPERATIONS = 96


def _formula_source(value: str) -> str | None:
    source = value.strip()
    source = re.sub(r"^\$+|\$+$", "", source)
    source = re.sub(r"^\\\[|\\\]$", "", source)
    source = source.replace(r"\left", "").replace(r"\right", "")
    source = source.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    source = re.sub(r"\\(?:,|;|!|quad|qquad)\s*", "", source)
    for _ in range(8):
        stripped = _FORMULA_STYLE_COMMAND.sub(r"\1", source)
        if stripped == source:
            break
        source = stripped
    commands = _FORMULA_COMMAND.findall(source)
    if (
        not source
        or len(source) > _FORMULA_MAX_CHARACTERS
        or len(commands) > _FORMULA_MAX_COMMANDS
        or any(command not in _FORMULA_COMMANDS for command in commands)
    ):
        return None
    return source


@lru_cache(maxsize=1024)
def _parse_bounded_formula(value: str) -> Any | None:
    source = _formula_source(value)
    if source is None:
        return None
    try:
        from sympy import (
            Derivative,
            Integral,
            Limit,
            Product,
            Sum,
            count_ops,
            preorder_traversal,
        )
        from sympy.core.basic import Basic
        from sympy.core.function import AppliedUndef
        from sympy.parsing.latex import parse_latex

        expression = parse_latex(source, strict=True, backend="antlr")
    except Exception:
        return None
    if not isinstance(expression, Basic):
        return None
    if expression.has(Derivative, Integral, Limit, Product, Sum, AppliedUndef):
        return None
    if sum(1 for _ in preorder_traversal(expression)) > _FORMULA_MAX_NODES:
        return None
    if int(count_ops(expression, visual=False)) > _FORMULA_MAX_OPERATIONS:
        return None
    if len(expression.free_symbols) > 32:
        return None
    return expression


def _symbolic_expression_equivalent(predicted: Any, gold: Any) -> bool:
    try:
        from sympy import cancel, together, trigsimp

        delta = predicted - gold
    except Exception:
        return False
    if delta == 0 or delta.is_zero is True:
        return True
    for transform in (
        together,
        lambda value: cancel(together(value)),
        trigsimp,
    ):
        try:
            candidate = transform(delta)
        except Exception:
            continue
        if candidate == 0 or candidate.is_zero is True:
            return True
    return False


def _symbolic_equation_equivalent(predicted: Any, gold: Any) -> bool:
    try:
        from sympy import cancel
        from sympy.core.relational import Equality

        if not isinstance(predicted, Equality) or not isinstance(gold, Equality):
            return False
        predicted_residual = predicted.lhs - predicted.rhs
        gold_residual = gold.lhs - gold.rhs
        if _symbolic_expression_equivalent(predicted_residual, gold_residual):
            return True
        ratio = cancel(predicted_residual / gold_residual)
    except Exception:
        return False
    return (
        not ratio.free_symbols
        and ratio.is_zero is False
        and ratio.is_finite is not False
    )


def formula_equivalent(predicted: str, gold: str) -> bool:
    """Conservatively verify bounded elementary expression or equation equivalence."""

    if _formula_normalize(predicted) == _formula_normalize(gold):
        return True
    predicted_expression = _parse_bounded_formula(predicted)
    gold_expression = _parse_bounded_formula(gold)
    if predicted_expression is None or gold_expression is None:
        return False
    try:
        from sympy.core.relational import Equality
    except ImportError:
        return False
    if isinstance(predicted_expression, Equality) or isinstance(
        gold_expression,
        Equality,
    ):
        return _symbolic_equation_equivalent(
            predicted_expression,
            gold_expression,
        )
    return _symbolic_expression_equivalent(
        predicted_expression,
        gold_expression,
    )


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
            rationale_similarity = (
                semantic_match(
                    response.rationale,
                    [context.gold_rationale],
                )
                if response.rationale
                else 0.0
            )
            components["rationale_text_similarity"] = rationale_similarity
            applicable.add("rationale_text_similarity")
            if config.rationale_verifier == "evidence_semantic":
                components["grounded_rationale_consistency"] = (
                    box_score * rationale_similarity
                )
                applicable.add("grounded_rationale_consistency")
            elif context.reasoning_trace is not None:
                program_fact_score = _program_fact_score(
                    response.rationale,
                    context.reasoning_trace,
                )
                consistency = (
                    box_score * rationale_similarity * program_fact_score
                )
                components["rationale_program_fact_score"] = (
                    program_fact_score
                )
                components["program_trace_consistency"] = consistency
                components["grounded_rationale_consistency"] = consistency
                applicable.update(
                    {
                        "grounded_rationale_consistency",
                        "program_trace_consistency",
                        "rationale_program_fact_score",
                    }
                )
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
                formula_equivalent(response.answer, gold)
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
