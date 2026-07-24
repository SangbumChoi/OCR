"""Executable latent document graphs for exact synthetic reasoning supervision.

The graph is authored before pixels. Renderers consume its node values, while questions are
resolved by a small deterministic operation registry. This keeps hard answers, rationales, and
evidence references tied to the same source instead of duplicating arithmetic in templates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .hard_locale import hard_text


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


_NUMERIC_LITERAL = re.compile(
    r"(?<![\w.])(?P<currency>[$€£¥₩])?"
    r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<percent>\s*%)?"
)


def _numeric_facts(text: str) -> list[dict[str, Any]]:
    """Extract ordered numeric facts without losing explicit percentage semantics."""

    facts: list[dict[str, Any]] = []
    for match in _NUMERIC_LITERAL.finditer(text):
        value = float(match.group("value").replace(",", ""))
        facts.append(
            {
                "value": value,
                "percent": bool(match.group("percent")),
            }
        )
    return facts


@dataclass(frozen=True)
class GraphNode:
    """A rendered or latent fact in a document."""

    node_id: str
    kind: str
    value: Any
    label: str = ""
    unit: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    """A typed relation, optionally carrying a numeric weight such as ownership."""

    edge_id: str
    source: str
    relation: str
    target: str
    weight: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphQuery:
    """A deterministic query over graph nodes or edges."""

    query_id: str
    question: str
    operation: str
    inputs: tuple[str, ...]
    answer_type: str
    metric: str = "relaxed_acc"
    evidence: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    answer_format: str = "decimal"
    expected: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "inputs": list(self.inputs),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ResolvedQuery:
    """Gold answer and explanation recomputed from a :class:`LatentDocumentGraph`."""

    query_id: str
    answer: str
    answer_value: Any
    rationale: str
    evidence_keys: tuple[str, ...]
    reasoning_trace: dict[str, Any]


@dataclass(frozen=True)
class DifficultySpec:
    """Machine-readable curriculum coordinates for one generated document."""

    level: int
    reasoning_hops: int
    distractor_count: int
    visual_density: float
    cross_region: bool
    skills: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 5:
            raise ValueError("difficulty level must be within [1, 5]")
        if self.reasoning_hops < 1:
            raise ValueError("reasoning_hops must be positive")
        if self.distractor_count < 0:
            raise ValueError("distractor_count cannot be negative")
        if not 0.0 <= self.visual_density <= 1.0:
            raise ValueError("visual_density must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["skills"] = list(self.skills)
        return data


class LatentDocumentGraph:
    """Validated graph whose queries are executable without a teacher model."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        graph_id: str,
        template_family: str,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge] = (),
        queries: Iterable[GraphQuery] = (),
        metadata: dict[str, Any] | None = None,
        language: str = "en",
    ):
        self.graph_id = graph_id
        self.template_family = template_family
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.queries = tuple(queries)
        self.metadata = dict(metadata or {})
        self.language = language
        self._nodes = {node.node_id: node for node in self.nodes}
        self._edges = {edge.edge_id: edge for edge in self.edges}
        self._queries = {query.query_id: query for query in self.queries}
        self.validate()

    def validate(self) -> None:
        """Fail closed on dangling references, duplicate IDs, and stale expected answers."""

        if not self.graph_id.strip() or not self.template_family.strip():
            raise ValueError("graph_id and template_family are required")
        if len(self._nodes) != len(self.nodes):
            raise ValueError("graph node IDs must be unique")
        if len(self._edges) != len(self.edges):
            raise ValueError("graph edge IDs must be unique")
        if len(self._queries) != len(self.queries):
            raise ValueError("graph query IDs must be unique")
        for edge in self.edges:
            if edge.source not in self._nodes or edge.target not in self._nodes:
                raise ValueError(f"edge {edge.edge_id!r} has a dangling endpoint")
            if edge.weight is not None:
                _number(edge.weight, f"edge {edge.edge_id} weight")
        for query in self.queries:
            resolved = self.resolve(query.query_id)
            if query.expected is not None and resolved.answer != query.expected:
                raise ValueError(
                    f"query {query.query_id!r} expected {query.expected!r}, "
                    f"but recomputed {resolved.answer!r}"
                )

    def node(self, node_id: str) -> GraphNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown graph node {node_id!r}") from exc

    def edge(self, edge_id: str) -> GraphEdge:
        try:
            return self._edges[edge_id]
        except KeyError as exc:
            raise ValueError(f"unknown graph edge {edge_id!r}") from exc

    def _node_values(self, query: GraphQuery) -> list[Any]:
        return [self.node(node_id).value for node_id in query.inputs]

    def _evaluate(self, query: GraphQuery) -> tuple[Any, str]:
        edge_operation = query.operation in {
            "path_product",
            "sum_products",
        }
        values = [] if edge_operation else self._node_values(query)
        labels = [] if edge_operation else [
            self.node(node_id).label or node_id for node_id in query.inputs
        ]

        if query.operation == "value":
            if len(values) != 1:
                raise ValueError("value requires exactly one node")
            return values[0], hard_text(
                self.language,
                "r_read",
                label=labels[0],
                value=values[0],
            )
        if query.operation in {
            "sum",
            "mean",
            "difference",
            "ratio",
            "percent_change",
            "relative_reduction",
        }:
            nums = [_number(value, label) for value, label in zip(values, labels)]
            if not nums:
                raise ValueError(f"{query.operation} requires at least one node")
            if query.operation == "sum":
                value = sum(nums)
                return value, hard_text(
                    self.language,
                    "r_add",
                    values=", ".join(str(v) for v in nums),
                    result=f"{value:g}",
                )
            if query.operation == "mean":
                value = sum(nums) / len(nums)
                return value, hard_text(
                    self.language,
                    "r_average",
                    values=" + ".join(str(v) for v in nums),
                    count=len(nums),
                    result=f"{value:g}",
                )
            if len(nums) != 2:
                raise ValueError(f"{query.operation} requires exactly two nodes")
            if query.operation == "difference":
                value = nums[0] - nums[1]
                return value, hard_text(
                    self.language,
                    "r_subtract",
                    right_label=labels[1],
                    right=f"{nums[1]:g}",
                    left_label=labels[0],
                    left=f"{nums[0]:g}",
                    result=f"{value:g}",
                )
            if nums[1] == 0:
                raise ValueError(f"{query.operation} denominator cannot be zero")
            if query.operation == "ratio":
                value = nums[0] / nums[1]
                return value, hard_text(
                    self.language,
                    "r_divide",
                    left=f"{nums[0]:g}",
                    right=f"{nums[1]:g}",
                    result=f"{value:g}",
                )
            if query.operation == "relative_reduction":
                value = (nums[1] - nums[0]) / abs(nums[1]) * 100.0
                return value, hard_text(
                    self.language,
                    "r_reduction",
                    right=f"{nums[1]:g}",
                    left=f"{nums[0]:g}",
                    denominator=f"{abs(nums[1]):g}",
                    result=f"{value:g}",
                )
            value = (nums[0] - nums[1]) / abs(nums[1]) * 100.0
            return value, hard_text(
                self.language,
                "r_change",
                left=f"{nums[0]:g}",
                right=f"{nums[1]:g}",
                denominator=f"{abs(nums[1]):g}",
                result=f"{value:g}",
            )
        if query.operation in {"argmax", "argmin"}:
            if not values:
                raise ValueError(f"{query.operation} requires at least one node")
            nums = [_number(value, label) for value, label in zip(values, labels)]
            index = (max if query.operation == "argmax" else min)(
                range(len(nums)), key=nums.__getitem__
            )
            output = query.parameters.get("outputs")
            answer = output[index] if isinstance(output, list) else labels[index]
            direction = hard_text(
                self.language,
                "largest" if query.operation == "argmax" else "smallest",
            )
            return answer, hard_text(
                self.language,
                "r_extreme",
                label=labels[index],
                direction=direction,
                value=f"{nums[index]:g}",
            )
        if query.operation == "weighted_sum":
            weights = query.parameters.get("weights")
            if not isinstance(weights, list) or len(weights) != len(values):
                raise ValueError("weighted_sum requires one weight per input node")
            nums = [_number(value, label) for value, label in zip(values, labels)]
            ws = [_number(weight, "weight") for weight in weights]
            terms = [value * weight for value, weight in zip(nums, ws)]
            total = sum(terms)
            return total, hard_text(
                self.language,
                "r_weighted",
                terms=" + ".join(
                    f"{value:g} x {weight:g}"
                    for value, weight in zip(nums, ws)
                ),
                result=f"{total:g}",
            )
        if query.operation == "path_product":
            edges = [self.edge(edge_id) for edge_id in query.inputs]
            weights = [_number(edge.weight, f"edge {edge.edge_id} weight") for edge in edges]
            value = math.prod(weights)
            return value, hard_text(
                self.language,
                "r_path",
                weights=" x ".join(f"{weight:g}" for weight in weights),
                result=f"{value:g}",
            )
        if query.operation == "sum_products":
            paths = query.parameters.get("paths")
            if not isinstance(paths, list) or not paths:
                raise ValueError("sum_products requires a non-empty paths list")
            products: list[float] = []
            path_text: list[str] = []
            for path in paths:
                if not isinstance(path, list) or not path:
                    raise ValueError("each sum_products path must contain edge IDs")
                weights = [
                    _number(self.edge(edge_id).weight, f"edge {edge_id} weight")
                    for edge_id in path
                ]
                products.append(math.prod(weights))
                path_text.append(" x ".join(f"{weight:g}" for weight in weights))
            value = sum(products)
            return value, hard_text(
                self.language,
                "r_paths",
                paths=") + (".join(path_text),
                result=f"{value:g}",
            )
        raise ValueError(f"unsupported graph operation {query.operation!r}")

    @staticmethod
    def _format(value: Any, answer_format: str) -> str:
        if answer_format == "text":
            return str(value)
        number = _number(value, "query result")
        if answer_format == "integer":
            if not math.isclose(number, round(number), abs_tol=1e-9):
                raise ValueError(f"non-integral result {number} cannot use integer format")
            return str(round(number))
        if answer_format == "money":
            return f"${number:,.2f}"
        if answer_format == "percent":
            return f"{number:.2f}%"
        if answer_format == "fraction_percent":
            return f"{number * 100.0:.2f}%"
        if answer_format.startswith("decimal:"):
            places = int(answer_format.split(":", 1)[1])
            return f"{number:.{places}f}"
        if answer_format == "decimal":
            return f"{number:g}"
        raise ValueError(f"unsupported answer format {answer_format!r}")

    def resolve(self, query_id: str) -> ResolvedQuery:
        try:
            query = self._queries[query_id]
        except KeyError as exc:
            raise ValueError(f"unknown graph query {query_id!r}") from exc
        value, rationale = self._evaluate(query)
        evidence_nodes = query.evidence or tuple(
            node_id for node_id in query.inputs if node_id in self._nodes
        )
        evidence_keys = tuple(
            str(self.node(node_id).attributes["field_key"])
            for node_id in evidence_nodes
            if self.node(node_id).attributes.get("field_key")
        )
        answer = self._format(value, query.answer_format)
        trace_inputs: list[dict[str, Any]] = []
        edge_ids: list[str] = []
        if query.operation == "path_product":
            edge_ids.extend(query.inputs)
        elif query.operation == "sum_products":
            paths = query.parameters.get("paths")
            if isinstance(paths, list):
                edge_ids.extend(
                    edge_id
                    for path in paths
                    if isinstance(path, list)
                    for edge_id in path
                    if isinstance(edge_id, str)
                )
        if edge_ids:
            for edge_id in dict.fromkeys(edge_ids):
                edge = self.edge(edge_id)
                trace_inputs.append(
                    {
                        "id": edge.edge_id,
                        "input_type": "edge",
                        "edge_source": edge.source,
                        "relation": edge.relation,
                        "edge_target": edge.target,
                        "weight": edge.weight,
                    }
                )
        else:
            for node_id in query.inputs:
                node = self.node(node_id)
                trace_inputs.append(
                    {
                        "id": node.node_id,
                        "input_type": "node",
                        "kind": node.kind,
                        "label": node.label,
                        "value": node.value,
                        "unit": node.unit,
                    }
                )
        reasoning_trace = {
            "schema_version": 1,
            "operation": query.operation,
            "inputs": trace_inputs,
            "parameters": query.parameters,
            "answer_value": value,
            "answer": answer,
            "required_numeric_facts": _numeric_facts(rationale),
        }
        reasoning_trace["trace_fingerprint"] = _hash(reasoning_trace)
        return ResolvedQuery(
            query_id=query.query_id,
            answer=answer,
            answer_value=value,
            rationale=rationale,
            evidence_keys=evidence_keys,
            reasoning_trace=reasoning_trace,
        )

    def add_questions(self, builder: Any) -> None:
        """Attach every graph query to a compatible ``DocBuilder``."""

        for query in self.queries:
            resolved = self.resolve(query.query_id)
            builder.qa(
                query.question,
                resolved.answer,
                metric=query.metric,
                answer_type=query.answer_type,
                rationale=resolved.rationale,
                languages=[self.language],
                evidence_keys=list(resolved.evidence_keys),
                derived=True,
                graph_query_id=query.query_id,
                reasoning_trace=resolved.reasoning_trace,
            )
        builder.semantic_graph = self.to_dict()

    @property
    def template_fingerprint(self) -> str:
        """Hash graph topology and query programs while intentionally excluding content values."""

        skeleton = {
            "schema_version": self.SCHEMA_VERSION,
            "template_family": self.template_family,
            "nodes": [
                {
                    "id": node.node_id,
                    "kind": node.kind,
                    "unit": node.unit,
                    "attribute_keys": sorted(node.attributes),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                    "weighted": edge.weight is not None,
                }
                for edge in self.edges
            ],
            "queries": [
                {
                    "id": query.query_id,
                    "operation": query.operation,
                    "inputs": query.inputs,
                    "answer_type": query.answer_type,
                    "answer_format": query.answer_format,
                    "parameter_shape": _parameter_shape(query.parameters),
                }
                for query in self.queries
            ],
        }
        return _hash(skeleton)

    @property
    def content_fingerprint(self) -> str:
        """Hash the complete latent semantics, independent of rendering noise."""

        return _hash(
            {
                "template_family": self.template_family,
                "nodes": [node.to_dict() for node in self.nodes],
                "edges": [edge.to_dict() for edge in self.edges],
                "queries": [query.to_dict() for query in self.queries],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "template_family": self.template_family,
            "template_fingerprint": self.template_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "language": self.language,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "queries": [
                {
                    **query.to_dict(),
                    "resolved": asdict(self.resolve(query.query_id)),
                }
                for query in self.queries
            ],
            "metadata": self.metadata,
        }


def _parameter_shape(value: Any) -> Any:
    """Preserve program structure for template hashing without retaining authored values."""

    if isinstance(value, dict):
        return {key: _parameter_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_parameter_shape(item) for item in value]
    return type(value).__name__
