"""Ablation-safe projection of rich synthetic ground truth."""

from __future__ import annotations

from typing import Any


def apply_supervision_toggles(gt: dict[str, Any], cfg: Any) -> None:
    """Remove every alternate path to supervision disabled by an ablation arm.

    The graph is richer than the legacy QA mirror, so toggles must redact both views. In
    particular, removing only ``spotting`` is insufficient: a grounding QA can still encode the
    same box in its answer.
    """

    if not cfg.emit_spotting:
        gt.pop("spotting", None)
        gt["qa"] = [
            query
            for query in gt.get("qa", [])
            if query.get("metric") != "grounding"
        ]
        for query in gt.get("qa", []):
            query.pop("box", None)
            query.pop("evidence_keys", None)
        for query in (gt.get("semantic_graph") or {}).get("queries", []):
            (query.get("resolved") or {}).pop("evidence_keys", None)

    if not cfg.emit_rationale:
        for query in gt.get("qa", []):
            query.pop("rationale", None)
        graph = gt.get("semantic_graph")
        if graph:
            provenance_keys = {
                "schema_version",
                "graph_id",
                "language",
                "template_family",
                "template_fingerprint",
                "content_fingerprint",
                "difficulty",
            }
            gt["semantic_graph"] = {
                key: value for key, value in graph.items() if key in provenance_keys
            }
            gt["semantic_graph"]["supervision_redacted"] = True

    if not getattr(cfg, "emit_understanding", True):
        gt["qa"] = [query for query in gt.get("qa", []) if not query.get("derived")]
        gt.pop("semantic_graph", None)
