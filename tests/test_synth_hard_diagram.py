"""Programmatic hard diagrams keep topology, labels, and path programs aligned."""

from __future__ import annotations

import random

import numpy as np
import pytest

from docvlm_eval.synth.dto import DocSample
from docvlm_eval.synth.hard_diagram import hard_process_diagram
from docvlm_eval.synth.quality import audit_render_evidence
from docvlm_eval.synth.to_samples import case_to_samples


@pytest.mark.parametrize(
    ("level", "query_count"),
    [(1, 1), (2, 2), (3, 3), (4, 5), (5, 6)],
)
def test_diagram_curriculum_emits_executable_queries(level, query_count):
    image, ground_truth = hard_process_diagram(
        random.Random(17),
        level=level,
    )
    graph = ground_truth["semantic_graph"]

    assert len(graph["queries"]) == query_count
    assert ground_truth["difficulty"]["level"] == level
    assert all(query["resolved"]["answer"] for query in graph["queries"])
    assert all(query["resolved"]["rationale"] for query in graph["queries"])
    assert "input_gate:" not in ground_truth["fields"]["full_text"]
    assert np.asarray(image).std() > 10


def test_diagram_layouts_preserve_semantics_but_change_geometry():
    outputs = {
        family: hard_process_diagram(
            random.Random(31),
            level=5,
            layout_family=family,
        )
        for family in ("classic-v1", "compact-v1", "report-v1")
    }
    fingerprints = {
        ground_truth["semantic_graph"]["content_fingerprint"]
        for _, ground_truth in outputs.values()
    }
    template_fingerprints = {
        ground_truth["semantic_graph"]["template_fingerprint"]
        for _, ground_truth in outputs.values()
    }
    sizes = {image.size for image, _ in outputs.values()}
    layout_fingerprints = {
        ground_truth["render"]["layout_fingerprint"]
        for _, ground_truth in outputs.values()
    }

    assert len(fingerprints) == 1
    assert len(template_fingerprints) == 1
    assert len(sizes) == 3
    assert len(layout_fingerprints) == 3


def test_diagram_spatial_evidence_is_visible_and_reaches_samples():
    image, ground_truth = hard_process_diagram(
        random.Random(43),
        level=5,
        layout_family="classic-v1",
    )
    structured = DocSample.from_builder_gt(ground_truth).to_dict()
    report = audit_render_evidence(
        image,
        structured,
        required_spotting_keys=ground_truth["render"][
            "required_spotting_keys"
        ],
    )
    samples = case_to_samples(
        structured,
        "diagram.png",
        "hard_diagram",
        render_variant="clean",
    )
    expected = next(
        sample
        for sample in samples
        if sample.answer_type == "H-diagram-expected-count"
    )
    topology = next(
        sample
        for sample in samples
        if sample.answer_type == "T-diagram-topology"
    )

    assert report["status"] == "passed"
    assert report["required_spotting_keys"] == len(
        ground_truth["render"]["required_spotting_keys"]
    )
    assert expected.meta["evidence_count"] == 7
    assert len(expected.meta["boxes"]) == 7
    assert expected.meta["reasoning_trace"]["operation"] == "weighted_sum"
    assert len(
        expected.meta["reasoning_trace"]["trace_fingerprint"]
    ) == 64
    assert topology.answers == ["Fusion Review"]
    assert topology.meta["evidence_count"] == 1
