from __future__ import annotations

import copy
import json
from pathlib import Path

from docvlm_eval.student.architecture_commonality import (
    load_architecture_catalog,
)
from docvlm_eval.student.source_selection import (
    build_source_selection_matrix,
    validate_source_selection_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def _read(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _build():
    return build_source_selection_matrix(
        _read("small_vlm_architecture_commonality.json"),
        _read("small_vlm_weight_commonality.json"),
        load_architecture_catalog(
            ROOT / "configs" / "small_vlm_architectures.yaml"
        ),
        real_payload_preflight=_read(
            "selective_transfer_lfm_real_source_preflight.json"
        ),
    )


def _source(report, target_id, source_id):
    target = next(
        item for item in report["targets"]
        if item["target"] == target_id
    )
    return next(
        item for item in target["sources"]
        if item["source"] == source_id
    )


def _decision(source, component):
    return next(
        item for item in source["decisions"]
        if item["component"] == component
    )


def test_matrix_selects_lfm_only_under_aligned_operator_contract():
    report = _build()
    native = _source(report, "docvlm-800m", "lfm2.5-vl-1.6b")
    aligned = _source(
        report,
        "docvlm-lfm-aligned-814m",
        "lfm2.5-vl-1.6b",
    )

    assert _decision(native, "language.attention")["action"] == (
        "feature_or_relation_distillation"
    )
    assert _decision(aligned, "language.attention")["action"] == (
        "direct_copy_candidate"
    )
    assert _decision(aligned, "language.mlp")["action"] == (
        "structured_transfer_candidate"
    )
    assert _decision(
        aligned, "language.short_convolution"
    )["action"] == "direct_copy_candidate"
    assert _decision(
        aligned, "language.token_embeddings"
    )["action"] == "token_identity_map_required"
    assert aligned["real_payload_evidence"]["status"] == "verified"


def test_matrix_keeps_population_statistics_out_of_copy_authority():
    report = _build()
    aligned = _source(
        report,
        "docvlm-lfm-aligned-814m",
        "lfm2.5-vl-1.6b",
    )
    attention = _decision(aligned, "language.attention")

    assert attention["population_unstable_roles"]
    assert attention["population_prior_authorizes_copy"] is False
    assert report["decision_contract"][
        "population_statistics_establish_basis_alignment"
    ] is False
    assert report["quality_claim_authorized"] is False
    assert report["promotion_claim_authorized"] is False


def test_missing_position_weight_role_requires_payload_preflight():
    report = _build()
    internvl = _source(report, "docvlm-800m", "internvl3-1b")

    assert _decision(internvl, "vision.position")["action"] == (
        "pairwise_payload_preflight_required"
    )


def test_matrix_validation_rejects_weakened_claim_boundary():
    report = _build()
    assert validate_source_selection_matrix(report)["status"] == "pass"

    tampered = copy.deepcopy(report)
    tampered["decision_contract"][
        "population_statistics_establish_basis_alignment"
    ] = True
    assert validate_source_selection_matrix(tampered)["status"] == "fail"


def test_matrix_validation_rejects_stale_upstream_evidence():
    report = _build()
    architecture = _read("small_vlm_architecture_commonality.json")
    architecture["default_target"]["prevalence_threshold"] = 0.75

    audit = validate_source_selection_matrix(
        report,
        architecture_report=architecture,
        weight_report=_read("small_vlm_weight_commonality.json"),
        profiles=load_architecture_catalog(
            ROOT / "configs" / "small_vlm_architectures.yaml"
        ),
        real_payload_preflight=_read(
            "selective_transfer_lfm_real_source_preflight.json"
        ),
    )

    assert audit["status"] == "fail"
    assert "stale architecture_report_fingerprint" in audit["errors"]
