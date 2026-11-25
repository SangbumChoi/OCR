from __future__ import annotations

import copy

import pytest

from docvlm_eval.student.confirmatory_evidence import (
    build_confirmatory_evidence,
)


SWEEP = "docvlm-smol-vision-transfer-sweep"


def _comparison():
    replicates = ["seed_0", "seed_1", "seed_2"]
    attestations = {
        f"{arm}--{replicate}": {
            "attestation_sha256": "sha256:" + ("a" * 64),
            "claim_scope": "execution_contract_only",
            "contract_status": "pass",
            "quality_claim_authorized": False,
            "stage_count": 25,
        }
        for arm in ("lfm_language_only", "lfm_smol_dual")
        for replicate in replicates
    }
    gates = {
        gate: "pass"
        for gate in (
            "parameter_budget",
            "training_feasibility",
            "generalization",
            "grounding",
            "reasoning",
            "multilingual",
            "reliability",
            "generation_stability",
        )
    }
    return {
        "schema_version": 6,
        "sweep": SWEEP,
        "sweep_fingerprint": "sha256:" + ("b" * 64),
        "baseline": "lfm_language_only",
        "replicates": replicates,
        "execution_attestations": attestations,
        "variants": {
            "lfm_smol_dual": {
                "gate_status": "pass",
                "gate_statuses": gates,
            }
        },
        "promotion": {
            "status": "promote",
            "selected_variants": ["lfm_smol_dual"],
            "contract": {
                "minimum_replicates": 3,
                "required_gates": list(gates),
            },
            "multiple_comparisons": {
                "method": (
                    "bonferroni_one_sided_percentile_bootstrap"
                ),
                "comparison_count": 8,
                "familywise_alpha": 0.05,
            },
        },
    }


def test_confirmatory_evidence_authorizes_only_sealed_promoted_run():
    quality, promotion = build_confirmatory_evidence(
        _comparison(),
        expected_sweep=SWEEP,
        expected_sweep_fingerprint="sha256:" + ("b" * 64),
        baseline="lfm_language_only",
        candidate="lfm_smol_dual",
    )

    assert quality["quality_claim_authorized"] is True
    assert quality["gate_status"] == "pass"
    assert quality["replicates"] == 3
    assert promotion["multiplicity_controlled"] is True
    assert promotion["promotion_claim_authorized"] is True


def test_confirmatory_evidence_rejects_unsealed_or_failed_gate():
    comparison = _comparison()
    first = next(iter(comparison["execution_attestations"].values()))
    first["contract_status"] = "fail"
    comparison["variants"]["lfm_smol_dual"]["gate_statuses"][
        "generation_stability"
    ] = "fail"

    quality, promotion = build_confirmatory_evidence(
        comparison,
        expected_sweep=SWEEP,
        expected_sweep_fingerprint="sha256:" + ("b" * 64),
        baseline="lfm_language_only",
        candidate="lfm_smol_dual",
    )

    assert quality["quality_claim_authorized"] is False
    assert quality["unsealed_runs"]
    assert quality["failed_or_missing_gates"] == [
        "generation_stability"
    ]
    assert promotion["promotion_claim_authorized"] is False


def test_confirmatory_evidence_requires_three_unique_replicates():
    comparison = copy.deepcopy(_comparison())
    comparison["replicates"] = ["seed_0", "seed_1"]

    with pytest.raises(ValueError, match="three unique"):
        build_confirmatory_evidence(
            comparison,
            expected_sweep=SWEEP,
            expected_sweep_fingerprint="sha256:" + ("b" * 64),
            baseline="lfm_language_only",
            candidate="lfm_smol_dual",
        )


def test_confirmatory_evidence_accepts_deployment_attestations():
    comparison = _comparison()
    for attestation in comparison["execution_attestations"].values():
        attestation["claim_scope"] = "deployment_capability"
        attestation["quality_claim_authorized"] = True

    quality, promotion = build_confirmatory_evidence(
        comparison,
        expected_sweep=SWEEP,
        expected_sweep_fingerprint="sha256:" + ("b" * 64),
        baseline="lfm_language_only",
        candidate="lfm_smol_dual",
    )

    assert quality["quality_claim_authorized"] is True
    assert promotion["promotion_claim_authorized"] is True


def test_confirmatory_evidence_rejects_unexpected_run():
    comparison = _comparison()
    comparison["execution_attestations"]["other--seed_0"] = copy.deepcopy(
        next(iter(comparison["execution_attestations"].values()))
    )

    quality, promotion = build_confirmatory_evidence(
        comparison,
        expected_sweep=SWEEP,
        expected_sweep_fingerprint="sha256:" + ("b" * 64),
        baseline="lfm_language_only",
        candidate="lfm_smol_dual",
    )

    assert quality["unexpected_runs"] == ["other--seed_0"]
    assert quality["quality_claim_authorized"] is False
    assert promotion["promotion_claim_authorized"] is False
