"""Requirement-level readiness audit for the end-to-end document VLM goal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..method_catalog import (
    EXPECTED_CATEGORY_COUNTS,
    audit_adopted_method_evidence,
    load_method_catalog,
)
from .sweep import SweepPlan


_REQUIRED_CASES = {
    "hard_table",
    "hard_chart",
    "hard_investment",
    "hard_science",
    "hard_diagram",
    "audit_packet",
    "investment_dossier",
}
_REQUIRED_LANGUAGES = {"en", "es", "ko", "ja", "zh"}
_REQUIRED_SFT_MIX = {
    "extraction_and_spotting",
    "tables_charts_formulas",
    "relational_document_qa",
    "multilingual_and_reading_order",
    "reliability_and_abstention",
}
_REQUIRED_STAGES = {"pretrain", "sft", "rlvr", "evaluate_baseline", "evaluate"}
_LONG_OUTPUT_PATTERNS = {
    "table",
    "html",
    "full",
    "transcription",
    "reading-order",
    "long-context",
    "pubtabnet",
    "omnidoc",
    "latex",
}


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    phase: str,
    status: str,
    evidence: dict[str, Any],
) -> None:
    if status not in {"pass", "pending", "fail"}:
        raise ValueError(f"invalid readiness status: {status}")
    checks.append(
        {
            "id": check_id,
            "phase": phase,
            "status": status,
            "evidence": evidence,
        }
    )


def audit_end_to_end_goal_readiness(
    plan: SweepPlan,
    *,
    repo_root: str | Path,
    method_catalog_path: str | Path,
    method_evidence_path: str | Path,
    synth_config_path: str | Path,
    vision_preflight_path: str | Path,
    language_preflight_path: str | Path,
    pilot_readiness_path: str | Path,
    execution_state_path: str | Path,
    quality_evidence_path: str | Path | None = None,
    promotion_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit implementation, execution, and quality without conflating them."""

    repo = Path(repo_root).resolve()
    catalog_source = Path(method_catalog_path).resolve()
    evidence_source = Path(method_evidence_path).resolve()
    synth_source = Path(synth_config_path).resolve()
    vision_source = Path(vision_preflight_path).resolve()
    language_source = Path(language_preflight_path).resolve()
    readiness_source = Path(pilot_readiness_path).resolve()
    execution_source = Path(execution_state_path).resolve()
    quality_source = (
        None
        if quality_evidence_path is None
        else Path(quality_evidence_path).resolve()
    )
    promotion_source = (
        None
        if promotion_evidence_path is None
        else Path(promotion_evidence_path).resolve()
    )

    methods = load_method_catalog(catalog_source)
    evidence_spec = yaml.safe_load(
        evidence_source.read_text(encoding="utf-8")
    )
    method_evidence = audit_adopted_method_evidence(
        methods,
        evidence_spec,
        repo_root=repo,
    )
    synth = yaml.safe_load(synth_source.read_text(encoding="utf-8"))
    vision_preflight = json.loads(
        vision_source.read_text(encoding="utf-8")
    )
    language_preflight = json.loads(
        language_source.read_text(encoding="utf-8")
    )
    readiness = json.loads(readiness_source.read_text(encoding="utf-8"))
    execution = json.loads(execution_source.read_text(encoding="utf-8"))
    quality_evidence = (
        None
        if quality_source is None
        else json.loads(quality_source.read_text(encoding="utf-8"))
    )
    promotion_evidence = (
        None
        if promotion_source is None
        else json.loads(promotion_source.read_text(encoding="utf-8"))
    )
    variants = {variant.arm_id: variant for variant in plan.variants}
    dual = variants.get("lfm_smol_dual")
    checks: list[dict[str, Any]] = []

    categories: dict[str, int] = {}
    for method in methods:
        category = str(method.get("category") or "")
        categories[category] = categories.get(category, 0) + 1
    _check(
        checks,
        check_id="frontier_method_survey",
        phase="implementation",
        status=(
            "pass"
            if len(methods) == 100
            and categories == EXPECTED_CATEGORY_COUNTS
            and all(
                method.get("benefit")
                and method.get("limitation")
                and method.get("decision")
                and method.get("source")
                for method in methods
            )
            else "fail"
        ),
        evidence={
            "methods": len(methods),
            "categories": dict(sorted(categories.items())),
            "expected_categories": EXPECTED_CATEGORY_COUNTS,
            "benefit_and_limitation_required": True,
        },
    )
    _check(
        checks,
        check_id="adopted_method_executable_evidence",
        phase="implementation",
        status="pass" if method_evidence["status"] == "pass" else "fail",
        evidence={
            "status": method_evidence["status"],
            "adopted_methods": method_evidence["adopted_methods"],
            "certified_methods": method_evidence["certified_methods"],
            "errors": len(method_evidence["errors"]),
            "fingerprint": method_evidence["fingerprint"],
        },
    )

    blueprint = (
        {} if dual is None else dual.plan.resolved_blueprint
    )
    arms = {
        str(arm.get("id"))
        for arm in blueprint.get("initialization_arms", [])
        if isinstance(arm, dict)
    }
    parameters = {} if dual is None else dual.parameters
    student = blueprint.get("student") or {}
    _check(
        checks,
        check_id="sub1b_multimodal_architecture",
        phase="implementation",
        status=(
            "pass"
            if dual is not None
            and 0 < int(parameters.get("total") or 0) < 1_000_000_000
            and isinstance(student.get("vision"), dict)
            and isinstance(student.get("connector"), dict)
            and isinstance(student.get("language"), dict)
            and {
                "I0_random",
                "I8_lfm_aligned_language",
                "I9_lfm_smol_dual",
            }.issubset(arms)
            else "fail"
        ),
        evidence={
            "parameters": parameters,
            "has_vision": isinstance(student.get("vision"), dict),
            "has_connector": isinstance(student.get("connector"), dict),
            "has_language": isinstance(student.get("language"), dict),
            "initialization_arms": sorted(arms),
        },
    )

    experiment = {} if dual is None else dual.plan.raw_spec
    cases = set(experiment.get("synthetic", {}).get("cases") or [])
    languages = set(synth.get("base", {}).get("languages") or [])
    sft_mix = set(
        blueprint.get("training", {})
        .get("posttraining", {})
        .get("sft", {})
        .get("data_mix", {})
    )
    _check(
        checks,
        check_id="hard_document_data_coverage",
        phase="implementation",
        status=(
            "pass"
            if _REQUIRED_CASES.issubset(cases)
            and _REQUIRED_LANGUAGES.issubset(languages)
            and _REQUIRED_SFT_MIX.issubset(sft_mix)
            and synth.get("base", {}).get("emit_spotting") is True
            and synth.get("base", {}).get("emit_understanding") is True
            and synth.get("base", {}).get(
                "emit_counterfactual_pairs"
            )
            is True
            else "fail"
        ),
        evidence={
            "hard_cases": sorted(cases & _REQUIRED_CASES),
            "required_hard_cases": sorted(_REQUIRED_CASES),
            "languages": sorted(languages),
            "sft_task_mixtures": sorted(sft_mix),
            "spotting": synth.get("base", {}).get("emit_spotting"),
            "understanding": synth.get("base", {}).get(
                "emit_understanding"
            ),
            "counterfactual_pairs": synth.get("base", {}).get(
                "emit_counterfactual_pairs"
            ),
        },
    )

    stage_names = {
        arm: set(variant.plan.stage_names)
        for arm, variant in variants.items()
    }
    posttraining = (
        blueprint.get("training", {}).get("posttraining", {})
    )
    rlvr = posttraining.get("rlvr") or {}
    reward_mix = rlvr.get("reward_mix") or {}
    _check(
        checks,
        check_id="two_stage_training_and_rlvr",
        phase="implementation",
        status=(
            "pass"
            if set(variants) == {"lfm_language_only", "lfm_smol_dual"}
            and all(
                _REQUIRED_STAGES.issubset(names)
                for names in stage_names.values()
            )
            and posttraining.get("sft", {}).get("objective")
            and rlvr.get("algorithm") == "grpo"
            and {
                "answer_correctness",
                "box_iou",
                "table_tree_similarity",
                "chart_numeric_tolerance",
                "formula_equivalence",
                "grounded_rationale_consistency",
            }.issubset(reward_mix)
            else "fail"
        ),
        evidence={
            "stages_by_arm": {
                arm: sorted(names & _REQUIRED_STAGES)
                for arm, names in stage_names.items()
            },
            "sft_objective": posttraining.get("sft", {}).get("objective"),
            "rlvr_algorithm": rlvr.get("algorithm"),
            "reward_components": sorted(reward_mix),
        },
    )

    vision_transfer = vision_preflight.get("transfer") or {}
    language_transfer = language_preflight.get("transfer") or {}
    _check(
        checks,
        check_id="executed_selective_weight_transfer",
        phase="implementation",
        status=(
            "pass"
            if vision_preflight.get("quality_claim_authorized") is False
            and vision_preflight.get("promotion_claim_authorized") is False
            and vision_transfer.get("value_verified") is True
            and vision_transfer.get("vision_scope") == "transformer_blocks"
            and int(vision_transfer.get("shape_skips") or 0) == 0
            and int(vision_transfer.get("semantic_skips") or 0) == 0
            and language_preflight.get("quality_claim_authorized") is False
            and language_transfer.get("value_verified") is True
            and int(language_transfer.get("shape_skips") or 0) == 0
            and int(language_transfer.get("semantic_skips") or 0) == 0
            else "fail"
        ),
        evidence={
            "vision": {
                "copied_parameters": vision_transfer.get(
                    "copied_parameters"
                ),
                "coverage": vision_transfer.get(
                    "realized_component_parameter_fraction"
                ),
                "scope": vision_transfer.get("vision_scope"),
                "value_verified": vision_transfer.get("value_verified"),
            },
            "language": {
                "copied_parameters": language_transfer.get(
                    "copied_parameters"
                ),
                "coverage": language_transfer.get(
                    "realized_component_parameter_fraction"
                ),
                "value_verified": language_transfer.get("value_verified"),
            },
        },
    )

    evaluation = experiment.get("evaluation") or {}
    generation_gate = next(
        (
            gate
            for gate in blueprint.get("evaluation_gates", [])
            if gate.get("id") == "generation_stability"
        ),
        {},
    )
    patterns = set(generation_gate.get("answer_type_patterns") or [])
    budgets = evaluation.get("max_new_tokens_by_answer_type") or {}
    _check(
        checks,
        check_id="structured_generation_safeguards",
        phase="implementation",
        status=(
            "pass"
            if evaluation.get("max_new_tokens_hard_cap") == 512
            and evaluation.get("repetition_guard_min_tokens") == 24
            and evaluation.get("repetition_guard_max_period") == 16
            and evaluation.get("repetition_guard_repetitions") == 3
            and budgets.get("table*") == 512
            and budgets.get("recognition_fullpage") == 512
            and _LONG_OUTPUT_PATTERNS.issubset(patterns)
            and generation_gate.get(
                "max_degenerate_repetition_rate"
            )
            == 0.0
            else "fail"
        ),
        evidence={
            "hard_cap": evaluation.get("max_new_tokens_hard_cap"),
            "repetition_guard": {
                "min_tokens": evaluation.get(
                    "repetition_guard_min_tokens"
                ),
                "max_period": evaluation.get(
                    "repetition_guard_max_period"
                ),
                "repetitions": evaluation.get(
                    "repetition_guard_repetitions"
                ),
            },
            "long_output_patterns": sorted(patterns),
            "max_degenerate_repetition_rate": generation_gate.get(
                "max_degenerate_repetition_rate"
            ),
        },
    )

    _check(
        checks,
        check_id="pilot_submission_contract",
        phase="implementation",
        status=(
            "pass"
            if readiness.get("overall_status") == "pass"
            and readiness.get("pilot_submission_authorized") is True
            and readiness.get("quality_claim_authorized") is False
            and readiness.get(
                "target_cuda_feasibility_claim_authorized"
            )
            is False
            else "fail"
        ),
        evidence={
            "status": readiness.get("overall_status"),
            "checks": readiness.get("counts"),
            "fingerprint": readiness.get("fingerprint"),
            "pilot_submission_authorized": readiness.get(
                "pilot_submission_authorized"
            ),
            "quality_claim_authorized": readiness.get(
                "quality_claim_authorized"
            ),
        },
    )

    execution_attested = (
        execution.get("state") == "completed_attested"
        and execution.get("training_execution_attested") is True
    )
    _check(
        checks,
        check_id="target_gpu_execution",
        phase="execution",
        status="pass" if execution_attested else "pending",
        evidence={
            "state": execution.get("state"),
            "training_execution_attested": execution.get(
                "training_execution_attested"
            ),
            "observation_scope": execution.get("observation_scope"),
        },
    )

    quality_authorized = bool(
        isinstance(quality_evidence, dict)
        and quality_evidence.get("quality_claim_authorized") is True
    )
    quality_valid = bool(
        quality_authorized
        and quality_evidence.get("claim_scope")
        == "heldout_quality_evidence"
        and quality_evidence.get("gate_status") == "pass"
        and str(
            quality_evidence.get("execution_attestation_fingerprint")
            or ""
        ).startswith("sha256:")
    )
    quality_status = (
        "pending"
        if quality_evidence is None
        else "pass"
        if execution_attested and quality_valid
        else "fail"
    )
    _check(
        checks,
        check_id="heldout_quality_evidence",
        phase="quality",
        status=quality_status,
        evidence={
            "execution_attested": execution_attested,
            "evidence_present": quality_evidence is not None,
            "claim_scope": (
                None
                if quality_evidence is None
                else quality_evidence.get("claim_scope")
            ),
            "gate_status": (
                None
                if quality_evidence is None
                else quality_evidence.get("gate_status")
            ),
            "quality_claim_authorized": quality_authorized,
        },
    )

    promotion_authorized = bool(
        isinstance(promotion_evidence, dict)
        and promotion_evidence.get("promotion_claim_authorized") is True
    )
    promotion_valid = bool(
        promotion_authorized
        and promotion_evidence.get("claim_scope")
        == "multi_seed_promotion_evidence"
        and int(promotion_evidence.get("replicates") or 0) >= 3
        and promotion_evidence.get("multiplicity_controlled") is True
        and str(
            promotion_evidence.get("quality_evidence_fingerprint") or ""
        ).startswith("sha256:")
    )
    promotion_status = (
        "pending"
        if promotion_evidence is None
        else "pass"
        if quality_status == "pass" and promotion_valid
        else "fail"
    )
    _check(
        checks,
        check_id="multi_seed_promotion_evidence",
        phase="quality",
        status=promotion_status,
        evidence={
            "execution_attested": execution_attested,
            "quality_evidence_passed": quality_status == "pass",
            "evidence_present": promotion_evidence is not None,
            "replicates": (
                None
                if promotion_evidence is None
                else promotion_evidence.get("replicates")
            ),
            "multiplicity_controlled": (
                None
                if promotion_evidence is None
                else promotion_evidence.get("multiplicity_controlled")
            ),
            "promotion_claim_authorized": promotion_authorized,
        },
    )

    implementation = [
        check for check in checks if check["phase"] == "implementation"
    ]
    statuses = [check["status"] for check in checks]
    implementation_ready = all(
        check["status"] == "pass" for check in implementation
    )
    goal_complete = all(status == "pass" for status in statuses)
    if goal_complete:
        overall_status = "complete"
    elif implementation_ready and "fail" not in statuses:
        overall_status = "implementation_ready_execution_pending"
    else:
        overall_status = "not_ready"
    result = {
        "schema_version": 1,
        "claim_scope": "end_to_end_document_vlm_goal_readiness_only",
        "overall_status": overall_status,
        "goal_complete": goal_complete,
        "implementation_ready": implementation_ready,
        "execution_complete": execution_attested,
        "quality_proven": quality_status == "pass",
        "promotion_authorized": promotion_status == "pass",
        "checks": checks,
        "counts": {
            "pass": statuses.count("pass"),
            "pending": statuses.count("pending"),
            "fail": statuses.count("fail"),
        },
        "inputs": {
            "method_catalog": _file_fingerprint(catalog_source),
            "method_evidence": _file_fingerprint(evidence_source),
            "synth_config": _file_fingerprint(synth_source),
            "vision_preflight": _file_fingerprint(vision_source),
            "language_preflight": _file_fingerprint(language_source),
            "pilot_readiness": _file_fingerprint(readiness_source),
            "execution_state": _file_fingerprint(execution_source),
            **(
                {"quality_evidence": _file_fingerprint(quality_source)}
                if quality_source is not None
                else {}
            ),
            **(
                {
                    "promotion_evidence": _file_fingerprint(
                        promotion_source
                    )
                }
                if promotion_source is not None
                else {}
            ),
            "sweep_fingerprint": plan.fingerprint,
        },
        "next_required_evidence": [
            check["id"]
            for check in checks
            if check["status"] != "pass"
        ],
        "limitations": [
            "Implementation readiness is not training execution.",
            "One-seed screening cannot authorize model promotion.",
            "Quality requires sealed heldout results from target hardware.",
        ],
    }
    result["fingerprint"] = _fingerprint(result)
    return result
