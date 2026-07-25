"""Fail-closed launch readiness checks for selective-transfer pilots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .architecture_commonality import load_architecture_catalog
from .source_selection import validate_source_selection_matrix
from .sweep import SweepPlan


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
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


def _fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_label(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    evidence: dict[str, Any],
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "pass" if passed else "fail",
            "evidence": evidence,
        }
    )


def _stage_names(variant: Any) -> list[str]:
    return [stage.name for stage in variant.plan.stages]


def _command_option(command: tuple[str, ...], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _evaluation_policy(variant: Any) -> dict[str, Any]:
    evaluation = variant.plan.raw_spec.get("evaluation") or {}
    return {
        "max_new_tokens": evaluation.get("max_new_tokens"),
        "max_new_tokens_hard_cap": evaluation.get(
            "max_new_tokens_hard_cap"
        ),
        "repetition_guard_min_tokens": evaluation.get(
            "repetition_guard_min_tokens"
        ),
        "repetition_guard_max_period": evaluation.get(
            "repetition_guard_max_period"
        ),
        "repetition_guard_repetitions": evaluation.get(
            "repetition_guard_repetitions"
        ),
        "sample_selection": evaluation.get("sample_selection"),
        "max_samples": evaluation.get("max_samples"),
        "max_new_tokens_by_answer_type": evaluation.get(
            "max_new_tokens_by_answer_type"
        )
        or {},
    }


def _public_selection(spec: dict[str, Any]) -> dict[str, Any]:
    components = spec.get("data", {}).get("components")
    if not isinstance(components, list):
        return {}
    for component in components:
        if (
            isinstance(component, dict)
            and isinstance(component.get("hub"), dict)
            and component["hub"].get("max_rows") is not None
        ):
            hub = component["hub"]
            return {
                "max_rows": int(hub["max_rows"]),
                "sampling_strategy": hub.get(
                    "sampling_strategy",
                    "global_hash",
                ),
                "min_rows_per_task": int(
                    hub.get("min_rows_per_task", 0)
                ),
            }
    return {}


def audit_lfm_transfer_pilot(
    plan: SweepPlan,
    *,
    repo_root: str | Path,
    sweep_path: str | Path,
    preflight_path: str | Path,
    source_selection_path: str | Path,
) -> dict[str, Any]:
    """Audit whether the LFM screening pilot is safe to submit.

    Passing this contract authorizes only submission of the screening run. It
    does not authorize a quality, target-CUDA feasibility, or promotion claim.
    """

    repo = Path(repo_root).resolve()
    sweep_source = Path(sweep_path).resolve()
    preflight_source = Path(preflight_path).resolve()
    source_selection_source = Path(source_selection_path).resolve()
    sweep_label = _source_label(sweep_source, repo)
    preflight_label = _source_label(preflight_source, repo)
    source_selection_label = _source_label(source_selection_source, repo)
    preflight = json.loads(preflight_source.read_text(encoding="utf-8"))
    source_selection = json.loads(
        source_selection_source.read_text(encoding="utf-8")
    )
    architecture_report_source = (
        repo / "docs" / "results" / "small_vlm_architecture_commonality.json"
    )
    weight_report_source = (
        repo / "docs" / "results" / "small_vlm_weight_commonality.json"
    )
    architecture_catalog_source = (
        repo / "configs" / "small_vlm_architectures.yaml"
    )
    architecture_report = json.loads(
        architecture_report_source.read_text(encoding="utf-8")
    )
    weight_report = json.loads(
        weight_report_source.read_text(encoding="utf-8")
    )
    architecture_profiles = load_architecture_catalog(
        architecture_catalog_source
    )
    smol_preflight_source = (
        repo
        / "docs"
        / "results"
        / "selective_transfer_smol_vision_real_source_preflight.json"
    )
    smol_preflight = json.loads(
        smol_preflight_source.read_text(encoding="utf-8")
    )
    variants = {variant.arm_id: variant for variant in plan.variants}
    required_arms = {
        "native_random",
        "lfm_random",
        "lfm_strict_transfer",
    }
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "screening_design",
        set(variants) == required_arms
        and plan.baseline == "lfm_random"
        and len(plan.replicates) == 1
        and plan.promotion is None,
        {
            "arms": sorted(variants),
            "baseline": plan.baseline,
            "replicates": list(plan.replicates),
            "promotion_enabled": plan.promotion is not None,
        },
    )

    random_variant = variants.get("lfm_random")
    transfer_variant = variants.get("lfm_strict_transfer")
    aligned_present = random_variant is not None and transfer_variant is not None
    geometry_equal = bool(
        aligned_present
        and random_variant.plan.resolved_blueprint
        == transfer_variant.plan.resolved_blueprint
        and random_variant.parameters == transfer_variant.parameters
    )
    target_parameters = (
        transfer_variant.parameters.get("total")
        if transfer_variant is not None
        else None
    )
    _check(
        checks,
        "sub1b_geometry_matched_control",
        geometry_equal
        and isinstance(target_parameters, int)
        and target_parameters < 1_000_000_000,
        {
            "geometry_equal": geometry_equal,
            "random_parameters": (
                None if random_variant is None else random_variant.parameters
            ),
            "transfer_parameters": (
                None
                if transfer_variant is None
                else transfer_variant.parameters
            ),
        },
    )

    random_arm = (
        None
        if random_variant is None
        else random_variant.plan.raw_spec.get("initialization", {}).get("arm")
    )
    transfer_initialization = (
        {}
        if transfer_variant is None
        else transfer_variant.plan.raw_spec.get("initialization", {})
    )
    transfer_arm = transfer_initialization.get("arm")
    source_hub = (
        transfer_initialization.get("language_source", {}).get("hub", {})
        if isinstance(transfer_initialization.get("language_source"), dict)
        else {}
    )
    source_revision = str(source_hub.get("revision") or "")
    _check(
        checks,
        "initialization_contrast",
        random_arm == "I0_random"
        and transfer_arm == "I8_lfm_aligned_language",
        {
            "random_arm": random_arm,
            "transfer_arm": transfer_arm,
        },
    )
    _check(
        checks,
        "immutable_language_source",
        source_hub.get("repo_id") == "LiquidAI/LFM2.5-VL-1.6B"
        and bool(_IMMUTABLE_REVISION.fullmatch(source_revision)),
        {
            "repo_id": source_hub.get("repo_id"),
            "revision": source_revision,
        },
    )

    preflight_source_identity = preflight.get("source") or {}
    source_matches = (
        preflight.get("claim_scope")
        == "real_source_initialization_contract_only"
        and preflight.get("quality_claim_authorized") is False
        and preflight_source_identity.get("model")
        == source_hub.get("repo_id")
        and preflight_source_identity.get("revision") == source_revision
        and str(
            preflight_source_identity.get("content_fingerprint") or ""
        ).startswith("sha256:")
        and int(preflight_source_identity.get("total_bytes") or 0) > 0
    )
    _check(
        checks,
        "executed_source_identity",
        source_matches,
        {
            "claim_scope": preflight.get("claim_scope"),
            "quality_claim_authorized": preflight.get(
                "quality_claim_authorized"
            ),
            "model": preflight_source_identity.get("model"),
            "revision": preflight_source_identity.get("revision"),
            "content_fingerprint": preflight_source_identity.get(
                "content_fingerprint"
            ),
            "total_bytes": preflight_source_identity.get("total_bytes"),
        },
    )

    preflight_target = preflight.get("target") or {}
    _check(
        checks,
        "executed_target_identity",
        preflight.get("initialization_arm") == transfer_arm
        and preflight_target.get("total_parameters") == target_parameters,
        {
            "initialization_arm": preflight.get("initialization_arm"),
            "target_parameters": preflight_target.get("total_parameters"),
            "compiled_target_parameters": target_parameters,
        },
    )

    transfer = preflight.get("transfer") or {}
    realized_fraction = float(
        transfer.get("realized_component_parameter_fraction") or 0.0
    )
    minimum_fraction = float(
        transfer.get("minimum_component_parameter_fraction") or 1.0
    )
    transfer_integrity = (
        transfer.get("component") == "language"
        and transfer.get("family") == "lfm2"
        and transfer.get("shape_policy") == "structured_mlp"
        and bool(transfer.get("attention_geometry_compatible"))
        and bool(transfer.get("short_convolution_compatible"))
        and bool(transfer.get("mlp_operator_compatible"))
        and not transfer.get("unhealthy_source_weight_roles")
        and int(transfer.get("shape_skips") or 0) == 0
        and int(transfer.get("semantic_skips") or 0) == 0
        and int(transfer.get("missing_source") or 0) == 0
        and bool(transfer.get("value_verified"))
        and realized_fraction >= minimum_fraction
    )
    _check(
        checks,
        "executed_transfer_integrity",
        transfer_integrity,
        {
            "copied_tensors": transfer.get("copied_tensors"),
            "copied_parameters": transfer.get("copied_parameters"),
            "realized_component_parameter_fraction": realized_fraction,
            "minimum_component_parameter_fraction": minimum_fraction,
            "structured_groups": transfer.get("structured_groups"),
            "shape_skips": transfer.get("shape_skips"),
            "semantic_skips": transfer.get("semantic_skips"),
            "missing_source": transfer.get("missing_source"),
            "value_verified": transfer.get("value_verified"),
        },
    )

    source_selection_audit = validate_source_selection_matrix(
        source_selection,
        architecture_report=architecture_report,
        weight_report=weight_report,
        profiles=architecture_profiles,
        real_payload_preflights=[preflight, smol_preflight],
    )
    aligned_selection = next(
        (
            target
            for target in source_selection.get("targets", [])
            if isinstance(target, dict)
            and target.get("target") == "docvlm-lfm-aligned-814m"
        ),
        {},
    )
    lfm_selection = next(
        (
            source
            for source in aligned_selection.get("sources", [])
            if isinstance(source, dict)
            and source.get("model_id") == source_hub.get("repo_id")
            and source.get("revision") == source_revision
        ),
        {},
    )
    action_counts = lfm_selection.get("action_counts") or {}
    payload_evidence = lfm_selection.get("real_payload_evidence") or {}
    _check(
        checks,
        "cross_model_source_selection",
        source_selection_audit["status"] == "pass"
        and source_selection.get("claim_scope")
        == "selective_transfer_source_selection_only"
        and source_selection.get("quality_claim_authorized") is False
        and source_selection.get("promotion_claim_authorized") is False
        and int(action_counts.get("direct_copy_candidate") or 0) >= 2
        and int(
            action_counts.get("structured_transfer_candidate") or 0
        )
        >= 1
        and payload_evidence.get("status") == "verified"
        and payload_evidence.get("source_content_fingerprint")
        == preflight_source_identity.get("content_fingerprint"),
        {
            "target": aligned_selection.get("target"),
            "source": lfm_selection.get("model_id"),
            "source_revision": lfm_selection.get("revision"),
            "direct_copy_candidates": action_counts.get(
                "direct_copy_candidate"
            ),
            "structured_transfer_candidates": action_counts.get(
                "structured_transfer_candidate"
            ),
            "real_payload_status": payload_evidence.get("status"),
            "matrix_validation_status": source_selection_audit["status"],
            "quality_claim_authorized": source_selection.get(
                "quality_claim_authorized"
            ),
            "promotion_claim_authorized": source_selection.get(
                "promotion_claim_authorized"
            ),
        },
    )

    stage_names = {
        arm: _stage_names(variant)
        for arm, variant in variants.items()
    }
    required_training_stages = {
        "acquire_language_checkpoint",
        "initialize_student",
        "pretrain",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
    }
    _check(
        checks,
        "end_to_end_training_topology",
        all(
            required_training_stages.issubset(names)
            for names in stage_names.values()
        ),
        {
            "required_stages": sorted(required_training_stages),
            "stage_counts": {
                arm: len(names) for arm, names in stage_names.items()
            },
        },
    )
    feasibility_arms = sorted(
        arm
        for arm, names in stage_names.items()
        if "training_feasibility_benchmark" in names
    )
    _check(
        checks,
        "strict_only_cuda_preflight",
        feasibility_arms == ["lfm_strict_transfer"],
        {"training_feasibility_arms": feasibility_arms},
    )

    training_gate = (
        transfer_variant.plan.resolved_blueprint.get("evaluation_gates", [])
        if transfer_variant is not None
        else []
    )
    training_gate = next(
        (
            gate
            for gate in training_gate
            if gate.get("id") == "training_feasibility"
        ),
        {},
    )
    _check(
        checks,
        "target_gpu_training_contract",
        training_gate.get("required_device_type") == "cuda"
        and training_gate.get("required_precision") == "bfloat16"
        and training_gate.get(
            "required_resolved_visual_attention_backend"
        )
        == "flex"
        and training_gate.get("require_gradient_checkpointing") is True
        and training_gate.get(
            "required_gradient_checkpointing_components"
        )
        == ["vision", "connector", "language"]
        and training_gate.get(
            "required_gradient_checkpointing_use_reentrant"
        )
        is False,
        {
            "required_device_type": training_gate.get(
                "required_device_type"
            ),
            "required_precision": training_gate.get(
                "required_precision"
            ),
            "required_visual_backend": training_gate.get(
                "required_resolved_visual_attention_backend"
            ),
            "gradient_checkpointing": training_gate.get(
                "require_gradient_checkpointing"
            ),
            "gradient_checkpointing_components": training_gate.get(
                "required_gradient_checkpointing_components"
            ),
            "gradient_checkpointing_use_reentrant": training_gate.get(
                "required_gradient_checkpointing_use_reentrant"
            ),
        },
    )

    tracked_stage_names = {
        "pretrain",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
    }
    tracking: dict[str, dict[str, dict[str, str | None]]] = {}
    tracking_valid = True
    for arm, variant in variants.items():
        arm_tracking = {}
        for stage in variant.plan.stages:
            if (
                stage.name not in tracked_stage_names
                and stage.name != "training_feasibility_benchmark"
            ):
                continue
            values = {
                "project": _command_option(
                    stage.command, "--wandb-project"
                ),
                "entity": _command_option(stage.command, "--wandb-entity"),
                "run": _command_option(stage.command, "--wandb-run"),
                "group": _command_option(stage.command, "--wandb-group"),
            }
            arm_tracking[stage.name] = values
            tracking_valid &= (
                values["project"] == "docvlm-ablation"
                and values["entity"] == "sbdc"
                and bool(values["run"])
                and bool(values["group"])
            )
        expected = set(tracked_stage_names)
        if arm == "lfm_strict_transfer":
            expected.add("training_feasibility_benchmark")
        tracking_valid &= set(arm_tracking) == expected
        tracking[arm] = arm_tracking
    _check(
        checks,
        "observable_wandb_runs",
        tracking_valid,
        {
            "entity": "sbdc",
            "project": "docvlm-ablation",
            "stages_by_arm": {
                arm: sorted(stages) for arm, stages in tracking.items()
            },
        },
    )

    policies = {
        arm: _evaluation_policy(variant)
        for arm, variant in variants.items()
    }
    unique_policies = {
        json.dumps(policy, sort_keys=True)
        for policy in policies.values()
    }
    policy = next(iter(policies.values()), {})
    budgets = policy.get("max_new_tokens_by_answer_type") or {}
    _check(
        checks,
        "matched_long_output_policy",
        len(unique_policies) == 1
        and policy.get("max_new_tokens") == 128
        and policy.get("max_new_tokens_hard_cap") == 512
        and policy.get("repetition_guard_min_tokens") == 24
        and policy.get("repetition_guard_max_period") == 16
        and policy.get("repetition_guard_repetitions") == 3
        and policy.get("sample_selection") == "answer_type_round_robin"
        and budgets.get("table*") == 512
        and budgets.get("recognition_fullpage") == 512,
        {
            "policy": policy,
            "matched_across_arms": len(unique_policies) == 1,
        },
    )

    generation_gate = (
        transfer_variant.plan.resolved_blueprint.get("evaluation_gates", [])
        if transfer_variant is not None
        else []
    )
    generation_gate = next(
        (
            gate
            for gate in generation_gate
            if gate.get("id") == "generation_stability"
        ),
        {},
    )
    patterns = {
        str(pattern) for pattern in generation_gate.get("answer_type_patterns", [])
    }
    _check(
        checks,
        "long_output_release_gate",
        _LONG_OUTPUT_PATTERNS.issubset(patterns)
        and generation_gate.get("max_degenerate_repetition_rate") == 0.0
        and float(generation_gate.get("max_token_rate", 1.0)) <= 0.05
        and generation_gate.get("max_structure_validity_drop") == 0.0,
        {
            "answer_type_patterns": sorted(patterns),
            "max_degenerate_repetition_rate": generation_gate.get(
                "max_degenerate_repetition_rate"
            ),
            "max_token_rate": generation_gate.get("max_token_rate"),
            "max_structure_validity_drop": generation_gate.get(
                "max_structure_validity_drop"
            ),
        },
    )

    pilot_spec = (
        transfer_variant.plan.raw_spec if transfer_variant is not None else {}
    )
    public_selection = _public_selection(pilot_spec)
    public_row_cap = public_selection.get("max_rows")
    synthetic_count = int(
        pilot_spec.get("synthetic", {}).get("count") or 0
    )
    pretraining_steps = int(
        pilot_spec.get("pretraining", {}).get("max_steps") or 0
    )
    sft_steps = int(
        pilot_spec.get("posttraining", {})
        .get("sft", {})
        .get("max_steps")
        or 0
    )
    rlvr_steps = int(
        pilot_spec.get("posttraining", {})
        .get("rlvr", {})
        .get("max_steps")
        or 0
    )
    _check(
        checks,
        "bounded_screening_budget",
        0 < synthetic_count <= 32
        and public_row_cap is not None
        and 0 < public_row_cap <= 256
        and public_selection.get("sampling_strategy") == "task_stratified"
        and public_selection.get("min_rows_per_task") == 16
        and 0 < pretraining_steps <= 25
        and 0 < sft_steps <= 10
        and 0 < rlvr_steps <= 5,
        {
            "synthetic_train_documents": synthetic_count,
            "public_row_cap": public_row_cap,
            "public_sampling_strategy": public_selection.get(
                "sampling_strategy"
            ),
            "public_min_rows_per_task": public_selection.get(
                "min_rows_per_task"
            ),
            "pretraining_steps": pretraining_steps,
            "sft_steps": sft_steps,
            "rlvr_steps": rlvr_steps,
        },
    )

    statuses = [check["status"] for check in checks]
    result = {
        "schema_version": 1,
        "claim_scope": "lfm_selective_transfer_pilot_submission_only",
        "overall_status": "pass" if all(
            status == "pass" for status in statuses
        ) else "fail",
        "pilot_submission_authorized": all(
            status == "pass" for status in statuses
        ),
        "quality_claim_authorized": False,
        "target_cuda_feasibility_claim_authorized": False,
        "sweep": {
            "name": plan.name,
            "source": sweep_label,
            "source_fingerprint": _file_fingerprint(sweep_source),
            "plan_fingerprint": plan.fingerprint,
            "baseline": plan.baseline,
            "replicates": list(plan.replicates),
            "arms": sorted(variants),
        },
        "real_source_preflight": {
            "source": preflight_label,
            "source_fingerprint": _file_fingerprint(preflight_source),
            "claim_scope": preflight.get("claim_scope"),
        },
        "source_selection": {
            "source": source_selection_label,
            "source_fingerprint": _file_fingerprint(
                source_selection_source
            ),
            "report_fingerprint": source_selection.get(
                "report_fingerprint"
            ),
            "claim_scope": source_selection.get("claim_scope"),
            "architecture_report_fingerprint": _file_fingerprint(
                architecture_report_source
            ),
            "weight_report_fingerprint": _file_fingerprint(
                weight_report_source
            ),
            "catalog_fingerprint": _file_fingerprint(
                architecture_catalog_source
            ),
            "smol_vision_preflight_fingerprint": _file_fingerprint(
                smol_preflight_source
            ),
        },
        "checks": checks,
        "counts": {
            "pass": statuses.count("pass"),
            "fail": statuses.count("fail"),
        },
        "limitations": [
            "This audit authorizes only submission of the one-seed screening pilot.",
            "The strict-transfer CUDA feasibility stage must execute and pass on the target GPU.",
            "Quality and promotion require the sealed paired three-seed confirmatory sweep.",
        ],
    }
    result["fingerprint"] = _fingerprint(result)
    return result


def audit_smol_vision_transfer_pilot(
    plan: SweepPlan,
    *,
    repo_root: str | Path,
    sweep_path: str | Path,
    vision_preflight_path: str | Path,
    language_preflight_path: str | Path,
    source_selection_path: str | Path,
) -> dict[str, Any]:
    """Audit the bounded Smol vision-block screening pilot."""

    repo = Path(repo_root).resolve()
    sweep_source = Path(sweep_path).resolve()
    vision_source = Path(vision_preflight_path).resolve()
    language_source = Path(language_preflight_path).resolve()
    selection_source = Path(source_selection_path).resolve()
    vision_preflight = json.loads(vision_source.read_text(encoding="utf-8"))
    language_preflight = json.loads(
        language_source.read_text(encoding="utf-8")
    )
    source_selection = json.loads(
        selection_source.read_text(encoding="utf-8")
    )
    architecture_report_source = (
        repo / "docs" / "results" / "small_vlm_architecture_commonality.json"
    )
    weight_report_source = (
        repo / "docs" / "results" / "small_vlm_weight_commonality.json"
    )
    architecture_catalog_source = (
        repo / "configs" / "small_vlm_architectures.yaml"
    )
    architecture_report = json.loads(
        architecture_report_source.read_text(encoding="utf-8")
    )
    weight_report = json.loads(
        weight_report_source.read_text(encoding="utf-8")
    )
    profiles = load_architecture_catalog(architecture_catalog_source)
    variants = {variant.arm_id: variant for variant in plan.variants}
    required_arms = {"lfm_language_only", "lfm_smol_dual"}
    language_only = variants.get("lfm_language_only")
    dual = variants.get("lfm_smol_dual")
    checks: list[dict[str, Any]] = []

    _check(
        checks,
        "screening_design",
        set(variants) == required_arms
        and plan.baseline == "lfm_language_only"
        and len(plan.replicates) == 1
        and plan.promotion is None,
        {
            "arms": sorted(variants),
            "baseline": plan.baseline,
            "replicates": list(plan.replicates),
            "promotion_enabled": plan.promotion is not None,
        },
    )

    pair_present = language_only is not None and dual is not None
    geometry_equal = bool(
        pair_present
        and language_only.plan.resolved_blueprint
        == dual.plan.resolved_blueprint
        and language_only.parameters == dual.parameters
    )
    target_parameters = (
        dual.parameters.get("total") if dual is not None else None
    )
    _check(
        checks,
        "sub1b_geometry_matched_control",
        geometry_equal
        and isinstance(target_parameters, int)
        and target_parameters < 1_000_000_000,
        {
            "geometry_equal": geometry_equal,
            "language_only_parameters": (
                None if language_only is None else language_only.parameters
            ),
            "dual_parameters": None if dual is None else dual.parameters,
        },
    )

    language_initialization = (
        {}
        if language_only is None
        else language_only.plan.raw_spec.get("initialization", {})
    )
    dual_initialization = (
        {}
        if dual is None
        else dual.plan.raw_spec.get("initialization", {})
    )
    language_hub = (
        dual_initialization.get("language_source", {}).get("hub", {})
        if isinstance(dual_initialization.get("language_source"), dict)
        else {}
    )
    vision_hub = (
        dual_initialization.get("vision_source", {}).get("hub", {})
        if isinstance(dual_initialization.get("vision_source"), dict)
        else {}
    )
    language_revision = str(language_hub.get("revision") or "")
    vision_revision = str(vision_hub.get("revision") or "")
    _check(
        checks,
        "initialization_contrast",
        language_initialization.get("arm") == "I8_lfm_aligned_language"
        and dual_initialization.get("arm") == "I9_lfm_smol_dual"
        and language_initialization.get("vision_source") is None
        and language_initialization.get("language_source")
        == dual_initialization.get("language_source"),
        {
            "language_only_arm": language_initialization.get("arm"),
            "dual_arm": dual_initialization.get("arm"),
            "unused_vision_source_acquired_by_control": (
                language_initialization.get("vision_source") is not None
            ),
            "language_source_matched": (
                language_initialization.get("language_source")
                == dual_initialization.get("language_source")
            ),
        },
    )
    _check(
        checks,
        "immutable_sources",
        language_hub.get("repo_id") == "LiquidAI/LFM2.5-VL-1.6B"
        and bool(_IMMUTABLE_REVISION.fullmatch(language_revision))
        and vision_hub.get("repo_id")
        == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        and bool(_IMMUTABLE_REVISION.fullmatch(vision_revision))
        and vision_hub.get("tensor_prefixes")
        == ["model.vision_model.encoder.layers."],
        {
            "language_repo_id": language_hub.get("repo_id"),
            "language_revision": language_revision,
            "vision_repo_id": vision_hub.get("repo_id"),
            "vision_revision": vision_revision,
            "vision_tensor_prefixes": vision_hub.get("tensor_prefixes"),
        },
    )

    preflight_source = vision_preflight.get("source") or {}
    preflight_target = vision_preflight.get("target") or {}
    transfer = vision_preflight.get("transfer") or {}
    _check(
        checks,
        "executed_vision_payload_identity",
        vision_preflight.get("claim_scope")
        == "real_source_vision_initialization_contract_only"
        and vision_preflight.get("quality_claim_authorized") is False
        and vision_preflight.get("promotion_claim_authorized") is False
        and preflight_source.get("model") == vision_hub.get("repo_id")
        and preflight_source.get("revision") == vision_revision
        and int(preflight_source.get("source_payload_bytes") or 0) > 0
        and str(
            preflight_source.get("selected_content_fingerprint") or ""
        ).startswith("sha256:"),
        {
            "claim_scope": vision_preflight.get("claim_scope"),
            "model": preflight_source.get("model"),
            "revision": preflight_source.get("revision"),
            "payload_bytes": preflight_source.get("source_payload_bytes"),
            "content_fingerprint": preflight_source.get(
                "selected_content_fingerprint"
            ),
            "quality_claim_authorized": vision_preflight.get(
                "quality_claim_authorized"
            ),
        },
    )
    expected_vision_parameters = (
        None if dual is None else dual.parameters.get("vision")
    )
    _check(
        checks,
        "executed_vision_transfer_integrity",
        preflight_target.get("parameters") == expected_vision_parameters
        and transfer.get("family") == "siglip"
        and transfer.get("vision_scope") == "transformer_blocks"
        and transfer.get("shape_policy") == "exact"
        and transfer.get("all_copied_keys_within_scope") is True
        and int(transfer.get("shape_skips") or 0) == 0
        and int(transfer.get("semantic_skips") or 0) == 0
        and int(transfer.get("missing_source") or 0) == 0
        and not transfer.get("unhealthy_source_weight_roles")
        and transfer.get("value_verified") is True
        and float(
            transfer.get("realized_component_parameter_fraction") or 0.0
        )
        >= 0.95,
        {
            "target_vision_parameters": preflight_target.get("parameters"),
            "compiled_vision_parameters": expected_vision_parameters,
            "copied_tensors": transfer.get("copied_tensors"),
            "copied_parameters": transfer.get("copied_parameters"),
            "realized_component_parameter_fraction": transfer.get(
                "realized_component_parameter_fraction"
            ),
            "shape_skips": transfer.get("shape_skips"),
            "semantic_skips": transfer.get("semantic_skips"),
            "missing_source": transfer.get("missing_source"),
            "scope_valid": transfer.get("all_copied_keys_within_scope"),
            "value_verified": transfer.get("value_verified"),
        },
    )

    selection_audit = validate_source_selection_matrix(
        source_selection,
        architecture_report=architecture_report,
        weight_report=weight_report,
        profiles=profiles,
        real_payload_preflights=[
            language_preflight,
            vision_preflight,
        ],
    )
    aligned = next(
        (
            target
            for target in source_selection.get("targets", [])
            if isinstance(target, dict)
            and target.get("target") == "docvlm-lfm-aligned-814m"
        ),
        {},
    )
    smol = next(
        (
            source
            for source in aligned.get("sources", [])
            if isinstance(source, dict)
            and source.get("model_id") == vision_hub.get("repo_id")
            and source.get("revision") == vision_revision
        ),
        {},
    )
    payload = smol.get("real_payload_evidence") or {}
    actions = smol.get("action_counts") or {}
    _check(
        checks,
        "cross_model_source_selection",
        selection_audit["status"] == "pass"
        and source_selection.get("quality_claim_authorized") is False
        and source_selection.get("promotion_claim_authorized") is False
        and int(actions.get("direct_copy_candidate") or 0) >= 1
        and payload.get("status") == "verified"
        and payload.get("source_content_fingerprint")
        == preflight_source.get("selected_content_fingerprint"),
        {
            "matrix_validation_status": selection_audit["status"],
            "target": aligned.get("target"),
            "source": smol.get("model_id"),
            "direct_copy_candidates": actions.get(
                "direct_copy_candidate"
            ),
            "real_payload_status": payload.get("status"),
            "quality_claim_authorized": source_selection.get(
                "quality_claim_authorized"
            ),
            "promotion_claim_authorized": source_selection.get(
                "promotion_claim_authorized"
            ),
        },
    )

    stage_names = {
        arm: _stage_names(variant) for arm, variant in variants.items()
    }
    common_stages = {
        "acquire_language_checkpoint",
        "initialize_student",
        "pretrain",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
    }
    topology_valid = bool(
        pair_present
        and common_stages.issubset(stage_names["lfm_language_only"])
        and common_stages.issubset(stage_names["lfm_smol_dual"])
        and "acquire_vision_checkpoint"
        not in stage_names["lfm_language_only"]
        and "acquire_vision_checkpoint" in stage_names["lfm_smol_dual"]
    )
    _check(
        checks,
        "single_selective_acquisition",
        topology_valid,
        {
            "language_only_acquires_vision": (
                "acquire_vision_checkpoint"
                in stage_names.get("lfm_language_only", [])
            ),
            "dual_acquires_vision": (
                "acquire_vision_checkpoint"
                in stage_names.get("lfm_smol_dual", [])
            ),
            "stage_counts": {
                arm: len(names) for arm, names in stage_names.items()
            },
        },
    )
    feasibility_arms = sorted(
        arm
        for arm, names in stage_names.items()
        if "training_feasibility_benchmark" in names
    )
    _check(
        checks,
        "dual_only_cuda_preflight",
        feasibility_arms == ["lfm_smol_dual"],
        {"training_feasibility_arms": feasibility_arms},
    )

    training_gate = (
        dual.plan.resolved_blueprint.get("evaluation_gates", [])
        if dual is not None
        else []
    )
    training_gate = next(
        (
            gate
            for gate in training_gate
            if gate.get("id") == "training_feasibility"
        ),
        {},
    )
    _check(
        checks,
        "target_gpu_training_contract",
        training_gate.get("required_device_type") == "cuda"
        and training_gate.get("required_precision") == "bfloat16"
        and training_gate.get(
            "required_resolved_visual_attention_backend"
        )
        == "flex"
        and training_gate.get("require_gradient_checkpointing") is True
        and training_gate.get(
            "required_gradient_checkpointing_components"
        )
        == ["vision", "connector", "language"]
        and training_gate.get(
            "required_gradient_checkpointing_use_reentrant"
        )
        is False,
        {
            "required_device_type": training_gate.get(
                "required_device_type"
            ),
            "required_precision": training_gate.get(
                "required_precision"
            ),
            "required_visual_backend": training_gate.get(
                "required_resolved_visual_attention_backend"
            ),
            "gradient_checkpointing": training_gate.get(
                "require_gradient_checkpointing"
            ),
            "gradient_checkpointing_components": training_gate.get(
                "required_gradient_checkpointing_components"
            ),
            "gradient_checkpointing_use_reentrant": training_gate.get(
                "required_gradient_checkpointing_use_reentrant"
            ),
        },
    )

    tracked = {
        "pretrain",
        "sft",
        "rlvr",
        "evaluate_baseline",
        "evaluate",
    }
    tracking_valid = True
    tracked_by_arm: dict[str, list[str]] = {}
    for arm, variant in variants.items():
        observed = []
        for stage in variant.plan.stages:
            if stage.name not in tracked and stage.name != (
                "training_feasibility_benchmark"
            ):
                continue
            observed.append(stage.name)
            tracking_valid &= (
                _command_option(stage.command, "--wandb-project")
                == "docvlm-ablation"
                and _command_option(stage.command, "--wandb-entity")
                == "sbdc"
                and bool(_command_option(stage.command, "--wandb-run"))
                and bool(_command_option(stage.command, "--wandb-group"))
            )
        expected = set(tracked)
        if arm == "lfm_smol_dual":
            expected.add("training_feasibility_benchmark")
        tracking_valid &= set(observed) == expected
        tracked_by_arm[arm] = sorted(observed)
    _check(
        checks,
        "observable_wandb_runs",
        tracking_valid,
        {
            "entity": "sbdc",
            "project": "docvlm-ablation",
            "stages_by_arm": tracked_by_arm,
        },
    )

    policies = {
        arm: _evaluation_policy(variant)
        for arm, variant in variants.items()
    }
    unique_policies = {
        json.dumps(policy, sort_keys=True)
        for policy in policies.values()
    }
    policy = next(iter(policies.values()), {})
    budgets = policy.get("max_new_tokens_by_answer_type") or {}
    _check(
        checks,
        "matched_long_output_policy",
        len(unique_policies) == 1
        and policy.get("max_new_tokens") == 128
        and policy.get("max_new_tokens_hard_cap") == 512
        and policy.get("repetition_guard_min_tokens") == 24
        and policy.get("repetition_guard_max_period") == 16
        and policy.get("repetition_guard_repetitions") == 3
        and policy.get("sample_selection") == "answer_type_round_robin"
        and budgets.get("table*") == 512
        and budgets.get("recognition_fullpage") == 512,
        {
            "policy": policy,
            "matched_across_arms": len(unique_policies) == 1,
        },
    )

    generation_gate = (
        dual.plan.resolved_blueprint.get("evaluation_gates", [])
        if dual is not None
        else []
    )
    generation_gate = next(
        (
            gate
            for gate in generation_gate
            if gate.get("id") == "generation_stability"
        ),
        {},
    )
    patterns = {
        str(pattern)
        for pattern in generation_gate.get("answer_type_patterns", [])
    }
    _check(
        checks,
        "long_output_release_gate",
        _LONG_OUTPUT_PATTERNS.issubset(patterns)
        and generation_gate.get("max_degenerate_repetition_rate") == 0.0
        and float(generation_gate.get("max_token_rate", 1.0)) <= 0.05
        and generation_gate.get("max_structure_validity_drop") == 0.0,
        {
            "answer_type_patterns": sorted(patterns),
            "max_degenerate_repetition_rate": generation_gate.get(
                "max_degenerate_repetition_rate"
            ),
            "max_token_rate": generation_gate.get("max_token_rate"),
            "max_structure_validity_drop": generation_gate.get(
                "max_structure_validity_drop"
            ),
        },
    )

    pilot_spec = dual.plan.raw_spec if dual is not None else {}
    public_selection = _public_selection(pilot_spec)
    public_row_cap = public_selection.get("max_rows")
    synthetic_count = int(
        pilot_spec.get("synthetic", {}).get("count") or 0
    )
    pretraining_steps = int(
        pilot_spec.get("pretraining", {}).get("max_steps") or 0
    )
    sft_steps = int(
        pilot_spec.get("posttraining", {})
        .get("sft", {})
        .get("max_steps")
        or 0
    )
    rlvr_steps = int(
        pilot_spec.get("posttraining", {})
        .get("rlvr", {})
        .get("max_steps")
        or 0
    )
    _check(
        checks,
        "bounded_screening_budget",
        0 < synthetic_count <= 32
        and public_row_cap is not None
        and 0 < public_row_cap <= 256
        and public_selection.get("sampling_strategy") == "task_stratified"
        and public_selection.get("min_rows_per_task") == 16
        and 0 < pretraining_steps <= 25
        and 0 < sft_steps <= 10
        and 0 < rlvr_steps <= 5,
        {
            "synthetic_train_documents": synthetic_count,
            "public_row_cap": public_row_cap,
            "public_sampling_strategy": public_selection.get(
                "sampling_strategy"
            ),
            "public_min_rows_per_task": public_selection.get(
                "min_rows_per_task"
            ),
            "pretraining_steps": pretraining_steps,
            "sft_steps": sft_steps,
            "rlvr_steps": rlvr_steps,
        },
    )

    statuses = [check["status"] for check in checks]
    passed = all(status == "pass" for status in statuses)
    result = {
        "schema_version": 1,
        "claim_scope": "smol_vision_transfer_pilot_submission_only",
        "overall_status": "pass" if passed else "fail",
        "pilot_submission_authorized": passed,
        "quality_claim_authorized": False,
        "target_cuda_feasibility_claim_authorized": False,
        "sweep": {
            "name": plan.name,
            "source": _source_label(sweep_source, repo),
            "source_fingerprint": _file_fingerprint(sweep_source),
            "plan_fingerprint": plan.fingerprint,
            "baseline": plan.baseline,
            "replicates": list(plan.replicates),
            "arms": sorted(variants),
        },
        "real_source_preflights": {
            "vision": {
                "source": _source_label(vision_source, repo),
                "source_fingerprint": _file_fingerprint(vision_source),
                "claim_scope": vision_preflight.get("claim_scope"),
            },
            "language": {
                "source": _source_label(language_source, repo),
                "source_fingerprint": _file_fingerprint(language_source),
                "claim_scope": language_preflight.get("claim_scope"),
            },
        },
        "source_selection": {
            "source": _source_label(selection_source, repo),
            "source_fingerprint": _file_fingerprint(selection_source),
            "report_fingerprint": source_selection.get(
                "report_fingerprint"
            ),
            "architecture_report_fingerprint": _file_fingerprint(
                architecture_report_source
            ),
            "weight_report_fingerprint": _file_fingerprint(
                weight_report_source
            ),
            "catalog_fingerprint": _file_fingerprint(
                architecture_catalog_source
            ),
        },
        "checks": checks,
        "counts": {
            "pass": statuses.count("pass"),
            "fail": statuses.count("fail"),
        },
        "limitations": [
            "This audit authorizes only submission of the one-seed screening pilot.",
            "The dual-source CUDA feasibility stage must execute and pass on the target GPU.",
            "Quality and promotion require a sealed paired multi-seed experiment.",
        ],
    }
    result["fingerprint"] = _fingerprint(result)
    return result
