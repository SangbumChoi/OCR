"""Evidence-composed source selection for selective weight transfer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_COMPONENT_ROLES = {
    "connector": ("connector.projection",),
    "language.attention": (
        "language.attention.q",
        "language.attention.k",
        "language.attention.v",
        "language.attention.o",
        "language.norm",
    ),
    "language.mlp": (
        "language.mlp.gate",
        "language.mlp.up",
        "language.mlp.down",
        "language.norm",
    ),
    "language.short_convolution": (
        "language.short_convolution",
        "language.norm",
    ),
    "language.token_embeddings": ("language.token_embedding",),
    "vision.patch_embedding": ("vision.patch_embedding",),
    "vision.position": (),
    "vision.transformer_blocks": (
        "vision.attention.q",
        "vision.attention.k",
        "vision.attention.v",
        "vision.attention.o",
        "vision.mlp.in",
        "vision.mlp.out",
        "vision.norm",
    ),
}


def _stable_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _weight_models(
    report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    models = report.get("models")
    if not isinstance(models, list):
        raise ValueError("weight report models must be a list")
    result = {}
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("weight report model entries must be objects")
        model_id = str(model.get("model_id") or "")
        if not model_id or model_id in result:
            raise ValueError("weight report model identities must be unique")
        result[model_id] = model
    return result


def _population_roles(
    report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    commonality = report.get("commonality")
    if not isinstance(commonality, Mapping):
        raise ValueError("weight report commonality must be an object")
    roles = commonality.get("common_roles")
    if not isinstance(roles, list):
        raise ValueError("weight report common roles must be a list")
    return {
        str(item["role"]): item
        for item in roles
        if isinstance(item, Mapping) and item.get("role")
    }


def _component_decision(
    component: str,
    architecture: Mapping[str, Any],
    weight_model: Mapping[str, Any],
    population: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mode = str(architecture.get("mode") or "")
    compatible = architecture.get("compatible") is True
    required = _COMPONENT_ROLES.get(component)
    if required is None:
        raise ValueError(f"unknown transfer component {component!r}")
    roles = weight_model.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("weight model roles must be an object")
    missing = [role for role in required if role not in roles]
    unhealthy = [
        role
        for role in required
        if role in roles
        and (
            not isinstance(roles[role], Mapping)
            or roles[role].get("sample_healthy") is not True
        )
    ]
    unstable_population = [
        role
        for role in required
        if role in population
        and population[role].get("stable_across_models") is not True
    ]

    if not compatible or mode == "distill_only":
        action = "feature_or_relation_distillation"
    elif unhealthy:
        action = "reject_unhealthy_source"
    elif mode == "token_rows":
        action = "token_identity_map_required"
    elif not required or missing:
        action = "pairwise_payload_preflight_required"
    elif mode == "structured_mlp":
        action = "structured_transfer_candidate"
    elif mode == "exact":
        action = "direct_copy_candidate"
    else:
        raise ValueError(
            f"unsupported compatible transfer mode {mode!r}"
        )
    result = {
        "component": component,
        "architecture_mode": mode,
        "architecture_compatible": compatible,
        "action": action,
        "population_prior_authorizes_copy": False,
    }
    if action != "feature_or_relation_distillation":
        result.update(
            {
                "required_weight_roles": list(required),
                "missing_weight_roles": missing,
                "unhealthy_weight_roles": unhealthy,
                "population_unstable_roles": unstable_population,
            }
        )
    return result


def _real_payload_evidence(
    *,
    target: str,
    profile: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if preflight is None:
        return {"status": "not_supplied"}
    source = preflight.get("source")
    transfer = preflight.get("transfer")
    target_record = preflight.get("target")
    if not all(
        isinstance(item, Mapping)
        for item in (source, transfer, target_record)
    ):
        return {"status": "invalid"}
    relevant = (
        target == "docvlm-lfm-aligned-814m"
        and source.get("model") == profile.get("model_id")
        and source.get("revision") == profile.get("revision")
    )
    if not relevant:
        return {"status": "not_applicable"}
    verified = (
        preflight.get("claim_scope")
        == "real_source_initialization_contract_only"
        and preflight.get("quality_claim_authorized") is False
        and transfer.get("component") == "language"
        and transfer.get("attention_geometry_compatible") is True
        and transfer.get("short_convolution_compatible") is True
        and transfer.get("mlp_operator_compatible") is True
        and transfer.get("value_verified") is True
        and not transfer.get("unhealthy_source_weight_roles")
        and int(transfer.get("shape_skips") or 0) == 0
        and int(transfer.get("semantic_skips") or 0) == 0
        and int(transfer.get("missing_source") or 0) == 0
        and float(
            transfer.get("realized_component_parameter_fraction") or 0
        )
        >= float(
            transfer.get("minimum_component_parameter_fraction") or 1
        )
    )
    return {
        "status": "verified" if verified else "failed",
        "component": transfer.get("component"),
        "copied_parameters": transfer.get("copied_parameters"),
        "realized_component_parameter_fraction": transfer.get(
            "realized_component_parameter_fraction"
        ),
        "value_verified": transfer.get("value_verified"),
        "shape_skips": transfer.get("shape_skips"),
        "semantic_skips": transfer.get("semantic_skips"),
        "missing_source": transfer.get("missing_source"),
        "source_content_fingerprint": source.get("content_fingerprint"),
    }


def build_source_selection_matrix(
    architecture_report: Mapping[str, Any],
    weight_report: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    *,
    real_payload_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose topology, sampled-weight, and real-payload transfer evidence."""

    if architecture_report.get("schema_version") != 2:
        raise ValueError("architecture report must use schema_version 2")
    if weight_report.get("schema_version") != 1:
        raise ValueError("weight report must use schema_version 1")
    profile_by_id = {}
    for profile in profiles:
        profile_id = str(profile.get("id") or "")
        if not profile_id or profile_id in profile_by_id:
            raise ValueError("architecture profile IDs must be unique")
        profile_by_id[profile_id] = profile
    weights = _weight_models(weight_report)
    population = _population_roles(weight_report)
    targets = []
    for key in ("default_target", "lfm_aligned_target"):
        target_report = architecture_report.get(key)
        if not isinstance(target_report, Mapping):
            raise ValueError(f"architecture report missing {key}")
        target_id = str(target_report.get("target") or "")
        compatibility = target_report.get("compatibility")
        if not isinstance(compatibility, list):
            raise ValueError(f"{key}.compatibility must be a list")
        sources = []
        for source in compatibility:
            if not isinstance(source, Mapping):
                raise ValueError("architecture compatibility must be objects")
            source_id = str(source.get("source") or "")
            profile = profile_by_id.get(source_id)
            if profile is None:
                raise ValueError(f"unknown architecture source {source_id!r}")
            model_id = str(profile.get("model_id") or "")
            weight_model = weights.get(model_id)
            if weight_model is None:
                raise ValueError(
                    f"missing sampled-weight evidence for {model_id}"
                )
            if weight_model.get("revision") != profile.get("revision"):
                raise ValueError(
                    f"revision mismatch for sampled source {model_id}"
                )
            decisions = source.get("decisions")
            if not isinstance(decisions, Mapping):
                raise ValueError("source transfer decisions must be an object")
            component_decisions = [
                _component_decision(
                    component,
                    decision,
                    weight_model,
                    population,
                )
                for component, decision in sorted(decisions.items())
            ]
            counts = Counter(
                item["action"] for item in component_decisions
            )
            payload = _real_payload_evidence(
                target=target_id,
                profile=profile,
                preflight=real_payload_preflight,
            )
            sources.append(
                {
                    "source": source_id,
                    "model_id": model_id,
                    "revision": profile.get("revision"),
                    "architecture_compatibility_fraction": source.get(
                        "compatibility_fraction"
                    ),
                    "sampled_weight_roles": weight_model.get(
                        "sampled_roles"
                    ),
                    "sampled_weight_unhealthy_roles": sorted(
                        role
                        for role, value in weight_model["roles"].items()
                        if not isinstance(value, Mapping)
                        or value.get("sample_healthy") is not True
                    ),
                    "decisions": component_decisions,
                    "action_counts": dict(sorted(counts.items())),
                    "real_payload_evidence": payload,
                }
            )
        sources.sort(
            key=lambda item: (
                item["real_payload_evidence"]["status"] != "verified",
                -item["action_counts"].get("direct_copy_candidate", 0),
                -item["action_counts"].get(
                    "structured_transfer_candidate", 0
                ),
                item["source"],
            )
        )
        targets.append({"target": target_id, "sources": sources})

    result = {
        "schema_version": 1,
        "claim_scope": "selective_transfer_source_selection_only",
        "inputs": {
            "architecture_report_fingerprint": _stable_fingerprint(
                architecture_report
            ),
            "weight_report_fingerprint": weight_report.get(
                "report_fingerprint"
            ),
            "catalog_fingerprint": _stable_fingerprint(list(profiles)),
            "real_payload_fingerprint": (
                None
                if real_payload_preflight is None
                else _stable_fingerprint(real_payload_preflight)
            ),
        },
        "decision_contract": {
            "population_statistics_establish_basis_alignment": False,
            "architecture_match_alone_authorizes_copy": False,
            "sampled_weight_health_alone_authorizes_copy": False,
            "token_rows_require_identity_map": True,
            "copy_requires_pairwise_payload_preflight": True,
            "empirical_quality_requires_matched_training": True,
        },
        "targets": targets,
        "quality_claim_authorized": False,
        "promotion_claim_authorized": False,
    }
    result["report_fingerprint"] = _stable_fingerprint(result)
    return result


def validate_source_selection_matrix(
    report: Mapping[str, Any],
    *,
    architecture_report: Mapping[str, Any] | None = None,
    weight_report: Mapping[str, Any] | None = None,
    profiles: Sequence[Mapping[str, Any]] | None = None,
    real_payload_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate internal integrity and the fail-closed claim boundary."""

    errors = []
    body = dict(report)
    fingerprint = body.pop("report_fingerprint", None)
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if fingerprint != _stable_fingerprint(body):
        errors.append("report fingerprint mismatch")
    contract = report.get("decision_contract")
    if not isinstance(contract, Mapping):
        errors.append("decision contract is missing")
    else:
        false_fields = (
            "population_statistics_establish_basis_alignment",
            "architecture_match_alone_authorizes_copy",
            "sampled_weight_health_alone_authorizes_copy",
        )
        if any(contract.get(field) is not False for field in false_fields):
            errors.append("basis-alignment claim boundary is weakened")
        true_fields = (
            "token_rows_require_identity_map",
            "copy_requires_pairwise_payload_preflight",
            "empirical_quality_requires_matched_training",
        )
        if any(contract.get(field) is not True for field in true_fields):
            errors.append("transfer preflight requirements are weakened")
    if report.get("quality_claim_authorized") is not False:
        errors.append("source selection cannot authorize quality")
    if report.get("promotion_claim_authorized") is not False:
        errors.append("source selection cannot authorize promotion")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("input fingerprints are missing")
    else:
        expected_inputs = {
            "architecture_report_fingerprint": (
                None
                if architecture_report is None
                else _stable_fingerprint(architecture_report)
            ),
            "weight_report_fingerprint": (
                None
                if weight_report is None
                else weight_report.get("report_fingerprint")
            ),
            "catalog_fingerprint": (
                None
                if profiles is None
                else _stable_fingerprint(list(profiles))
            ),
            "real_payload_fingerprint": (
                None
                if real_payload_preflight is None
                else _stable_fingerprint(real_payload_preflight)
            ),
        }
        for field, expected in expected_inputs.items():
            if expected is not None and inputs.get(field) != expected:
                errors.append(f"stale {field}")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "report_fingerprint": fingerprint,
    }
