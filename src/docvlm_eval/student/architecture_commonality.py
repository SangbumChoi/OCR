"""Cross-model architecture commonality for selective-transfer preflight."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


_COMPONENT_FIELDS = {
    "vision": (
        "family",
        "mixer",
        "width",
        "layers",
        "attention_heads",
        "patch_size",
        "norm",
        "activation",
        "position",
        "dynamic_resolution",
    ),
    "connector": (
        "family",
        "input_width",
        "output_width",
        "latent_tokens",
        "activation",
    ),
    "language": (
        "family",
        "mode",
        "mixer",
        "width",
        "layers",
        "attention_heads",
        "kv_heads",
        "head_dim",
        "mlp_width",
        "norm",
        "activation",
        "position",
        "rope_base",
        "vocab_size",
    ),
}


def load_architecture_catalog(path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("architecture catalog must use schema_version 1")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("architecture catalog profiles must be non-empty")
    normalized = [validate_architecture_profile(profile) for profile in profiles]
    ids = [profile["id"] for profile in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("architecture profile ids must be unique")
    return normalized


def validate_architecture_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise ValueError("architecture profile must be a mapping")
    for name in ("id", "model_id", "revision", "config_url"):
        if not isinstance(profile.get(name), str) or not profile[name]:
            raise ValueError(f"architecture profile {name} is required")
    normalized = dict(profile)
    for component, fields in _COMPONENT_FIELDS.items():
        value = profile.get(component)
        if not isinstance(value, Mapping):
            raise ValueError(f"architecture profile {component} is required")
        missing = set(fields) - set(value)
        if missing:
            raise ValueError(
                f"architecture profile {component} is missing {sorted(missing)}"
            )
        normalized[component] = {
            field: value[field] for field in fields
        }
    return normalized


def profile_from_blueprint(
    blueprint: Mapping[str, Any],
) -> dict[str, Any]:
    student = blueprint["student"]
    vision = student["vision"]
    connector = student["connector"]
    language = student["language"]
    full_attention_layers = language.get("full_attention_layers")
    language_mixer = (
        "global_attention"
        if full_attention_layers is None
        else "hybrid_short_convolution_attention"
    )
    return validate_architecture_profile(
        {
            "id": str(blueprint["name"]),
            "model_id": "local/sub1b-blueprint",
            "revision": "working-tree",
            "parameters_millions": (
                float(blueprint["budget"]["target_parameters"]) / 1e6
            ),
            "config_url": "configs/sub1b_architecture.yaml",
            "vision": {
                "family": str(vision["family"]),
                "mixer": "global_attention",
                "width": int(vision["width"]),
                "layers": int(vision["layers"]),
                "attention_heads": int(vision["attention_heads"]),
                "patch_size": int(vision["patch_size"]),
                "norm": "layer_norm",
                "activation": "gelu",
                "position": "learned_absolute_2d",
                "dynamic_resolution": True,
            },
            "connector": {
                "family": str(connector["family"]),
                "input_width": int(connector["input_width"]),
                "output_width": int(connector["output_width"]),
                "latent_tokens": int(connector["latent_tokens"]),
                "activation": "swiglu",
            },
            "language": {
                "family": "native_hybrid",
                "mode": str(language["family"]),
                "mixer": language_mixer,
                "width": int(language["width"]),
                "layers": int(language["layers"]),
                "attention_heads": int(language["attention_heads"]),
                "kv_heads": int(language["kv_heads"]),
                "head_dim": (
                    int(language["width"])
                    // int(language["attention_heads"])
                ),
                "mlp_width": int(language["mlp_width"]),
                "norm": "rms_norm",
                "activation": "swiglu",
                "position": "rope",
                "rope_base": float(language["rope_base"]),
                "vocab_size": int(language["vocab_size"]),
            },
        }
    )


def _decision(
    mode: str,
    reason: str,
    *,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "reason": reason,
        "checks": dict(checks),
        "compatible": mode in {"exact", "structured_mlp", "token_rows"},
    }


def transfer_compatibility(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_architecture_profile(source)
    target = validate_architecture_profile(target)
    sv, tv = source["vision"], target["vision"]
    sc, tc = source["connector"], target["connector"]
    sl, tl = source["language"], target["language"]

    vision_block_checks = {
        "attention_mixer": (
            sv["mixer"] == tv["mixer"] == "global_attention"
        ),
        "width": sv["width"] == tv["width"],
        "attention_heads": (
            sv["attention_heads"] == tv["attention_heads"]
        ),
        "norm": sv["norm"] == tv["norm"],
        "activation": sv["activation"] == tv["activation"],
    }
    vision_patch_checks = {
        "family": sv["family"] == tv["family"] == "vit",
        "width": sv["width"] == tv["width"],
        "patch_size": sv["patch_size"] == tv["patch_size"],
    }
    connector_checks = {
        field: sc[field] == tc[field]
        for field in (
            "family",
            "input_width",
            "output_width",
            "latent_tokens",
            "activation",
        )
    }
    language_geometry_checks = {
        "mode": sl["mode"] == tl["mode"] == "decoder_only",
        "width": sl["width"] == tl["width"],
        "attention_heads": (
            sl["attention_heads"] == tl["attention_heads"]
        ),
        "kv_heads": sl["kv_heads"] == tl["kv_heads"],
        "head_dim": sl["head_dim"] == tl["head_dim"],
        "norm": sl["norm"] == tl["norm"],
        "position": sl["position"] == tl["position"] == "rope",
        "rope_base": sl["rope_base"] == tl["rope_base"],
    }
    language_mlp_checks = {
        "mode": sl["mode"] == tl["mode"] == "decoder_only",
        "width": sl["width"] == tl["width"],
        "norm": sl["norm"] == tl["norm"],
        "activation": sl["activation"] == tl["activation"] == "swiglu",
    }
    exact_mlp = (
        all(language_mlp_checks.values())
        and sl["mlp_width"] == tl["mlp_width"]
    )
    structured_mlp = (
        all(language_mlp_checks.values())
        and int(sl["mlp_width"]) > int(tl["mlp_width"])
    )

    decisions = {
        "vision.patch_embedding": _decision(
            "exact" if all(vision_patch_checks.values()) else "distill_only",
            (
                "Patch kernels are directly aligned."
                if all(vision_patch_checks.values())
                else "Patch geometry or channel width differs."
            ),
            checks=vision_patch_checks,
        ),
        "vision.transformer_blocks": _decision(
            "exact" if all(vision_block_checks.values()) else "distill_only",
            (
                "Attention block semantics and geometry align."
                if all(vision_block_checks.values())
                else "Use feature or relation distillation across vision widths."
            ),
            checks=vision_block_checks,
        ),
        "vision.position": _decision(
            (
                "exact"
                if sv["position"] == tv["position"]
                and sv["patch_size"] == tv["patch_size"]
                else "distill_only"
            ),
            "Position weights require the same convention and patch grid.",
            checks={
                "position": sv["position"] == tv["position"],
                "patch_size": sv["patch_size"] == tv["patch_size"],
            },
        ),
        "connector": _decision(
            "exact" if all(connector_checks.values()) else "distill_only",
            (
                "Connector topology is identical."
                if all(connector_checks.values())
                else "Connector topology defines a different visual-token interface."
            ),
            checks=connector_checks,
        ),
        "language.attention": _decision(
            (
                "exact"
                if all(language_geometry_checks.values())
                else "distill_only"
            ),
            (
                "GQA and RoPE geometry align."
                if all(language_geometry_checks.values())
                else "Attention transfer is unsafe without identical GQA and RoPE geometry."
            ),
            checks=language_geometry_checks,
        ),
        "language.mlp": _decision(
            (
                "exact"
                if exact_mlp
                else "structured_mlp"
                if structured_mlp
                else "distill_only"
            ),
            (
                "SwiGLU projections align exactly."
                if exact_mlp
                else "Joint salience selection can reduce wider SwiGLU channels."
                if structured_mlp
                else "Hidden width, gating, or direction of width change is incompatible."
            ),
            checks={
                **language_mlp_checks,
                "source_mlp_not_narrower": (
                    int(sl["mlp_width"]) >= int(tl["mlp_width"])
                ),
            },
        ),
        "language.token_embeddings": _decision(
            "token_rows" if sl["width"] == tl["width"] else "distill_only",
            (
                "Only explicitly identity-mapped token rows may be copied."
                if sl["width"] == tl["width"]
                else "Embedding widths differ."
            ),
            checks={"width": sl["width"] == tl["width"]},
        ),
    }
    compatible = sum(item["compatible"] for item in decisions.values())
    return {
        "source": source["id"],
        "target": target["id"],
        "compatible_subcomponents": compatible,
        "total_subcomponents": len(decisions),
        "compatibility_fraction": compatible / len(decisions),
        "decisions": decisions,
    }


def architecture_commonality(
    profiles: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    prevalence_threshold: float = 0.6,
) -> dict[str, Any]:
    if not 0 < prevalence_threshold <= 1:
        raise ValueError("prevalence_threshold must be within (0, 1]")
    sources = [validate_architecture_profile(profile) for profile in profiles]
    target = validate_architecture_profile(target)
    minimum_count = math.ceil(prevalence_threshold * len(sources))
    common: list[dict[str, Any]] = []
    for component, fields in _COMPONENT_FIELDS.items():
        for field in fields:
            values = [
                json.dumps(
                    profile[component][field],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for profile in sources
            ]
            encoded, count = Counter(values).most_common(1)[0]
            if count >= minimum_count:
                common.append(
                    {
                        "feature": f"{component}.{field}",
                        "value": json.loads(encoded),
                        "count": count,
                        "fraction": count / len(sources),
                        "target_matches": (
                            target[component][field] == json.loads(encoded)
                        ),
                    }
                )
    compatibility = [
        transfer_compatibility(source, target) for source in sources
    ]
    compatibility.sort(
        key=lambda item: (
            -item["compatibility_fraction"],
            item["source"],
        )
    )
    return {
        "schema_version": 1,
        "source_count": len(sources),
        "target": target["id"],
        "prevalence_threshold": prevalence_threshold,
        "common_features": common,
        "compatibility": compatibility,
    }
