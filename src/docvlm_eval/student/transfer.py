"""Auditable selective transfer for compatible native and Hugging Face checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn


_BLOCK = re.compile(r"^(vision|language)\.blocks\.(\d+)\.(.+)$")
_MLP_WEIGHT = re.compile(
    r"^language\.blocks\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_LANGUAGE_ATTENTION = re.compile(
    r"^language\.blocks\.(\d+)\.attn\."
)
_SHAPE_POLICIES = {"exact", "structured_mlp"}


@dataclass
class TransferReport:
    family: str
    fractions: dict[str, float]
    shape_policy: str
    require_attention_geometry: bool
    require_healthy_source_weights: bool
    source_identity: dict[str, Any] | None = None
    source_attention_geometry: dict[str, Any] | None = None
    target_attention_geometry: dict[str, Any] | None = None
    attention_geometry_compatible: bool | None = None
    short_convolution_compatible: bool | None = None
    mlp_operator_compatible: bool | None = None
    source_weight_profile: dict[str, Any] | None = None
    source_weight_profile_fingerprint: str = ""
    unhealthy_source_weight_roles: list[str] = field(default_factory=list)
    source_topology_fingerprint: str = ""
    target_topology_fingerprint: str = ""
    copied_tensors: int = 0
    copied_parameters: int = 0
    structured_tensors: int = 0
    structured_parameters: int = 0
    structured_groups: list[dict[str, Any]] = field(default_factory=list)
    token_rows_copied: int = 0
    skipped_by_policy: int = 0
    skipped_shape: list[dict[str, Any]] = field(default_factory=list)
    skipped_semantic: list[dict[str, Any]] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)
    copied_keys: list[str] = field(default_factory=list)
    tensor_mappings: list[dict[str, Any]] = field(default_factory=list)
    mapping_fingerprint: str = ""
    copied_values_fingerprint: str = ""
    value_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _topology_fingerprint(
    state: Mapping[str, torch.Tensor],
) -> str:
    topology = [
        {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for key, tensor in sorted(state.items())
    ]
    return _json_fingerprint(topology)


def _tensor_value_fingerprint(
    tensor: torch.Tensor,
) -> str | None:
    if tensor.device.type == "meta":
        return None
    values = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    chunk_bytes = 4 * 1024 * 1024
    for start in range(0, values.numel(), chunk_bytes):
        chunk = values[start : start + chunk_bytes].cpu()
        digest.update(chunk.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _tensor_mapping(
    *,
    target_key: str,
    source_key: str,
    method: str,
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    copied_parameters: int,
    selection_fingerprint: str | None = None,
    copied_value_fingerprint: str | None = None,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "target": target_key,
        "source": source_key,
        "method": method,
        "target_shape": list(target_tensor.shape),
        "source_shape": list(source_tensor.shape),
        "target_dtype": str(target_tensor.dtype).removeprefix("torch."),
        "source_dtype": str(source_tensor.dtype).removeprefix("torch."),
        "copied_parameters": int(copied_parameters),
    }
    if selection_fingerprint is not None:
        mapping["selection_fingerprint"] = selection_fingerprint
    if copied_value_fingerprint is not None:
        mapping["copied_value_fingerprint"] = copied_value_fingerprint
    return mapping


def validate_transfer_report_attestation(
    report: Any,
    *,
    require_source_identity: bool = False,
    require_value_attestation: bool = False,
) -> dict[str, Any]:
    """Validate the topology and source-to-target manifest of a transfer report."""

    if not isinstance(report, dict):
        raise ValueError("transfer report must be a mapping")
    for field_name in (
        "source_topology_fingerprint",
        "target_topology_fingerprint",
        "mapping_fingerprint",
    ):
        value = report.get(field_name)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise ValueError(
                f"transfer report {field_name} is missing or invalid"
            )
    mappings = report.get("tensor_mappings")
    if not isinstance(mappings, list):
        raise ValueError("transfer report tensor_mappings must be a list")
    if report["mapping_fingerprint"] != _json_fingerprint(mappings):
        raise ValueError("transfer report mapping fingerprint mismatch")
    copied_value_fingerprints = [
        mapping.get("copied_value_fingerprint")
        for mapping in mappings
    ]
    if (
        require_value_attestation
        or "copied_values_fingerprint" in report
    ) and report.get("copied_values_fingerprint") != _json_fingerprint(
        copied_value_fingerprints
    ):
        raise ValueError("transfer report copied-values fingerprint mismatch")
    if int(report.get("copied_tensors", -1)) != len(mappings):
        raise ValueError(
            "transfer report copied_tensors does not match its mappings"
        )
    copied_keys = report.get("copied_keys")
    if copied_keys != [mapping.get("target") for mapping in mappings]:
        raise ValueError(
            "transfer report copied_keys does not match its mappings"
        )
    required_mapping_fields = {
        "target",
        "source",
        "method",
        "target_shape",
        "source_shape",
        "target_dtype",
        "source_dtype",
        "copied_parameters",
    }
    for mapping in mappings:
        if not isinstance(mapping, dict) or not required_mapping_fields <= set(
            mapping
        ):
            raise ValueError("transfer report tensor mapping is incomplete")
        if mapping["method"] not in {
            "exact",
            "structured_mlp",
            "token_rows",
        }:
            raise ValueError("transfer report tensor mapping method is invalid")
        if int(mapping["copied_parameters"]) <= 0:
            raise ValueError(
                "transfer report tensor mapping copied no parameters"
            )
        if mapping["method"] != "exact":
            selection = mapping.get("selection_fingerprint")
            if (
                not isinstance(selection, str)
                or not selection.startswith("sha256:")
            ):
                raise ValueError(
                    "non-exact transfer mapping selection fingerprint "
                    "is missing"
                )
        value_fingerprint = mapping.get("copied_value_fingerprint")
        if require_value_attestation and (
            not isinstance(value_fingerprint, str)
            or not value_fingerprint.startswith("sha256:")
        ):
            raise ValueError(
                "transfer mapping copied-value fingerprint is missing"
            )
    if require_value_attestation and report.get("value_verified") is not True:
        raise ValueError("transfer report values were not verified")
    weight_profile = report.get("source_weight_profile")
    if weight_profile is not None:
        if not isinstance(weight_profile, dict):
            raise ValueError("transfer report source weight profile is invalid")
        profile_body = dict(weight_profile)
        profile_fingerprint = profile_body.pop(
            "profile_fingerprint",
            None,
        )
        if profile_fingerprint != _json_fingerprint(profile_body):
            raise ValueError(
                "transfer report source weight profile fingerprint mismatch"
            )
        if (
            report.get("source_weight_profile_fingerprint")
            != profile_fingerprint
        ):
            raise ValueError(
                "transfer report source weight profile identity mismatch"
            )
    if report.get("require_healthy_source_weights"):
        if weight_profile is None and report.get("value_verified") is True:
            raise ValueError(
                "materialized strict transfer lacks a source weight profile"
            )
        unhealthy = set(report.get("unhealthy_source_weight_roles") or [])
        if unhealthy:
            from .weight_commonality import semantic_weight_role

            copied_unhealthy = sorted(
                {
                    role
                    for mapping in mappings
                    if (
                        role := semantic_weight_role(
                            str(mapping["target"]),
                            mapping["target_shape"],
                        )
                    )
                    in unhealthy
                }
            )
            if copied_unhealthy:
                raise ValueError(
                    "transfer report copied unhealthy source weight roles: "
                    f"{copied_unhealthy}"
                )
    source_identity = report.get("source_identity")
    if require_source_identity and not isinstance(source_identity, dict):
        raise ValueError("transfer report source identity is missing")
    if source_identity is not None:
        files = source_identity.get("files")
        if (
            source_identity.get("schema_version") != 1
            or source_identity.get("kind") != "checkpoint_content"
            or not isinstance(files, list)
            or not files
        ):
            raise ValueError("transfer report source identity is invalid")
        paths: list[str] = []
        for record in files:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not record["path"]
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] < 0
                or not isinstance(record.get("sha256"), str)
                or not record["sha256"].startswith("sha256:")
            ):
                raise ValueError(
                    "transfer report source file identity is invalid"
                )
            paths.append(record["path"])
        if len(paths) != len(set(paths)):
            raise ValueError(
                "transfer report source file identities are duplicated"
            )
        if source_identity.get("content_fingerprint") != _json_fingerprint(
            files
        ):
            raise ValueError(
                "transfer report source content fingerprint mismatch"
            )
        if int(source_identity.get("total_bytes", -1)) != sum(
            record["bytes"] for record in files
        ):
            raise ValueError("transfer report source byte count mismatch")
    return report


def canonicalize_source_state(
    source: Mapping[str, torch.Tensor],
    family: str = "student",
) -> dict[str, torch.Tensor]:
    """Map supported source state names into native student semantic names."""
    if family not in {"student", "siglip", "llama", "lfm2"}:
        raise ValueError(
            "source family must be student, siglip, llama, or lfm2"
        )
    out: dict[str, torch.Tensor] = {}
    for original, tensor in source.items():
        key = original.removeprefix("module.")
        if family == "siglip":
            key = re.sub(
                r"^(?:model\.)?vision_model\.embeddings\.patch_embedding\.",
                "vision.patch_embed.",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?vision_model\.embeddings\.position_embedding\.weight$",
                "vision.position_embedding",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?vision_model\.encoder\.layers\.(\d+)\.",
                r"vision.blocks.\1.",
                key,
            )
            key = key.replace(".self_attn.", ".attn.")
            key = key.replace(".attn.out_proj.", ".attn.o_proj.")
            key = key.replace(".attn.q_layernorm.", ".attn.q_norm.")
            key = key.replace(".attn.k_layernorm.", ".attn.k_norm.")
            key = key.replace(".layer_norm1.", ".norm1.")
            key = key.replace(".layer_norm2.", ".norm2.")
            key = key.replace(".mlp.fc1.", ".mlp.fc1.")
            key = key.replace(".mlp.fc2.", ".mlp.fc2.")
            key = re.sub(
                r"^(?:model\.)?vision_model\.post_layernorm\.",
                "vision.norm.",
                key,
            )
        elif family == "llama":
            key = re.sub(
                r"^(?:model\.)?(?:language_model\.)?model\.embed_tokens\.",
                "language.token_embedding.",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?(?:language_model\.)?model\.layers\.(\d+)\.",
                r"language.blocks.\1.",
                key,
            )
            key = key.replace(".self_attn.", ".attn.")
            key = key.replace(".input_layernorm.", ".norm1.")
            key = key.replace(".post_attention_layernorm.", ".norm2.")
            key = re.sub(
                r"^(?:model\.)?(?:language_model\.)?model\.norm\.",
                "language.norm.",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?(?:language_model\.)?lm_head\.",
                "lm_head.",
                key,
            )
        elif family == "lfm2":
            key = re.sub(
                r"^(?:model\.)?language_model\.",
                "",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?embed_tokens\.",
                "language.token_embedding.",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?layers\.(\d+)\.",
                r"language.blocks.\1.",
                key,
            )
            key = key.replace(".self_attn.", ".attn.")
            key = key.replace(".attn.out_proj.", ".attn.o_proj.")
            key = key.replace(".attn.q_layernorm.", ".attn.q_norm.")
            key = key.replace(".attn.k_layernorm.", ".attn.k_norm.")
            key = key.replace(".operator_norm.", ".norm1.")
            key = key.replace(".ffn_norm.", ".norm2.")
            key = key.replace(
                ".feed_forward.w1.",
                ".mlp.gate_proj.",
            )
            key = key.replace(
                ".feed_forward.w3.",
                ".mlp.up_proj.",
            )
            key = key.replace(
                ".feed_forward.w2.",
                ".mlp.down_proj.",
            )
            key = re.sub(
                r"^(?:model\.)?embedding_norm\.",
                "language.norm.",
                key,
            )
            key = re.sub(
                r"^(?:model\.)?lm_head\.",
                "lm_head.",
                key,
            )
        out[key] = tensor
    return out


def _block_count(keys: list[str], component: str) -> int:
    indices = []
    prefix = f"{component}.blocks."
    for key in keys:
        if key.startswith(prefix):
            try:
                indices.append(int(key.split(".")[2]))
            except (IndexError, ValueError):
                pass
    return max(indices, default=-1) + 1


def _language_layer_types(keys: list[str]) -> dict[int, str]:
    layer_types: dict[int, str] = {}
    for key in keys:
        match = _BLOCK.match(key)
        if match is None or match.group(1) != "language":
            continue
        index = int(match.group(2))
        suffix = match.group(3)
        observed = (
            "attention"
            if suffix.startswith("attn.")
            else "short_conv"
            if suffix.startswith("conv.")
            else None
        )
        if observed is None:
            continue
        previous = layer_types.get(index)
        if previous is not None and previous != observed:
            raise ValueError(
                f"language source layer {index} mixes attention and convolution"
            )
        layer_types[index] = observed
    return layer_types


def _typed_depth_map(
    target_index: int,
    target_types: Mapping[int, str],
    source_types: Mapping[int, str],
) -> int | None:
    layer_type = target_types.get(target_index)
    if layer_type is None:
        return None
    target_indices = sorted(
        index for index, value in target_types.items() if value == layer_type
    )
    source_indices = sorted(
        index for index, value in source_types.items() if value == layer_type
    )
    if not source_indices:
        return None
    target_rank = target_indices.index(target_index)
    source_rank = _depth_map(
        target_rank,
        len(target_indices),
        len(source_indices),
    )
    return source_indices[source_rank]


def _selected_blocks(count: int, fraction: float) -> set[int]:
    selected_count = min(count, max(0, round(count * fraction)))
    if selected_count == 0:
        return set()
    if selected_count == 1:
        return {count // 2}
    return {
        round(index * (count - 1) / (selected_count - 1))
        for index in range(selected_count)
    }


def _depth_map(target_index: int, target_count: int, source_count: int) -> int:
    if target_count <= 1 or source_count <= 1:
        return 0
    return round(target_index * (source_count - 1) / (target_count - 1))


def _target_attention_geometry(
    student: nn.Module,
) -> dict[str, Any] | None:
    config = getattr(student, "config", None)
    language = getattr(config, "language", None)
    if language is None:
        return None
    hidden = int(language.width)
    heads = int(language.attention_heads)
    return {
        "hidden_width": hidden,
        "attention_heads": heads,
        "kv_heads": int(language.kv_heads),
        "head_dim": hidden // heads,
        "rope_base": float(language.rope_base),
        "rope_layout": str(language.rope_layout),
        "norm_eps": float(language.norm_eps),
        "qk_norm": bool(language.qk_norm),
        "attention_bias": bool(language.attention_bias),
        "mlp_bias": bool(language.mlp_bias),
        "conv_kernel_size": int(language.conv_kernel_size),
        "conv_bias": bool(language.conv_bias),
    }


def _normalize_attention_geometry(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "hidden_width",
        "attention_heads",
        "kv_heads",
        "head_dim",
        "rope_base",
    }
    if not required <= set(value):
        raise ValueError(
            "attention geometry must contain hidden_width, attention_heads, "
            "kv_heads, head_dim, and rope_base"
        )
    normalized: dict[str, Any] = {
        "hidden_width": int(value["hidden_width"]),
        "attention_heads": int(value["attention_heads"]),
        "kv_heads": int(value["kv_heads"]),
        "head_dim": int(value["head_dim"]),
        "rope_base": float(value["rope_base"]),
        "rope_layout": str(value.get("rope_layout", "interleaved")),
        "norm_eps": float(value.get("norm_eps", 1e-6)),
        "qk_norm": bool(value.get("qk_norm", False)),
        "attention_bias": bool(value.get("attention_bias", True)),
        "mlp_bias": bool(value.get("mlp_bias", True)),
        "conv_kernel_size": int(value.get("conv_kernel_size", 3)),
        "conv_bias": bool(value.get("conv_bias", False)),
    }
    if (
        any(
            int(normalized[key]) <= 0
            for key in (
                "hidden_width",
                "attention_heads",
                "kv_heads",
                "head_dim",
            )
        )
        or float(normalized["rope_base"]) <= 0
        or float(normalized["norm_eps"]) <= 0
        or int(normalized["conv_kernel_size"]) < 2
    ):
        raise ValueError("attention geometry values must be positive")
    if normalized["rope_layout"] not in {"interleaved", "half_split"}:
        raise ValueError("attention geometry RoPE layout is invalid")
    return normalized


def _axis_squared_l2(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute channel salience in bounded-memory chunks."""
    channels = tensor.shape[axis]
    score = torch.empty(channels, dtype=torch.float32, device=tensor.device)
    for start in range(0, channels, 1024):
        length = min(1024, channels - start)
        chunk = tensor.narrow(axis, start, length).detach().float()
        reduce_axis = 1 if axis == 0 else 0
        score[start : start + length] = chunk.square().sum(dim=reduce_axis)
    return score


def _structured_mlp_group(
    target_prefix: str,
    source_prefix: str,
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    projections = ("gate_proj", "up_proj", "down_proj")
    target_keys = {
        projection: f"{target_prefix}.{projection}.weight"
        for projection in projections
    }
    source_keys = {
        projection: f"{source_prefix}.{projection}.weight"
        for projection in projections
    }
    if any(key not in target for key in target_keys.values()) or any(
        key not in source for key in source_keys.values()
    ):
        return None
    target_gate = target[target_keys["gate_proj"]]
    target_up = target[target_keys["up_proj"]]
    target_down = target[target_keys["down_proj"]]
    source_gate = source[source_keys["gate_proj"]]
    source_up = source[source_keys["up_proj"]]
    source_down = source[source_keys["down_proj"]]
    tensors = (
        target_gate,
        target_up,
        target_down,
        source_gate,
        source_up,
        source_down,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        return None
    target_channels, target_hidden = target_gate.shape
    source_channels, source_hidden = source_gate.shape
    if (
        target_up.shape != target_gate.shape
        or source_up.shape != source_gate.shape
        or target_down.shape != (target_hidden, target_channels)
        or source_down.shape != (source_hidden, source_channels)
        or target_hidden != source_hidden
        or source_channels <= target_channels
    ):
        return None

    is_shape_only = any(tensor.device.type == "meta" for tensor in tensors)
    if is_shape_only:
        reduced = {
            key: torch.empty_like(target[key])
            for key in target_keys.values()
        }
        selection = "shape_only_compatibility"
        fingerprint = None
        preview: list[int] = []
    else:
        score = _axis_squared_l2(source_gate, 0)
        score.add_(_axis_squared_l2(source_up, 0))
        score.add_(_axis_squared_l2(source_down, 1))
        indices = (
            torch.argsort(
                score.detach().cpu(),
                descending=True,
                stable=True,
            )[:target_channels]
            .sort()
            .values.to(source_gate.device)
        )
        reduced = {
            target_keys["gate_proj"]: source_gate.index_select(0, indices),
            target_keys["up_proj"]: source_up.index_select(0, indices),
            target_keys["down_proj"]: source_down.index_select(1, indices),
        }
        index_list = [int(value) for value in indices.detach().cpu().tolist()]
        fingerprint = (
            "sha256:"
            + hashlib.sha256(
                ",".join(str(value) for value in index_list).encode("ascii")
            ).hexdigest()
        )
        preview = (
            index_list
            if len(index_list) <= 16
            else index_list[:8] + index_list[-8:]
        )
        selection = "joint_l2_salience"

    return reduced, {
        "target_prefix": target_prefix,
        "source_prefix": source_prefix,
        "source_channels": source_channels,
        "target_channels": target_channels,
        "hidden_width": target_hidden,
        "selection": selection,
        "channel_index_fingerprint": fingerprint,
        "channel_index_preview": preview,
    }


@torch.no_grad()
def selective_transfer(
    student: nn.Module,
    source: Mapping[str, torch.Tensor],
    fractions: Mapping[str, float],
    *,
    family: str = "student",
    token_map: Mapping[int, int] | None = None,
    copy_token_embeddings: bool | None = None,
    shape_policy: str = "exact",
    source_identity: Mapping[str, Any] | None = None,
    source_attention_geometry: Mapping[str, Any] | None = None,
    require_attention_geometry: bool = False,
    require_healthy_source_weights: bool = False,
) -> TransferReport:
    """Copy policy-compatible tensors under component and depth controls.

    The structured MLP policy only reduces a complete SwiGLU group with a shared
    salience-selected intermediate-channel map. Other width mismatches remain random.
    """
    if shape_policy not in _SHAPE_POLICIES:
        raise ValueError(
            f"shape_policy must be one of {sorted(_SHAPE_POLICIES)}"
        )
    normalized_fractions = {
        component: float(fractions.get(component, 0.0))
        for component in ("vision", "language", "connector")
    }
    if any(not 0.0 <= value <= 1.0 for value in normalized_fractions.values()):
        raise ValueError("transfer fractions must be between zero and one")
    source_geometry = _normalize_attention_geometry(
        source_attention_geometry
    )
    target_geometry = _target_attention_geometry(student)
    attention_geometry_fields = (
        "hidden_width",
        "attention_heads",
        "kv_heads",
        "head_dim",
        "rope_base",
        "rope_layout",
        "norm_eps",
        "qk_norm",
        "attention_bias",
    )
    geometry_compatible = (
        None
        if source_geometry is None or target_geometry is None
        else all(
            source_geometry[field] == target_geometry[field]
            for field in attention_geometry_fields
        )
    )
    short_convolution_compatible = (
        None
        if source_geometry is None or target_geometry is None
        else all(
            source_geometry[field] == target_geometry[field]
            for field in (
                "hidden_width",
                "norm_eps",
                "conv_kernel_size",
                "conv_bias",
            )
        )
    )
    mlp_operator_compatible = (
        None
        if source_geometry is None or target_geometry is None
        else all(
            source_geometry[field] == target_geometry[field]
            for field in (
                "hidden_width",
                "norm_eps",
                "mlp_bias",
            )
        )
    )
    if (
        require_attention_geometry
        and normalized_fractions["language"] > 0
        and geometry_compatible is None
    ):
        raise ValueError(
            "strict language attention transfer requires source and target "
            "attention geometry"
        )
    source = canonicalize_source_state(source, family)
    target = student.state_dict()
    materialized_source = any(
        tensor.device.type != "meta" for tensor in source.values()
    )
    source_weight_profile = None
    unhealthy_source_weight_roles: set[str] = set()
    if materialized_source:
        from .weight_commonality import sketch_state_dict

        source_weight_profile = sketch_state_dict(
            source,
            model_id=(
                str(source_identity.get("content_fingerprint"))
                if isinstance(source_identity, Mapping)
                and source_identity.get("content_fingerprint")
                else f"in-memory/{family}"
            ),
        )
        unhealthy_source_weight_roles = {
            role
            for role, summary in source_weight_profile["roles"].items()
            if not summary["sample_healthy"]
        }
    report = TransferReport(
        family=family,
        fractions=normalized_fractions,
        shape_policy=shape_policy,
        require_attention_geometry=require_attention_geometry,
        require_healthy_source_weights=require_healthy_source_weights,
        source_identity=(
            json.loads(
                json.dumps(
                    source_identity,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            if source_identity is not None
            else None
        ),
        source_attention_geometry=source_geometry,
        target_attention_geometry=target_geometry,
        attention_geometry_compatible=geometry_compatible,
        short_convolution_compatible=short_convolution_compatible,
        mlp_operator_compatible=mlp_operator_compatible,
        source_weight_profile=source_weight_profile,
        source_weight_profile_fingerprint=(
            str(source_weight_profile["profile_fingerprint"])
            if source_weight_profile is not None
            else ""
        ),
        unhealthy_source_weight_roles=sorted(
            unhealthy_source_weight_roles
        ),
        source_topology_fingerprint=_topology_fingerprint(source),
        target_topology_fingerprint=_topology_fingerprint(target),
    )
    if copy_token_embeddings is None:
        copy_token_embeddings = family == "student"
    source_keys = list(source)
    target_keys = list(target)
    target_language_types = _language_layer_types(target_keys)
    source_language_types = _language_layer_types(source_keys)
    target_counts = {
        component: _block_count(target_keys, component)
        for component in ("vision", "language")
    }
    source_counts = {
        component: _block_count(source_keys, component)
        for component in ("vision", "language")
    }
    selected = {
        component: _selected_blocks(target_counts[component], normalized_fractions[component])
        for component in ("vision", "language")
    }
    copied_storages: set[Any] = set()
    structured_cache: dict[
        tuple[str, str],
        tuple[dict[str, torch.Tensor], dict[str, Any]] | None,
    ] = {}
    recorded_structured_groups: set[tuple[str, str]] = set()

    for target_key, target_tensor in target.items():
        component = target_key.split(".", 1)[0]
        if component == "lm_head":
            component = "language"
        if component not in normalized_fractions or normalized_fractions[component] <= 0:
            report.skipped_by_policy += 1
            continue
        if target_key == "lm_head.weight" and "language.token_embedding.weight" in target:
            report.skipped_by_policy += 1
            continue
        if require_healthy_source_weights:
            from .weight_commonality import semantic_weight_role

            source_role = semantic_weight_role(
                target_key,
                target_tensor.shape,
            )
            if source_role in unhealthy_source_weight_roles:
                report.skipped_by_policy += 1
                report.skipped_semantic.append(
                    {
                        "target": target_key,
                        "source": None,
                        "reason": "unhealthy_source_weight_role",
                        "role": source_role,
                    }
                )
                continue
        if target_key == "language.token_embedding.weight" and token_map is not None:
            source_tensor = source.get(target_key)
            if source_tensor is None:
                report.missing_source.append(target_key)
                continue
            if source_tensor.ndim != 2 or source_tensor.shape[1] != target_tensor.shape[1]:
                report.skipped_shape.append(
                    {
                        "target": target_key,
                        "source": target_key,
                        "target_shape": list(target_tensor.shape),
                        "source_shape": list(source_tensor.shape),
                    }
                )
                continue
            valid_pairs = sorted(
                (int(target_id), int(source_id))
                for target_id, source_id in token_map.items()
                if 0 <= int(target_id) < target_tensor.shape[0]
                and 0 <= int(source_id) < source_tensor.shape[0]
            )
            if valid_pairs:
                target_ids = torch.tensor(
                    [pair[0] for pair in valid_pairs],
                    dtype=torch.long,
                    device=target_tensor.device,
                )
                source_ids = torch.tensor(
                    [pair[1] for pair in valid_pairs],
                    dtype=torch.long,
                    device=source_tensor.device,
                )
                rows = source_tensor.index_select(0, source_ids).to(
                    device=target_tensor.device,
                    dtype=target_tensor.dtype,
                )
                target_tensor.index_copy_(0, target_ids, rows)
                copied_value_fingerprint = _tensor_value_fingerprint(rows)
                target_rows = target_tensor.index_select(0, target_ids)
                if (
                    copied_value_fingerprint
                    != _tensor_value_fingerprint(target_rows)
                ):
                    raise RuntimeError(
                        "token-row transfer value verification failed"
                    )
                report.copied_tensors += 1
                report.token_rows_copied = len(valid_pairs)
                report.copied_parameters += len(valid_pairs) * target_tensor.shape[1]
                report.copied_keys.append(target_key)
                report.tensor_mappings.append(
                    _tensor_mapping(
                        target_key=target_key,
                        source_key=target_key,
                        method="token_rows",
                        target_tensor=target_tensor,
                        source_tensor=source_tensor,
                        copied_parameters=(
                            len(valid_pairs) * target_tensor.shape[1]
                        ),
                        selection_fingerprint=_json_fingerprint(valid_pairs),
                        copied_value_fingerprint=(
                            copied_value_fingerprint
                        ),
                    )
                )
            continue
        if target_key == "language.token_embedding.weight" and not copy_token_embeddings:
            report.skipped_by_policy += 1
            continue

        source_key = target_key
        block = _BLOCK.match(target_key)
        if block:
            block_component, target_index_text, suffix = block.groups()
            target_index = int(target_index_text)
            if target_index not in selected[block_component]:
                report.skipped_by_policy += 1
                continue
            source_count = source_counts[block_component]
            if source_count == 0:
                report.missing_source.append(target_key)
                continue
            source_index = (
                _typed_depth_map(
                    target_index,
                    target_language_types,
                    source_language_types,
                )
                if block_component == "language"
                and source_language_types
                else _depth_map(
                    target_index,
                    target_counts[block_component],
                    source_count,
                )
            )
            if source_index is None:
                report.skipped_by_policy += 1
                report.skipped_semantic.append(
                    {
                        "target": target_key,
                        "source": None,
                        "reason": "layer_type_mismatch",
                    }
                )
                continue
            source_key = f"{block_component}.blocks.{source_index}.{suffix}"

        if (
            require_attention_geometry
            and _LANGUAGE_ATTENTION.match(target_key)
            and not geometry_compatible
        ):
            report.skipped_by_policy += 1
            report.skipped_semantic.append(
                {
                    "target": target_key,
                    "source": source_key,
                    "reason": "attention_geometry_mismatch",
                }
            )
            continue
        if (
            require_attention_geometry
            and ".conv." in target_key
            and not short_convolution_compatible
        ):
            report.skipped_by_policy += 1
            report.skipped_semantic.append(
                {
                    "target": target_key,
                    "source": source_key,
                    "reason": "short_convolution_geometry_mismatch",
                }
            )
            continue
        source_tensor = source.get(source_key)
        if source_tensor is None:
            report.missing_source.append(source_key)
            continue
        structured = False
        original_source_tensor = source_tensor
        structured_selection_fingerprint = None
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mlp_match = _MLP_WEIGHT.match(target_key)
            if (
                shape_policy == "structured_mlp"
                and mlp_match
                and (
                    not require_attention_geometry
                    or mlp_operator_compatible
                )
            ):
                target_index = int(mlp_match.group(1))
                source_index = int(source_key.split(".")[2])
                target_prefix = f"language.blocks.{target_index}.mlp"
                source_prefix = f"language.blocks.{source_index}.mlp"
                group_key = (target_prefix, source_prefix)
                if group_key not in structured_cache:
                    structured_cache[group_key] = _structured_mlp_group(
                        target_prefix,
                        source_prefix,
                        target,
                        source,
                    )
                group = structured_cache[group_key]
                if group is not None and target_key in group[0]:
                    source_tensor = group[0][target_key]
                    structured = True
                    structured_selection_fingerprint = group[1][
                        "channel_index_fingerprint"
                    ]
                    if group_key not in recorded_structured_groups:
                        report.structured_groups.append(group[1])
                        recorded_structured_groups.add(group_key)
            if not structured:
                report.skipped_shape.append(
                    {
                        "target": target_key,
                        "source": source_key,
                        "target_shape": list(target_tensor.shape),
                        "source_shape": list(source_tensor.shape),
                    }
                )
                continue
        copied_value = source_tensor.to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        )
        copied_value_fingerprint = _tensor_value_fingerprint(copied_value)
        target_tensor.copy_(copied_value)
        if copied_value_fingerprint != _tensor_value_fingerprint(
            target_tensor
        ):
            raise RuntimeError("tensor transfer value verification failed")
        report.copied_tensors += 1
        if structured:
            report.structured_tensors += 1
            report.structured_parameters += target_tensor.numel()
        storage_key = (
            ("meta", target_key)
            if target_tensor.device.type == "meta"
            else (
                target_tensor.untyped_storage().data_ptr(),
                target_tensor.storage_offset(),
                target_tensor.numel(),
            )
        )
        if storage_key not in copied_storages:
            report.copied_parameters += target_tensor.numel()
            copied_storages.add(storage_key)
        report.copied_keys.append(target_key)
        report.tensor_mappings.append(
            _tensor_mapping(
                target_key=target_key,
                source_key=source_key,
                method="structured_mlp" if structured else "exact",
                target_tensor=target_tensor,
                source_tensor=original_source_tensor,
                copied_parameters=target_tensor.numel(),
                selection_fingerprint=structured_selection_fingerprint,
                copied_value_fingerprint=copied_value_fingerprint,
            )
        )
    student.load_state_dict(target)
    report.mapping_fingerprint = _json_fingerprint(report.tensor_mappings)
    report.copied_values_fingerprint = _json_fingerprint(
        [
            mapping.get("copied_value_fingerprint")
            for mapping in report.tensor_mappings
        ]
    )
    report.value_verified = all(
        isinstance(mapping.get("copied_value_fingerprint"), str)
        for mapping in report.tensor_mappings
    )
    return report
