"""Auditable selective transfer for compatible native and Hugging Face checkpoints."""

from __future__ import annotations

import hashlib
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
_SHAPE_POLICIES = {"exact", "structured_mlp"}


@dataclass
class TransferReport:
    family: str
    fractions: dict[str, float]
    shape_policy: str
    copied_tensors: int = 0
    copied_parameters: int = 0
    structured_tensors: int = 0
    structured_parameters: int = 0
    structured_groups: list[dict[str, Any]] = field(default_factory=list)
    token_rows_copied: int = 0
    skipped_by_policy: int = 0
    skipped_shape: list[dict[str, Any]] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)
    copied_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    report = TransferReport(
        family=family,
        fractions=normalized_fractions,
        shape_policy=shape_policy,
    )
    source = canonicalize_source_state(source, family)
    target = student.state_dict()
    if copy_token_embeddings is None:
        copy_token_embeddings = family == "student"
    source_keys = list(source)
    target_keys = list(target)
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
                report.copied_tensors += 1
                report.token_rows_copied = len(valid_pairs)
                report.copied_parameters += len(valid_pairs) * target_tensor.shape[1]
                report.copied_keys.append(target_key)
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
            source_index = _depth_map(
                target_index,
                target_counts[block_component],
                source_count,
            )
            source_key = f"{block_component}.blocks.{source_index}.{suffix}"

        source_tensor = source.get(source_key)
        if source_tensor is None:
            report.missing_source.append(source_key)
            continue
        structured = False
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mlp_match = _MLP_WEIGHT.match(target_key)
            if shape_policy == "structured_mlp" and mlp_match:
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
        target_tensor.copy_(
            source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        )
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
    student.load_state_dict(target)
    return report
