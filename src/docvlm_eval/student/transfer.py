"""Auditable selective transfer for same-shape student, SigLIP, and Llama checkpoints."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn


_BLOCK = re.compile(r"^(vision|language)\.blocks\.(\d+)\.(.+)$")


@dataclass
class TransferReport:
    family: str
    fractions: dict[str, float]
    copied_tensors: int = 0
    copied_parameters: int = 0
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
    """Map common HF SigLIP/Llama state names into the native student's semantic names."""
    if family not in {"student", "siglip", "llama"}:
        raise ValueError("source family must be student, siglip, or llama")
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


@torch.no_grad()
def selective_transfer(
    student: nn.Module,
    source: Mapping[str, torch.Tensor],
    fractions: Mapping[str, float],
    *,
    family: str = "student",
    token_map: Mapping[int, int] | None = None,
    copy_token_embeddings: bool | None = None,
) -> TransferReport:
    """Copy exact-shape tensors under component and depth-fraction controls.

    Width-mismatched tensors are reported and left random. They must be learned by distillation;
    this function deliberately does not crop or interpolate hidden dimensions.
    """
    normalized_fractions = {
        component: float(fractions.get(component, 0.0))
        for component in ("vision", "language", "connector")
    }
    if any(not 0.0 <= value <= 1.0 for value in normalized_fractions.values()):
        raise ValueError("transfer fractions must be between zero and one")
    report = TransferReport(family=family, fractions=normalized_fractions)
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
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
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
