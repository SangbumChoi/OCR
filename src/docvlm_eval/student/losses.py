"""Differentiable structured losses used by the native document VLM student."""

from __future__ import annotations

import torch


def decode_normalized_box(raw: torch.Tensor) -> torch.Tensor:
    """Map four unconstrained values to a valid normalized ``x1,y1,x2,y2`` box."""
    start = raw[..., :2].sigmoid()
    extent = raw[..., 2:].sigmoid() * (1.0 - start)
    return torch.cat((start, start + extent), dim=-1)


def generalized_box_iou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean generalized-IoU loss for aligned normalized ``xyxy`` boxes."""
    predicted = predicted.float()
    target = target.float()
    intersection_start = torch.maximum(predicted[..., :2], target[..., :2])
    intersection_end = torch.minimum(predicted[..., 2:], target[..., 2:])
    intersection_size = (intersection_end - intersection_start).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    predicted_size = (predicted[..., 2:] - predicted[..., :2]).clamp_min(0)
    target_size = (target[..., 2:] - target[..., :2]).clamp_min(0)
    predicted_area = predicted_size[..., 0] * predicted_size[..., 1]
    target_area = target_size[..., 0] * target_size[..., 1]
    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp_min(eps)

    enclosing_start = torch.minimum(predicted[..., :2], target[..., :2])
    enclosing_end = torch.maximum(predicted[..., 2:], target[..., 2:])
    enclosing_size = (enclosing_end - enclosing_start).clamp_min(0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(eps)
    return (1.0 - giou).mean()
