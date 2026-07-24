"""Differentiable structured losses used by the native document VLM student."""

from __future__ import annotations

import torch


BOX_IOU_LOSSES = frozenset({"giou", "diou", "ciou"})


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


def distance_box_iou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean distance-IoU loss for aligned normalized ``xyxy`` boxes."""
    iou, center_penalty, _, _ = _distance_iou_terms(
        predicted,
        target,
        eps,
    )
    return (1.0 - iou + center_penalty).mean()


def _distance_iou_terms(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    predicted_center = (predicted[..., :2] + predicted[..., 2:]) * 0.5
    target_center = (target[..., :2] + target[..., 2:]) * 0.5
    center_distance_squared = (
        predicted_center - target_center
    ).square().sum(dim=-1)
    enclosing_start = torch.minimum(predicted[..., :2], target[..., :2])
    enclosing_end = torch.maximum(predicted[..., 2:], target[..., 2:])
    enclosing_diagonal_squared = (
        enclosing_end - enclosing_start
    ).square().sum(dim=-1)
    center_penalty = (
        center_distance_squared
        / enclosing_diagonal_squared.clamp_min(eps)
    )
    return iou, center_penalty, predicted_size, target_size


def complete_box_iou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean complete-IoU loss for aligned normalized ``xyxy`` boxes."""
    iou, center_penalty, predicted_size, target_size = _distance_iou_terms(
        predicted,
        target,
        eps,
    )
    aspect_delta = torch.atan(
        target_size[..., 0].clamp_min(eps)
        / target_size[..., 1].clamp_min(eps)
    ) - torch.atan(
        predicted_size[..., 0].clamp_min(eps)
        / predicted_size[..., 1].clamp_min(eps)
    )
    aspect_penalty = 4.0 / (torch.pi**2) * aspect_delta.square()
    with torch.no_grad():
        aspect_weight = aspect_penalty / (
            1.0 - iou + aspect_penalty
        ).clamp_min(eps)
    return (
        1.0
        - iou
        + center_penalty
        + aspect_weight * aspect_penalty
    ).mean()


def box_iou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    """Dispatch one supported IoU-family loss."""
    if kind == "giou":
        return generalized_box_iou_loss(predicted, target)
    if kind == "diou":
        return distance_box_iou_loss(predicted, target)
    if kind == "ciou":
        return complete_box_iou_loss(predicted, target)
    raise ValueError(f"unsupported box IoU loss {kind!r}")
