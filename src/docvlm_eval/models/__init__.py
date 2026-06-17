"""Model adapters + registry. Import :func:`build_model` to instantiate by key."""

from .base import GenConfig, ModelAdapter
from .registry import build_model, list_models, register

__all__ = ["GenConfig", "ModelAdapter", "build_model", "list_models", "register"]
