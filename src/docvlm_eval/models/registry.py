"""Model registry.

Maps a short ``--model`` key to an adapter class. The pipeline calls :func:`build_model`;
``scripts/evaluate.py`` calls :func:`list_models`. New models are registered with the
``@register("key")`` decorator inside their adapter module.
"""

from __future__ import annotations

from typing import Callable, Type

from .base import ModelAdapter

_REGISTRY: dict[str, Type[ModelAdapter]] = {}


def register(key: str) -> Callable[[Type[ModelAdapter]], Type[ModelAdapter]]:
    def deco(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate model key: {key}")
        cls.key = key
        _REGISTRY[key] = cls
        return cls

    return deco


def build_model(key: str, **kwargs) -> ModelAdapter:
    # Import adapters lazily so a missing optional dep (e.g. one model's custom code)
    # doesn't break the whole registry.
    from . import (  # noqa: F401
        internvl,
        smolvlm,
        llava_ov,
        got_ocr,
        florence2,
        paddleocr_vl,
        h2ovl,
        ovis,
    )

    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    from . import (  # noqa: F401
        internvl,
        smolvlm,
        llava_ov,
        got_ocr,
        florence2,
        paddleocr_vl,
        h2ovl,
        ovis,
    )

    return sorted(_REGISTRY)
