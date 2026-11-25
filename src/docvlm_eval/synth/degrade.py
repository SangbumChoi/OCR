"""Photometric degradation presets (Augraphy). No geometry is applied, so ground-truth boxes
extracted from the clean render remain valid on the degraded copy. Augraphy is imported lazily.

Presets:
  scan        - flatbed scan: paper colour cast, texture, mild bleed-through, light JPEG
  photo       - phone photo of a page: uneven lighting + shadow, stronger JPEG
  fax         - fax / bad photocopy: dirty drum + heavy compression (worst legibility)
  historical  - aged manuscript: strong bleed-through, dark paper, stains
  screenshot  - digital-native (web/app capture): light compression + subtle noise, no paper
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from PIL import Image

PRESETS = ("scan", "photo", "fax", "historical", "screenshot")
RETRY_SEED_STRIDE = 1009


class DegradationError(RuntimeError):
    """Raised when no deterministic degradation attempt produces a valid image."""


def derive_degradation_seed(
    base_seed: int,
    document_key: str,
    variant: str | None,
    language: str,
) -> int:
    """Derive a stable, variant-specific seed so degradation patterns do not repeat."""
    material = f"{base_seed}:{document_key}:{variant}:{language}:degradation"
    return int(hashlib.md5(material.encode()).hexdigest(), 16) % (2**30)


def degrade(img: Image.Image, preset: str = "scan", seed: int | None = None) -> Image.Image | None:
    """Apply a degradation preset. Returns None if Augraphy/cv2 are unavailable."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
    try:
        import numpy as np
        from augraphy import (
            AugraphyPipeline, BadPhotoCopy, BleedThrough, Brightness, ColorPaper,
            DirtyDrum, Jpeg, LightingGradient, NoiseTexturize, ShadowCast, SubtleNoise,
        )
    except Exception as e:  # pragma: no cover - exercised only without the extra
        print("  [skip degrade]", e)
        return None
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def safe(cls, **kw):
        try:
            return cls(**kw)
        except Exception:
            return None

    presets = {
        "scan": ([safe(BleedThrough, p=0.4)],
                 [safe(ColorPaper, p=0.6), safe(NoiseTexturize, p=0.5)],
                 [safe(SubtleNoise, p=0.5), safe(Jpeg, quality_range=(45, 75))]),
        "photo": ([],
                  [safe(ColorPaper, p=0.3)],
                  [safe(LightingGradient), safe(ShadowCast, p=0.7),
                   safe(Brightness, brightness_range=(0.85, 1.1)), safe(Jpeg, quality_range=(35, 60))]),
        "fax": ([],
                [safe(DirtyDrum, p=0.7)],
                [safe(BadPhotoCopy, p=0.9), safe(Jpeg, quality_range=(20, 40))]),
        "historical": ([safe(BleedThrough, p=0.8)],
                       [safe(ColorPaper, p=0.9), safe(NoiseTexturize, p=0.8)],
                       [safe(Brightness, brightness_range=(0.7, 0.95)),
                        safe(ShadowCast, p=0.6), safe(SubtleNoise)]),
        "screenshot": ([], [],
                       [safe(SubtleNoise, p=0.4), safe(Jpeg, quality_range=(55, 80))]),
    }
    ink, paper, post = presets[preset]
    import numpy as np
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    pipe = AugraphyPipeline(
        ink_phase=[a for a in ink if a],
        paper_phase=[a for a in paper if a],
        post_phase=[a for a in post if a],
    )
    out = pipe(arr)
    return Image.fromarray(out[:, :, ::-1])


def degrade_with_retries(
    img: Image.Image,
    preset: str,
    *,
    base_seed: int,
    max_attempts: int = 3,
    validator: Callable[[Image.Image], object] | None = None,
) -> tuple[Image.Image, int, int]:
    """Return the first deterministic degradation that renders and passes ``validator``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    errors: list[str] = []
    for attempt_index in range(max_attempts):
        seed = base_seed + attempt_index * RETRY_SEED_STRIDE
        try:
            candidate = degrade(img, preset, seed=seed)
            if candidate is None:
                raise DegradationError("degradation dependencies are unavailable")
            if candidate.size != img.size:
                raise DegradationError(
                    f"geometry changed from {img.size} to {candidate.size}"
                )
            if validator is not None:
                validator(candidate)
            return candidate, seed, attempt_index + 1
        except Exception as error:
            errors.append(f"seed={seed}: {type(error).__name__}: {error}")
    preview = " | ".join(errors)
    raise DegradationError(
        f"no valid {preset!r} degradation after {max_attempts} attempt(s): {preview}"
    )
