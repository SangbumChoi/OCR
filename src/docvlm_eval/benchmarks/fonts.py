"""Robust font loading for the rendered probes.

Hardcoding ``/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`` breaks on environments that put
fonts elsewhere (e.g. Colab) -> ``OSError: cannot open resource``. This searches common
locations, then matplotlib's bundled DejaVuSans, then falls back to PIL's default font so the
generators never crash.
"""

from __future__ import annotations

import glob
from functools import lru_cache


_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "/content",  # Colab working dir (in case a font is dropped there)
]


@lru_cache(maxsize=8)
def _find(bold: bool) -> str | None:
    want = "DejaVuSans-Bold" if bold else "DejaVuSans"
    # 1) exact DejaVu in common dirs (recursive)
    for d in _DIRS:
        for pat in (f"{d}/**/{want}.ttf", f"{d}/**/DejaVuSans*.ttf"):
            hits = glob.glob(pat, recursive=True)
            if hits:
                return sorted(hits)[0]
    # 2) any .ttf we can find
    for d in _DIRS:
        hits = glob.glob(f"{d}/**/*.ttf", recursive=True)
        if hits:
            return sorted(hits)[0]
    # 3) matplotlib's bundled DejaVuSans (always present if matplotlib is installed)
    try:
        import matplotlib.font_manager as fm

        return fm.findfont("DejaVu Sans Bold" if bold else "DejaVu Sans")
    except Exception:
        return None


def load_font(size: int, bold: bool = False):
    """Return a PIL ImageFont at ``size`` (truetype if any font is found, else default)."""
    from PIL import ImageFont

    path = _find(bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:  # Pillow >= 10 supports a size for the default bitmap font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_CJK_GLOB = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSerifCJK*.ttc",
    "/usr/share/fonts/**/ipag*.ttf",
    "/usr/share/fonts/**/*gothic*.ttf",
    "/usr/share/fonts/**/*CJK*.*",
    "/usr/share/fonts/**/*Han*.*",
]


@lru_cache(maxsize=4)
def _find_cjk() -> str | None:
    for pat in _CJK_GLOB:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def load_cjk_font(size: int):
    """A CJK-capable font (Korean/Japanese/Chinese). Falls back to the latin font if none —
    in which case CJK glyphs render as tofu and the generator should skip/flag CJK samples."""
    from PIL import ImageFont

    path = _find_cjk()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return load_font(size)


def have_cjk() -> bool:
    return _find_cjk() is not None

