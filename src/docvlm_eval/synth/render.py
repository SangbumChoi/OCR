"""Render HTML/CSS to a raster image with exact ground-truth boxes.

The whole point of the synth package: the image and its ground truth come from the *same*
source. We render HTML/CSS with WeasyPrint -> PDF, rasterize the PDF with PyMuPDF, and read text
positions straight out of the PDF so spotting boxes are pixel-exact (and stay valid on a
photometrically-degraded copy, because no geometry is changed).

Heavy deps (weasyprint, pymupdf) are imported lazily so importing docvlm_eval stays cheap and the
core test suite runs without the [synth] extra.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class RenderResult:
    """A rendered page plus the open PDF doc (kept so boxes can be queried)."""

    image: Image.Image
    dpi: int
    _doc: object = field(repr=False, default=None)
    _page0: object = field(repr=False, default=None)
    _fallback_boxes: dict[str, list[list[int]]] = field(
        repr=False,
        default_factory=dict,
    )

    @property
    def zoom(self) -> float:
        return self.dpi / 72.0

    @property
    def page_count(self) -> int:
        return self._doc.page_count if self._doc else 1

    def native_search_boxes(self, text: str) -> list[list[int]]:
        """Return boxes from the PDF text layer without any fallback."""
        if not self._page0 or not text:
            return []
        z = self.zoom
        return [[round(r.x0 * z), round(r.y0 * z), round(r.x1 * z), round(r.y1 * z)]
                for r in self._page0.search_for(text)]

    def search_boxes(self, text: str) -> list[list[int]]:
        """Return all pixel boxes for text, preferring an installed exact fallback."""
        if text in self._fallback_boxes:
            return self._fallback_boxes[text]
        return self.native_search_boxes(text)

    @property
    def color_probe_fallbacks(self) -> tuple[str, ...]:
        """Texts whose PDF lookup was replaced by a color-probe lookup."""
        return tuple(self._fallback_boxes)

    def full_text(self) -> str:
        """The page's complete text in reading order (PyMuPDF) — the exact rendered text, so a
        correct-by-construction full-document OCR target (no hand-assembly)."""
        if not self._page0:
            return ""
        raw = self._page0.get_text("text") or ""
        return "\n".join(ln.strip() for ln in raw.splitlines() if ln.strip())

    def close(self) -> None:
        if self._doc:
            self._doc.close()
            self._doc = None


def render_html(html: str, css: str = "", dpi: int = 150, base_url: str = ".") -> RenderResult:
    """HTML/CSS -> RenderResult (page-0 raster at `dpi` + queryable PDF). Caller must close()."""
    import fitz  # PyMuPDF
    from weasyprint import HTML

    pdf = HTML(string=f"<style>{css}</style>{html}", base_url=str(base_url)).write_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return RenderResult(image=img, dpi=dpi, _doc=doc, _page0=page)


def _probe_color(index: int) -> tuple[int, int, int]:
    """Return a deterministic high-contrast marker color."""
    return (
        32 + (index * 73) % 192,
        32 + (index * 109) % 192,
        32 + (index * 151) % 192,
    )


def _mark_text_nodes(
    html: str,
    targets: Iterable[str],
) -> tuple[str, dict[str, list[tuple[int, int, int]]]]:
    """Wrap visible target occurrences in layout-neutral solid-color spans."""
    from bs4 import BeautifulSoup, Comment, NavigableString

    ordered_targets = tuple(
        sorted({text for text in targets if text}, key=lambda text: (-len(text), text))
    )
    soup = BeautifulSoup(html, "html.parser")
    colors: dict[str, list[tuple[int, int, int]]] = {
        text: [] for text in ordered_targets
    }
    marker_index = 1
    for node in list(soup.find_all(string=True)):
        if isinstance(node, Comment) or (
            node.parent
            and node.parent.name in {"script", "style", "template"}
        ):
            continue
        source = str(node)
        cursor = 0
        parts: list[object] = []
        while cursor < len(source):
            candidates = []
            for target in ordered_targets:
                position = source.find(target, cursor)
                if position >= 0:
                    candidates.append((position, -len(target), target))
            if not candidates:
                parts.append(NavigableString(source[cursor:]))
                break
            position, _negative_length, target = min(candidates)
            if position > cursor:
                parts.append(NavigableString(source[cursor:position]))
            color = _probe_color(marker_index)
            marker_index += 1
            colors[target].append(color)
            span = soup.new_tag("span")
            rgb = ",".join(str(channel) for channel in color)
            span["data-docvlm-color-probe"] = str(marker_index - 1)
            span["style"] = (
                f"background-color:rgb({rgb});color:rgb({rgb});"
                "box-decoration-break:clone;-webkit-box-decoration-break:clone"
            )
            span.string = target
            parts.append(span)
            cursor = position + len(target)
        if parts:
            node.replace_with(*parts)
    return str(soup), colors


def _visible_occurrence_counts(
    html: str,
    targets: Iterable[str],
) -> dict[str, int]:
    """Count target substrings in rendered text nodes with one HTML parse."""
    from bs4 import BeautifulSoup, Comment

    ordered_targets = tuple(dict.fromkeys(text for text in targets if text))
    counts = {text: 0 for text in ordered_targets}
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.find_all(string=True):
        if isinstance(node, Comment) or (
            node.parent
            and node.parent.name in {"script", "style", "template"}
        ):
            continue
        source = str(node)
        for text in ordered_targets:
            counts[text] += source.count(text)
    return counts


def _extract_probe_boxes(
    original: Image.Image,
    probe: Image.Image,
    colors: dict[str, list[tuple[int, int, int]]],
) -> dict[str, list[list[int]]]:
    """Recover one union box per marked occurrence from a probe raster."""
    if probe.size != original.size:
        raise RuntimeError("color-probe render changed the page geometry")
    original_array = np.asarray(original.convert("RGB"), dtype=np.int16)
    probe_array = np.asarray(probe.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(probe_array - original_array), axis=2) > 4
    resolved: dict[str, list[list[int]]] = {}
    for text, text_colors in colors.items():
        boxes: list[list[int]] = []
        for color in text_colors:
            distance = np.max(
                np.abs(
                    probe_array
                    - np.asarray(color, dtype=np.int16)[None, None, :]
                ),
                axis=2,
            )
            ys, xs = np.nonzero(changed & (distance <= 3))
            if xs.size:
                boxes.append(
                    [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ]
                )
        if boxes:
            resolved[text] = boxes
    return resolved


def _run_color_probe(
    rr: RenderResult,
    html: str,
    css: str,
    texts: Iterable[str],
    *,
    base_url: str,
) -> dict[str, list[list[int]]]:
    probe_html, colors = _mark_text_nodes(html, texts)
    if not any(colors.values()):
        return {}
    probe = render_html(
        probe_html,
        css,
        dpi=rr.dpi,
        base_url=base_url,
    )
    try:
        return _extract_probe_boxes(rr.image, probe.image, colors)
    finally:
        probe.close()


def prepare_color_probe_fallback(
    rr: RenderResult,
    html: str,
    css: str,
    required_occurrences: dict[str, int],
    *,
    base_url: str = ".",
) -> tuple[str, ...]:
    """Install color-probe boxes for PDF lookups that miss required occurrences.

    Targets are marked together first. Any overlapping target that could not be
    marked in that batch receives one isolated probe pass.
    """
    visible_counts = _visible_occurrence_counts(
        html,
        required_occurrences,
    )
    missing: list[str] = []
    expected_counts: dict[str, int] = {}
    for text, count in required_occurrences.items():
        if not text:
            continue
        required = max(1, count)
        native_count = len(rr.native_search_boxes(text))
        visible_count = visible_counts[text]
        expected_counts[text] = max(required, visible_count)
        if native_count < required or (
            visible_count >= required
            and native_count != visible_count
        ):
            missing.append(text)
    if not missing:
        return ()
    resolved = _run_color_probe(
        rr,
        html,
        css,
        missing,
        base_url=base_url,
    )
    unresolved = [
        text
        for text in missing
        if len(resolved.get(text, ()))
        < expected_counts[text]
    ]
    for text in unresolved:
        isolated = _run_color_probe(
            rr,
            html,
            css,
            [text],
            base_url=base_url,
        )
        if isolated.get(text):
            resolved[text] = isolated[text]
    for text in missing:
        boxes = resolved.get(text, [])
        if len(boxes) >= expected_counts[text]:
            rr._fallback_boxes[text] = boxes
    return rr.color_probe_fallbacks


def resolve_boxes(rr: RenderResult, targets: list[tuple[str, str, int]]) -> dict[str, list[int]]:
    """Resolve (key, text, occurrence_index) -> pixel box.

    Handles repeated strings: for each distinct text we search once (hits come back in reading
    order) and hand out the n-th hit to the n-th registered occurrence. Missing hits are skipped
    (the key simply won't appear), so callers can assert presence in tests.
    """
    by_text: dict[str, list[list[int]]] = {}
    for _key, text, _occ in targets:
        if text not in by_text:
            by_text[text] = rr.search_boxes(text)
    out: dict[str, list[int]] = {}
    for key, text, occ in targets:
        hits = by_text.get(text, [])
        if occ < len(hits):
            out[key] = hits[occ]
    return out


def occurrence_counter() -> Counter:
    """A counter for assigning occurrence indices to repeated spot targets."""
    return Counter()
