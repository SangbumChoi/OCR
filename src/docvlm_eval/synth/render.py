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

from PIL import Image


@dataclass
class RenderResult:
    """A rendered page plus the open PDF doc (kept so boxes can be queried)."""

    image: Image.Image
    dpi: int
    _doc: object = field(repr=False, default=None)
    _page0: object = field(repr=False, default=None)

    @property
    def zoom(self) -> float:
        return self.dpi / 72.0

    @property
    def page_count(self) -> int:
        return self._doc.page_count if self._doc else 1

    def search_boxes(self, text: str) -> list[list[int]]:
        """All pixel boxes [x1,y1,x2,y2] for `text` on page 0, in reading order."""
        if not self._page0 or not text:
            return []
        z = self.zoom
        return [[round(r.x0 * z), round(r.y0 * z), round(r.x1 * z), round(r.y1 * z)]
                for r in self._page0.search_for(text)]

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
