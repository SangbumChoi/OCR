"""DocBuilder — a small builder so synthetic documents *produce their ground truth*.

The problem this solves: when GT is typed by hand next to the rendered HTML, the two drift (a
value shown on the page disagrees with the label). Here every value is declared **once** through a
primitive that both (a) emits the HTML that renders it and (b) registers the matching ground
truth, and spotting boxes are resolved from the rendered PDF — so the pixels and the labels can
never disagree.

Each primitive returns nothing; it appends HTML and records GT. Call ``.build(dpi)`` to render
and resolve boxes, getting back ``(PIL image, gt dict)``. The GT schema is uniform across cases:

    type, stressors, anchor_metric,
    fields{key:value}, spotting{key:[x1,y1,x2,y2]}, table_html, selection{group:[...]},
    redacted{key:hidden}, reading_order[...], probes[{kind,question,expected}],
    source, render{dpi,size_px,page_count}

Redacted values are stored under ``redacted`` and never rendered, so they are absent from the
image on purpose — that is the anti-hallucination (correct-or-abstain) target.
"""

from __future__ import annotations

import html as _html
from collections import Counter
from dataclasses import dataclass, field

from PIL import Image

from .render import render_html, resolve_boxes


def esc(s: object) -> str:
    return _html.escape(str(s))


@dataclass
class DocBuilder:
    doc_type: str
    stressors: list[str]
    anchor_metric: str
    css: str = ""
    page: str = "A5"            # CSS @page size token, e.g. "A5", "A4", "90mm 58mm", "1280px 900px"
    margin: str = "12mm"

    def __post_init__(self):
        self._html: list[str] = []
        self.fields: dict[str, object] = {}
        self.selection: dict[str, list[str]] = {}
        self.redacted: dict[str, str] = {}
        self.reading_order: list = []
        self.probes: list[dict] = []
        self.table_html: str | None = None
        self._spots: list[tuple[str, str, int]] = []   # (key, text, occurrence)
        self._occ: Counter = Counter()

    # -- low level ---------------------------------------------------------
    def raw(self, html: str) -> None:
        self._html.append(html)

    def _spot(self, key: str, text: str) -> None:
        """Register that `key`'s box is the rendered position of `text` (occurrence-aware)."""
        occ = self._occ[text]
        self._occ[text] += 1
        self._spots.append((key, str(text), occ))

    def spot(self, key: str, text: str) -> None:
        """Public: register a spotting box for text already emitted via raw()/interpolation."""
        self._spot(key, text)

    # -- content primitives ------------------------------------------------
    def title(self, text: str, *, level: int = 1) -> None:
        self.raw(f"<h{level} class=t>{esc(text)}</h{level}>")

    def line(self, html: str, *, cls: str = "") -> None:
        self.raw(f"<p class='{cls}'>{html}</p>")

    def field(self, label: str | None, value: str, *, key: str, spot: bool = False,
              cls: str = "", lang: str = "en") -> None:
        """A labelled value. Registers fields[key]=value; optionally a spotting box on the value."""
        self.fields[key] = value
        lab = f"<b>{esc(label)}:</b> " if label else ""
        self.raw(f"<p class='fld {cls}' lang='{lang}'>{lab}<span class=v>{esc(value)}</span></p>")
        if spot:
            self._spot(key, value)

    def transcript(self, text: str, *, key: str = "transcript", cls: str = "", lang: str = "en",
                   spot: bool = False) -> None:
        """A block of text whose exact string is the GT (for NED/CER)."""
        self.fields[key] = text
        self.raw(f"<p class='tx {cls}' lang='{lang}'>{esc(text)}</p>")
        if spot:
            self._spot(key, text)

    def table(self, header: list[str], rows: list[list[str]], *, key: str = "table",
              footer: list[str] | None = None, spot_cells: list[tuple[int, int]] | None = None,
              cls: str = "") -> None:
        """Render an HTML table AND store the TEDS-gold HTML (identical structure by construction).

        spot_cells: list of (row, col) into the *body* rows to register a spotting box for, keyed
        ``{key}_r{row}c{col}`` on the cell text.
        """
        def row(cells, tag):
            return "<tr>" + "".join(f"<{tag}>{esc(c)}</{tag}>" for c in cells) + "</tr>"
        body = "".join(row(r, "td") for r in rows)
        foot = row(footer, "td") if footer else ""
        gold = f"<table>{row(header, 'td')}{body}{foot}</table>"
        self.table_html = gold
        self.fields[key + "_rows"] = len(rows)
        self.raw(f"<table class='{cls}'>{row(header,'th')}{body}{foot}</table>")
        for (r, c) in (spot_cells or []):
            if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
                self._spot(f"{key}_r{r}c{c}", rows[r][c])

    def checkboxes(self, group: str, options: list[tuple[str, bool]], *, cls: str = "") -> None:
        """Selection marks. Registers selection[group]=checked-labels and a box per option label."""
        self.selection[group] = [t for t, c in options if c]
        for t, c in options:
            mark = "&#9746;" if c else "&#9744;"   # ☒ / ☐
            self.raw(f"<div class='chk {cls}'><span class=box>{mark}</span> {esc(t)}</div>")
            self._spot(f"{group}:{t}", t)

    def redaction(self, prefix_html: str, hidden_value: str, *, key: str, bar: str = "█" * 12,
                  suffix_html: str = "") -> None:
        """Render `prefix [black bar] suffix`; the hidden value is GT-only (never drawn) -> the
        abstain target. Also adds an abstain probe automatically."""
        self.redacted[key] = hidden_value
        self.raw(f"<p>{prefix_html}<span class=rd>{bar}</span>{suffix_html}</p>")
        self.probe("abstain", f"What is the {key.replace('_', ' ')}?",
                   "[redacted] — blacked out; must NOT invent a value")

    def order(self, items: list, *, note: str = "") -> None:
        """Declare the canonical reading order (sequence GT). Rendering is done by other primitives."""
        self.reading_order = items if not note else {"order": items, "note": note}

    def probe(self, kind: str, question: str, expected: str) -> None:
        self.probes.append({"kind": kind, "question": question, "expected": expected})

    def task(self, text: str) -> None:
        self.fields["_task"] = text

    # -- render ------------------------------------------------------------
    def _full_css(self) -> str:
        base = f"""
        @page {{ size: {self.page}; margin: {self.margin}; }}
        * {{ box-sizing: border-box; }}
        body {{ font-family:'Liberation Sans',sans-serif; color:#111; font-size:11px; }}
        h1,h2,h3 {{ margin:0 0 6px; }}
        table {{ border-collapse:collapse; width:100%; font-size:10px; }}
        td,th {{ border:1px solid #888; padding:3px 5px; text-align:left; }}
        .chk {{ margin:4px 0; font-size:13px; }} .box {{ font-size:15px; margin-right:6px; }}
        .rd {{ background:#111; color:#111; }}
        .muted {{ color:#555; }}
        """
        return base + self.css

    def build(self, dpi: int = 150) -> tuple[Image.Image, dict]:
        rr = render_html("".join(self._html), self._full_css(), dpi=dpi)
        try:
            spotting = resolve_boxes(rr, self._spots)
            gt = {
                "type": self.doc_type,
                "stressors": list(self.stressors),
                "anchor_metric": self.anchor_metric,
                "fields": dict(self.fields),
                "source": "SYNTHETIC (docvlm_eval.synth) — renders the task; not official data",
                "render": {"dpi": dpi, "size_px": list(rr.image.size), "page_count": rr.page_count},
            }
            if spotting:
                gt["spotting"] = spotting
            if self.table_html:
                gt["table_html"] = self.table_html
            if self.selection:
                gt["selection"] = self.selection
            if self.redacted:
                gt["redacted"] = self.redacted
            if self.reading_order:
                gt["reading_order"] = self.reading_order
            if self.probes:
                gt["probes"] = self.probes
            return rr.image.copy(), gt
        finally:
            rr.close()
