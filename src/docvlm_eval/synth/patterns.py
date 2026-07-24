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
from dataclasses import dataclass

from PIL import Image

from .hard_layout import layout_fingerprint
from .hard_locale import HARD_DOCUMENT_LANGUAGES, hard_text
from .render import (
    prepare_color_probe_fallback,
    render_html,
    resolve_boxes,
)


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
    language: str = "en"
    layout_family: str | None = None

    def __post_init__(self):
        self._html: list[str] = []
        self.fields: dict[str, object] = {}
        self.selection: dict[str, list[str]] = {}
        self.redacted: dict[str, str] = {}
        self.reading_order: list = []
        self.probes: list[dict] = []
        self.table_html: str | None = None
        self.qas: list[dict] = []
        self._spots: list[tuple[str, str, int]] = []   # (key, text, occurrence)
        self._occ: Counter = Counter()
        self._derivations: list = []   # model-free understanding-GT requests (resolved in build)
        self._fulltext_q: str | None = None   # full-document OCR target (answer filled from render)
        # per-field metadata consumed by DocSample.from_builder_gt (A4 language, A7 small-text)
        self.field_lang: dict[str, str] = {}
        self.field_role: dict[str, str] = {}
        self.field_font_px: dict[str, float] = {}
        self.semantic_graph: dict | None = None
        self.difficulty: dict | None = None

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
              cls: str = "", lang: str | None = None, role: str = "kie-value",
              font_px: float | None = None) -> None:
        """A labelled value. Registers fields[key]=value; optionally a spotting box on the value.

        ``lang`` (A4) and ``font_px`` (A7 small-text slice) are recorded as per-field metadata."""
        lang = lang or self.language
        self.fields[key] = value
        self.field_lang[key] = lang
        self.field_role[key] = role
        if font_px is not None:
            self.field_font_px[key] = font_px
        lab = f"<b>{esc(label)}:</b> " if label else ""
        self.raw(f"<p class='fld {cls}' lang='{lang}'>{lab}<span class=v>{esc(value)}</span></p>")
        if spot:
            self._spot(key, value)

    def transcript(self, text: str, *, key: str = "transcript", cls: str = "",
                   lang: str | None = None,
                   spot: bool = False, role: str = "transcript",
                   font_px: float | None = None) -> None:
        """A block of text whose exact string is the GT (for NED/CER)."""
        lang = lang or self.language
        self.fields[key] = text
        self.field_lang[key] = lang
        self.field_role[key] = role
        if font_px is not None:
            self.field_font_px[key] = font_px
        self.raw(f"<p class='tx {cls}' lang='{lang}'>{esc(text)}</p>")
        if spot:
            self._spot(key, text)

    def table(self, header: list[str], rows: list[list[str]], *, key: str = "table",
              footer: list[str] | None = None, spot_cells: list[tuple[int, int]] | None = None,
              cls: str = "", region: str | None = "the table") -> None:
        """Render an HTML table AND store the TEDS-gold HTML (identical structure by construction).

        spot_cells: list of (row, col) into the *body* rows to register a spotting box for, keyed
        ``{key}_r{row}c{col}`` on the cell text.
        region: if given (default "the table"), auto-register an L1-region derivation that bounds the
        whole table — derived from the header row + the last body row (the two short, reliable rows
        that define the rectangle's top and bottom), first-instance to avoid stray matches.
        """
        def row(cells, tag):
            return "<tr>" + "".join(f"<{tag}>{esc(c)}</{tag}>" for c in cells) + "</tr>"
        body = "".join(row(r, "td") for r in rows)
        foot = row(footer, "td") if footer else ""
        gold = f"<table>{row(header, 'td')}{body}{foot}</table>"
        self.table_html = gold
        self.fields[key + "_rows"] = len(rows)
        self.field_lang[key + "_rows"] = self.language
        self.raw(f"<table class='{cls}'>{row(header,'th')}{body}{foot}</table>")
        for (r, c) in (spot_cells or []):
            if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
                self._spot(f"{key}_r{r}c{c}", rows[r][c])
        if region and rows:
            # region = header row (top edge + full width) ∪ first column (left edge + full height);
            # union of top-right + bottom-left corners gives the whole table rectangle. We avoid the
            # last *row* because short cells there (e.g. a Qty "1") match digits elsewhere on the page
            # and would drag the box outside the table.
            from .derive import Derivation
            col0 = [str(r[0]) for r in rows] + ([str(footer[0])] if footer else [])
            members = [str(c) for c in header] + col0
            # pad so the region wraps the whole table (ruled border + the empty right column),
            # not just the cell-text extent.
            self._derivations.append(Derivation("region", texts=members, label=region,
                                                key=f"{key}_region", pad_frac=0.012))

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

    def bubble(self, text: str, *, side: str = "in", key: str | None = None) -> str:
        """A chat bubble. Renders the row+bubble, appends `text` to reading_order, optional spot.
        `side` ('in'/'out') drives CSS the case supplies. Returns the text for convenience."""
        if not isinstance(self.reading_order, list):
            self.reading_order = []
        self.reading_order.append(text)
        self.raw(f"<div class='row {side}'><div class='bub {side}'>{esc(text)}</div></div>")
        if key:
            self._spot(key, text)
        return text

    def panel(self, text: str, *, index: int, side: str = "left") -> str:
        """A comic/webtoon panel with one speech bubble. Renders panel, appends to reading_order."""
        if not isinstance(self.reading_order, list):
            self.reading_order = []
        self.reading_order.append(text)
        self.raw(f"<div class=panel><span class=pno>{index}</span>"
                 f"<div class='bubble {side}'>{esc(text)}</div></div>")
        return text

    def want_fulltext(self, question: str = "Transcribe all the text in this document in reading order.") -> None:
        """Request a FULL-DOCUMENT OCR target. The answer is filled in ``build()`` from the rendered
        page's exact text layer (PyMuPDF), so the GT is correct by construction — this trains/evaluates
        whole-page reading (the bulk of document understanding), not just field spotting. Best for
        digital-native / clean-Latin docs where the text layer extracts faithfully."""
        self._fulltext_q = question

    def qa(self, question: str, answer, *, metric: str = "anls", answer_type: str = "kie",
           key: str | None = None, concise: bool = True, rationale: str | None = None,
           languages: list[str] | None = None,
           evidence_keys: list[str] | None = None, derived: bool = False,
           graph_query_id: str | None = None) -> None:
        """Register an answerable (question, answer) pair over content already rendered, so the case
        can be turned into eval Samples. `answer` may be a string or a list of acceptable strings.

        ``rationale`` is the A2 chain-of-thought supervision target; ``key`` (if it matches a
        registered spot) links the answer to its A1 bounding box."""
        ans = answer if isinstance(answer, list) else [answer]
        q = question
        if concise and metric not in ("grounding", "teds"):
            question_language = (languages or [self.language])[0]
            locale = (
                question_language
                if question_language in HARD_DOCUMENT_LANGUAGES
                else "en"
            )
            q += f" {hard_text(locale, 'concise')}"
        qa_languages = languages or [self.language]
        self.qas.append({"key": key, "question": q, "answers": ans,
                         "metric": metric, "answer_type": answer_type,
                         **({"rationale": rationale} if rationale else {}),
                         **({"evidence_keys": evidence_keys} if evidence_keys else {}),
                         **({"derived": True} if derived else {}),
                         **({"graph_query_id": graph_query_id} if graph_query_id else {}),
                         "languages": qa_languages})

    def table_reason(self, header: list, rows: list, *, label: str = "the table", n: int = 3) -> None:
        """Auto-generate ``n`` varied, model-free REASONING QAs over a typed table (count / sum / mean
        / extreme / argmax-lookup / threshold / ordinal / row-compare / date-extreme), each with a
        rationale. Driven by ``synth.reasoning`` — the question subset varies per document."""
        from .reasoning import table_questions
        for d in table_questions(header, rows, label=label, n=n):
            self.qa(d["question"], d["answers"], metric=d["metric"], answer_type=d["answer_type"],
                    rationale=d.get("rationale"))

    def seq_reason(self, items: list, *, attr: str, label: str = "items",
                   value_names: dict | None = None, n: int = 3) -> None:
        """Auto-generate model-free reasoning over a labelled SEQUENCE (e.g. chat bubbles by sender,
        comic panels by side): total, count-per-group, which group has more. ``items`` = list of
        (group_value, ...); ``value_names`` maps raw values to friendly words."""
        from .reasoning import sequence_questions
        for d in sequence_questions(items, attr=attr, label=label, value_names=value_names, n=n):
            self.qa(d["question"], d["answers"], metric=d["metric"], answer_type=d["answer_type"],
                    rationale=d.get("rationale"))

    def probe(self, kind: str, question: str, expected: str) -> None:
        self.probes.append({"kind": kind, "question": question, "expected": expected})

    # -- model-free UNDERSTANDING ground truth (resolved from the render at build time) -----
    # These derive non-OCR GT — where / how-many / totals — plus the reasoning that justifies it,
    # with no external model (see docvlm_eval.synth.derive). Each is gold by construction.
    def ask_where(self, text: str, *, label: str | None = None, occurrence: int = 0,
                  key: str | None = None) -> None:
        """'Where is <text>?' → its bounding box (derived from the rendered PDF)."""
        from .derive import Derivation
        self._derivations.append(Derivation("locate", text=text, label=label,
                                            occurrence=occurrence, key=key))

    def ask_count(self, text: str, *, key: str | None = None) -> None:
        """'How many times does <text> appear?' → exact occurrence count + the hit positions."""
        from .derive import Derivation
        self._derivations.append(Derivation("count", text=text, key=key))

    def ask_region(self, label: str, texts: list[str], *, key: str | None = None) -> None:
        """'Where is the <label> (e.g. the table)?' → bbox enclosing all the member strings."""
        from .derive import Derivation
        self._derivations.append(Derivation("region", texts=list(texts), label=label, key=key))

    def ask_aggregate(self, label: str, values, *, op: str = "sum", key: str | None = None) -> None:
        """'What is the <label>?' → arithmetic over known values, with the working as rationale."""
        from .derive import Derivation
        self._derivations.append(
            Derivation("aggregate", values=[float(v) for v in values], op=op, label=label, key=key))

    def task(self, text: str) -> None:
        self.fields["_task"] = text

    # -- render ------------------------------------------------------------
    def _full_css(self) -> str:
        base = f"""
        @page {{ size: {self.page}; margin: {self.margin}; }}
        * {{ box-sizing: border-box; }}
        html, body {{ margin:0; padding:0; }}  /* drop UA 8px body margin: a page-tall card (ID/passport) would otherwise overflow to page 2 and lose its bottom strip (e.g. the MRZ) */
        /* CJK/Arabic fallbacks are named explicitly (not just the `sans-serif` generic) so multilingual
           content — e.g. a Korean name in any case, not only the CJK-specific ones — renders into the
           searchable text layer reliably across environments; otherwise glyphs tofu and box derivation
           (ask_where/locate) silently finds nothing. */
        body {{ font-family:'Liberation Sans','Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC',
                'Noto Sans Arabic','Noto Sans Hebrew',sans-serif; color:#111; font-size:11px; }}
        h1,h2,h3 {{ margin:0 0 6px; }}
        table {{ border-collapse:collapse; width:100%; font-size:10px; }}
        td,th {{ border:1px solid #888; padding:3px 5px; text-align:left; }}
        .chk {{ margin:4px 0; font-size:13px; }} .box {{ font-size:15px; margin-right:6px; }}
        .rd {{ background:#111; color:#111; }}
        .muted {{ color:#555; }}
        """
        return base + self.css

    def build(
        self,
        dpi: int = 150,
        *,
        color_probe_fallback: bool = True,
    ) -> tuple[Image.Image, dict]:
        html = "".join(self._html)
        css = self._full_css()
        rr = render_html(html, css, dpi=dpi)
        try:
            if color_probe_fallback:
                required_occurrences: dict[str, int] = {}
                for _key, text, occurrence in self._spots:
                    required_occurrences[text] = max(
                        required_occurrences.get(text, 0),
                        occurrence + 1,
                    )
                for derivation in self._derivations:
                    if derivation.kind in {"locate", "count"}:
                        required_occurrences[derivation.text] = max(
                            required_occurrences.get(derivation.text, 0),
                            derivation.occurrence + 1,
                        )
                    elif derivation.kind == "region":
                        for text in derivation.texts or []:
                            required_occurrences[text] = max(
                                required_occurrences.get(text, 0),
                                1,
                            )
                prepare_color_probe_fallback(
                    rr,
                    html,
                    css,
                    required_occurrences,
                )
            spotting = resolve_boxes(rr, self._spots)
            # resolve model-free understanding GT (where/how-many/totals) against the open render
            if self._derivations:
                from .derive import resolve as _resolve_derivation
                for d in self._derivations:
                    qa = _resolve_derivation(rr, d)
                    if qa is None:  # validation: requested text not on the page -> warn, don't fake
                        print(f"  [warn] derivation {d.kind}({d.text or d.label!r}) found nothing "
                              f"in {self.doc_type} — skipped")
                        continue
                    self.qas.append(qa)
            if self._fulltext_q:                       # full-document OCR target from the exact render
                txt = rr.full_text()
                if txt:
                    self.fields["full_text"] = txt
                    self.field_lang["full_text"] = self.language
                    self.qas.append({"key": "full_text", "question": self._fulltext_q,
                                     "answers": [txt], "metric": "ned", "answer_type": "ocr-full",
                                     "languages": [self.language]})
                else:
                    print(f"  [warn] full_text empty for {self.doc_type} — skipped")
            render = {
                "dpi": dpi,
                "size_px": list(rr.image.size),
                "page_count": rr.page_count,
                "box_resolver": (
                    "pdf_text_then_color_probe"
                    if color_probe_fallback
                    else "pdf_text"
                ),
                "color_probe_fallback_count": len(
                    rr.color_probe_fallbacks
                ),
            }
            if self.layout_family is not None:
                render["layout_family"] = self.layout_family
                render["layout_fingerprint"] = layout_fingerprint(
                    self.doc_type,
                    self.layout_family,
                )
            gt = {
                "type": self.doc_type,
                "stressors": list(self.stressors),
                "anchor_metric": self.anchor_metric,
                "fields": dict(self.fields),
                "source": "SYNTHETIC (docvlm_eval.synth) — renders the task; not official data",
                "render": render,
            }
            if spotting:
                gt["spotting"] = spotting
            if self.table_html:
                gt["table_html"] = self.table_html
            if self.qas:
                gt["qa"] = self.qas
            if self.selection:
                gt["selection"] = self.selection
            if self.redacted:
                gt["redacted"] = self.redacted
            if self.reading_order:
                gt["reading_order"] = self.reading_order
            if self.probes:
                gt["probes"] = self.probes
            if self.semantic_graph is not None:
                gt["semantic_graph"] = self.semantic_graph
            if self.difficulty is not None:
                gt["difficulty"] = self.difficulty
            return rr.image.copy(), gt
        finally:
            rr.close()
