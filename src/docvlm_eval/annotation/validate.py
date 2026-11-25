"""Consistency checks for a :class:`~.schema.DocAnnotation`.

Hand annotation drifts in predictable ways — a typo in the KIE value that is not in the OCR line,
a QA citing a deleted line, a box drawn on the un-rotated image. Because every higher layer cites
lower-layer ids, those drifts are mechanically detectable. ``validate`` returns a list of
:class:`Issue`; ``errors`` break the record (refuse to export), ``warnings`` deserve a look.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schema import (
    ACQUISITIONS,
    GUIDE_KINDS,
    LAYERS,
    ORIENTATIONS,
    QA_TYPES,
    REGION_CLASSES,
    STATUS_ABSENT,
    STATUSES,
    STEP_OPS,
    VALUE_TYPES,
    Cell,
    DocAnnotation,
    Region,
    TextLine,
)

ERROR, WARNING = "error", "warning"


@dataclass
class Issue:
    level: str
    layer: str
    ref: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.layer}/{self.ref}: {self.message}"


def _canon(s: str) -> str:
    """Loose text key for 'is this value on the page?' — case/space/punctuation-insensitive."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\W_]+", "", s)


def _box_ok(b, W: int, H: int, tol: float = 2.0) -> str | None:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return "bbox must be [x1, y1, x2, y2]"
    x1, y1, x2, y2 = b
    if not (x1 < x2 and y1 < y2):
        return f"degenerate bbox {list(b)}"
    if x1 < -tol or y1 < -tol or x2 > W + tol or y2 > H + tol:
        return f"bbox {list(b)} outside the {W}x{H} image (drawn on a different frame?)"
    return None


def validate(ann: DocAnnotation) -> list[Issue]:
    out: list[Issue] = []

    def add(level, layer, ref, msg):
        out.append(Issue(level, layer, ref, msg))

    p = ann.page
    W, H = p.width, p.height
    if W <= 0 or H <= 0:
        add(ERROR, "page", ann.doc_id, "page width/height must be positive")
    if p.orientation not in ORIENTATIONS:
        add(ERROR, "page", ann.doc_id, f"orientation {p.orientation} not in {ORIENTATIONS}")
    if p.acquisition not in ACQUISITIONS:
        add(WARNING, "page", ann.doc_id, f"acquisition {p.acquisition!r} not in {ACQUISITIONS}")
    if abs(p.skew_deg) > 45:
        add(WARNING, "page", ann.doc_id, "skew > 45 deg — express it as orientation + residual skew")

    # ---- provenance: absent layers must be empty, statuses must be known
    for name in LAYERS:
        st = ann.layer_status(name)
        if st not in STATUSES:
            add(ERROR, name, "provenance", f"unknown status {st!r}")
    content = {"layout": ann.layout, "ocr": ann.ocr, "table": ann.tables, "kie": ann.kie,
               "understanding": not ann.understanding.is_empty()}
    for name, has in content.items():
        if has and ann.layer_status(name) == STATUS_ABSENT:
            add(ERROR, name, "provenance", "layer has content but status is 'absent' — set "
                                            "pseudo/human/verified so exports know to trust it")

    # ---- ids: unique across the record
    seen: dict[str, str] = {}
    for layer, coll in (("layout", ann.layout), ("ocr", ann.ocr), ("table", ann.tables),
                        ("kie", ann.kie), ("understanding", ann.understanding.guide),
                        ("understanding", ann.understanding.qa)):
        for o in coll:
            if not o.id:
                add(ERROR, layer, "?", "missing id")
            elif o.id in seen:
                add(ERROR, layer, o.id, f"duplicate id (also in {seen[o.id]})")
            else:
                seen[o.id] = layer
    idx = ann.index()

    def resolve(layer, ref, ids, kinds, what):
        for i in ids:
            o = idx.get(i)
            if o is None:
                add(ERROR, layer, ref, f"{what} cites unknown id {i!r}")
            elif kinds and not isinstance(o, kinds):
                add(ERROR, layer, ref, f"{what} id {i!r} is a {type(o).__name__}, expected "
                                       f"{'/'.join(k.__name__ for k in kinds)}")

    # ---- layout
    orders = [r.order for r in ann.layout if r.order is not None]
    if len(orders) != len(set(orders)):
        add(ERROR, "layout", "order", "reading-order indices are not unique")
    for r in ann.layout:
        if r.cls not in REGION_CLASSES:
            add(WARNING, "layout", r.id, f"class {r.cls!r} not in the closed vocabulary")
        if (m := _box_ok(r.bbox, W, H)):
            add(ERROR, "layout", r.id, m)
        if r.parent:
            resolve("layout", r.id, [r.parent], (Region,), "parent")

    # ---- ocr
    for ln in ann.ocr:
        if (m := _box_ok(ln.bbox, W, H)):
            add(ERROR, "ocr", ln.id, m)
        if not ln.text.strip() and ln.legibility != "illegible":
            add(WARNING, "ocr", ln.id, "empty text on a line not marked illegible")
        if ln.region:
            reg = idx.get(ln.region)
            if not isinstance(reg, Region):
                add(ERROR, "ocr", ln.id, f"region {ln.region!r} is not a layout region")
            elif not _box_ok(ln.bbox, W, H):
                cx, cy = (ln.bbox[0] + ln.bbox[2]) / 2, (ln.bbox[1] + ln.bbox[3]) / 2
                x1, y1, x2, y2 = reg.bbox
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    add(WARNING, "ocr", ln.id, f"line centre lies outside its region {reg.id}")

    # ---- tables
    for t in ann.tables:
        if t.region:
            resolve("table", t.id, [t.region], (Region,), "region")
        grid: dict[tuple[int, int], tuple[int, int]] = {}
        for c in t.cells:
            ref = f"{t.id}:{c.row},{c.col}"
            if c.row < 0 or c.col < 0 or c.row + c.row_span > t.n_rows or c.col + c.col_span > t.n_cols:
                add(ERROR, "table", ref, f"cell (with span) outside the {t.n_rows}x{t.n_cols} grid")
                continue
            for dr in range(c.row_span):
                for dc in range(c.col_span):
                    k = (c.row + dr, c.col + dc)
                    if k in grid:
                        add(ERROR, "table", ref, f"overlaps cell {grid[k]}")
                    grid[k] = (c.row, c.col)
            resolve("table", ref, c.lines, (TextLine,), "cell.lines")
            if c.lines and c.text and _canon(c.text) != _canon(ann.evidence_text(c.lines)):
                add(WARNING, "table", ref, "cell text differs from its OCR lines")

    # ---- kie
    for f in ann.kie:
        if f.value_type not in VALUE_TYPES:
            add(WARNING, "kie", f.id, f"value_type {f.value_type!r} not in {VALUE_TYPES}")
        resolve("kie", f.id, f.value_lines + f.key_lines, (TextLine, Cell), "evidence")
        if not f.present:
            if f.value or f.value_lines:
                add(ERROR, "kie", f.id, "present=False but a value/evidence is given")
            continue
        if not f.value:
            add(ERROR, "kie", f.id, "present field has an empty value")
        if not f.value_lines:
            add(WARNING, "kie", f.id, "no value_lines — the value cannot be grounded or spotted")
        elif _canon(f.value) not in _canon(ann.evidence_text(f.value_lines)):
            add(WARNING, "kie", f.id, f"value {f.value!r} not found in its evidence lines "
                                      f"{ann.evidence_text(f.value_lines)!r}")

    # ---- understanding
    u = ann.understanding
    for g in u.guide:
        if g.kind not in GUIDE_KINDS:
            add(WARNING, "understanding", g.id, f"guide kind {g.kind!r} not in {GUIDE_KINDS}")
        resolve("understanding", g.id, g.scope, None, "scope")
    for q in u.qa:
        if q.qa_type not in QA_TYPES:
            add(WARNING, "understanding", q.id, f"qa_type {q.qa_type!r} not in {QA_TYPES}")
        if not q.question.strip():
            add(ERROR, "understanding", q.id, "empty question")
        if not q.answers:
            add(ERROR, "understanding", q.id, "no gold answer (use 'not stated' + qa_type=abstain)")
        if q.qa_type != "abstain" and not q.evidence:
            add(WARNING, "understanding", q.id, "no evidence ids — answer is ungrounded")
        resolve("understanding", q.id, q.evidence, None, "evidence")
        for k, s in enumerate(q.steps):
            if s.op not in STEP_OPS:
                add(WARNING, "understanding", f"{q.id}.s{k}", f"op {s.op!r} not in {STEP_OPS}")
            resolve("understanding", f"{q.id}.s{k}", s.evidence, None, "step evidence")
        last = q.steps[-1].result if q.steps else None
        if last is not None and q.answers and _canon(last) not in {_canon(a) for a in q.answers}:
            add(WARNING, "understanding", q.id, "last step result is not one of the answers")
    return out


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == ERROR]


def assert_valid(ann: DocAnnotation) -> None:
    errs = errors(validate(ann))
    if errs:
        raise ValueError(f"{ann.doc_id}: {len(errs)} annotation error(s):\n" +
                         "\n".join(str(e) for e in errs))


__all__ = ["ERROR", "WARNING", "Issue", "assert_valid", "errors", "validate"]
