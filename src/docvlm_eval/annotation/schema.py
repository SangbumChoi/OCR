"""The **layered document annotation** record (DAR) — one image, every annotation layer.

Hand-annotated real documents are expensive, so a small corpus has to carry as much supervision
per image as possible. Instead of one dataset per task (an OCR set, a KIE set, a VQA set, ...),
every image carries ALL layers in a single record, and each higher layer **cites the ids of the
lower layers it depends on** instead of copying text or boxes:

    page          orientation / skew / doc type / acquisition      (is the page readable as-is?)
      layout      regions: class + box + reading order + parent    (where are the parts?)
        ocr       text lines: text + box + owning region            (what does it say?)
          table   cells: (row, col, span) -> ocr line ids            (how is it structured?)
          kie     fields: key -> normalized value + evidence ids    (what is the value?)
            understanding
                  caption, reading guide (how to interpret), reasoned QA whose
                  steps cite evidence ids                           (what does it mean?)

Because the layers are linked by id, (1) a validator can check that every value really is on the
page (see :mod:`.validate`), (2) any subset of layers can be exported as a training mix with the
SAME images (the "which layer matters?" ablation in :mod:`.ablation`), and (3) a wrong end-task
answer can be attributed to the lowest layer that failed (:mod:`.diagnose`).

Coordinate convention: boxes are ``[x1, y1, x2, y2]`` in **pixels of the stored image** (the frame
the grounding metric scores in). ``page.orientation`` says how the stored image is rotated relative
to upright; :func:`.geometry.rotate_annotation` moves image + every box between frames.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from typing import Any

SCHEMA_VERSION = "dar-1.0"

# layer names, bottom-up; also the keys of Provenance.layers
LAYERS = ("page", "layout", "ocr", "table", "kie", "understanding")

# how trustworthy a layer's content is
STATUS_ABSENT = "absent"        # not annotated (layer must then be empty)
STATUS_PSEUDO = "pseudo"        # model-generated, not reviewed
STATUS_HUMAN = "human"          # written or corrected by a person
STATUS_VERIFIED = "verified"    # human + second-pass review
STATUSES = (STATUS_ABSENT, STATUS_PSEUDO, STATUS_HUMAN, STATUS_VERIFIED)

ORIENTATIONS = (0, 90, 180, 270)

# Closed vocabularies keep labels comparable across annotators. Extend deliberately, not ad hoc.
REGION_CLASSES = (
    "title", "header", "footer", "text", "list", "kv_block", "table", "figure", "chart", "caption",
    "formula", "stamp", "signature", "handwriting", "checkbox", "logo", "barcode", "page_number",
    "ui_element", "other",
)
ACQUISITIONS = ("digital", "scan", "photo", "screenshot", "fax", "unknown")
VALUE_TYPES = ("string", "number", "amount", "date", "time", "id", "phone", "email", "address",
               "bool", "enum")
GUIDE_KINDS = ("structure", "convention", "disambiguation", "domain", "pitfall")
QA_TYPES = ("lookup", "multi_hop", "arithmetic", "comparison", "aggregation", "temporal",
            "abstain", "layout")
STEP_OPS = ("locate", "read", "parse", "compare", "compute", "aggregate", "lookup", "infer",
            "conclude")


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


def _from_dict(cls, d: dict | None):
    """Build a flat dataclass from a dict, ignoring unknown keys (forward-compatible reads)."""
    if d is None:
        return None
    names = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------- layer: page
@dataclass
class PageInfo:
    """Image-level facts. ``orientation`` = clockwise degrees the UPRIGHT page was rotated by to
    produce the stored image (0/90/180/270); rotate the stored image by ``-orientation`` to read it."""
    width: int
    height: int
    orientation: int = 0
    skew_deg: float = 0.0                    # residual small-angle tilt, + = clockwise
    doc_type: str = ""                       # e.g. "invoice", "bank_statement" (taxonomy names)
    domain: str = ""                         # e.g. "finance", "medical"
    acquisition: str = "unknown"             # ACQUISITIONS
    languages: list[str] = field(default_factory=list)   # ISO codes, dominant first
    quality: list[str] = field(default_factory=list)     # e.g. blur, glare, low_res, crease, stamp_over_text
    notes: str = ""


# --------------------------------------------------------------------------- layer: layout
@dataclass
class Region:
    """A layout block. ``order`` = reading-order index among regions (0-based, unique)."""
    id: str
    cls: str
    bbox: list[float]
    order: int | None = None
    parent: str | None = None                # enclosing region id (e.g. a cell block inside a form)
    label: str = ""                          # free-form name, e.g. "line-item table"


# --------------------------------------------------------------------------- layer: ocr
@dataclass
class TextLine:
    """One transcribed line. ``text`` is verbatim (no normalisation); ``region`` = owning block."""
    id: str
    text: str
    bbox: list[float]
    region: str | None = None
    poly: list[list[float]] | None = None    # optional 4+ point polygon for rotated / curved text
    handwritten: bool = False
    legibility: str = "clear"                # clear | partial | illegible  ('#' marks unreadable chars)
    language: str | None = None


# --------------------------------------------------------------------------- layer: table
@dataclass
class Cell:
    row: int
    col: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    header: bool = False
    lines: list[str] = field(default_factory=list)   # ocr line ids inside this cell


@dataclass
class Table:
    id: str
    n_rows: int
    n_cols: int
    cells: list[Cell] = field(default_factory=list)
    region: str | None = None                # the layout region with cls="table"

    def to_html(self) -> str:
        """Render the grid as HTML (header cells as ``<th>``), honouring spans."""
        covered: set[tuple[int, int]] = set()
        at = {(c.row, c.col): c for c in self.cells}
        rows = []
        for r in range(self.n_rows):
            tds = []
            for c in range(self.n_cols):
                if (r, c) in covered:
                    continue
                cell = at.get((r, c))
                if cell is None:
                    tds.append("<td></td>")
                    continue
                for dr in range(cell.row_span):
                    for dc in range(cell.col_span):
                        covered.add((r + dr, c + dc))
                tag = "th" if cell.header else "td"
                attrs = (f' rowspan="{cell.row_span}"' if cell.row_span > 1 else "") + \
                        (f' colspan="{cell.col_span}"' if cell.col_span > 1 else "")
                tds.append(f"<{tag}{attrs}>{cell.text}</{tag}>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
        return "<table>" + "".join(rows) + "</table>"

    @classmethod
    def from_dict(cls, d: dict) -> Table:
        return cls(id=d["id"], n_rows=d["n_rows"], n_cols=d["n_cols"], region=d.get("region"),
                   cells=[_from_dict(Cell, c) for c in d.get("cells", [])])


# --------------------------------------------------------------------------- layer: kie
@dataclass
class KIEField:
    """One extracted value. ``value`` is as printed; ``normalized`` is the canonical typed form the
    end task is scored on (e.g. "2305.00", "2024-03-01"). ``present=False`` records a schema field
    that is NOT on this document — an explicit abstain gold, which is as important as a value."""
    id: str
    key: str                                 # dotted schema key, e.g. "invoice.total"
    value: str = ""
    normalized: str | None = None
    value_type: str = "string"               # VALUE_TYPES
    value_lines: list[str] = field(default_factory=list)  # evidence: ocr line ids holding the value
    key_lines: list[str] = field(default_factory=list)    # the printed label, e.g. "TOTAL:"
    present: bool = True
    notes: str = ""

    def target(self) -> str:
        return self.normalized if self.normalized not in (None, "") else self.value


# --------------------------------------------------------------------------- layer: understanding
@dataclass
class GuideNote:
    """A piece of *how to read this document* knowledge, scoped to the ids it is about.

    This is the layer an individual annotator is uniquely placed to write: the conventions a model
    cannot see in any single string — "the TOTAL row already includes tax", "dates are DD/MM",
    "the second column header spans two sub-columns", "the stamp overlaps but is not part of the
    address"."""
    id: str
    text: str
    kind: str = "structure"                  # GUIDE_KINDS
    scope: list[str] = field(default_factory=list)


@dataclass
class ReasonStep:
    """One step of a reasoning chain. ``evidence`` = ids (lines / cells as ``<table>:<r>,<c>`` /
    fields / regions) this step reads; ``result`` = the intermediate value it produces."""
    op: str                                  # STEP_OPS
    text: str
    evidence: list[str] = field(default_factory=list)
    result: str | None = None


@dataclass
class ReasonedQA:
    id: str
    question: str
    answers: list[str] = field(default_factory=list)     # gold VARIANTS of one answer
    qa_type: str = "lookup"                  # QA_TYPES
    metric: str = "anls"
    evidence: list[str] = field(default_factory=list)    # ids the final answer rests on
    steps: list[ReasonStep] = field(default_factory=list)
    difficulty: int = 1                      # 1 = single lookup … 3 = multi-hop + arithmetic

    @classmethod
    def from_dict(cls, d: dict) -> ReasonedQA:
        d = dict(d)
        steps = [_from_dict(ReasonStep, s) for s in d.pop("steps", [])]
        out = _from_dict(cls, d)
        out.steps = steps
        return out


@dataclass
class Understanding:
    caption: str = ""                        # 1-3 sentences: what the document is and what it is for
    guide: list[GuideNote] = field(default_factory=list)
    qa: list[ReasonedQA] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> Understanding:
        d = d or {}
        return cls(caption=d.get("caption", ""),
                   guide=[_from_dict(GuideNote, g) for g in d.get("guide", [])],
                   qa=[ReasonedQA.from_dict(q) for q in d.get("qa", [])])

    def is_empty(self) -> bool:
        return not (self.caption or self.guide or self.qa)


# --------------------------------------------------------------------------- provenance
@dataclass
class LayerProvenance:
    status: str = STATUS_ABSENT
    annotator: str = ""
    seconds: float = 0.0                     # human time spent — feeds the value-per-hour read-out
    tool: str = ""                           # labeler / model name when status == pseudo


@dataclass
class Provenance:
    layers: dict[str, LayerProvenance] = field(
        default_factory=lambda: {name: LayerProvenance() for name in LAYERS})
    created: str = ""
    updated: str = ""
    license: str = ""
    source: str = ""                         # where the image came from (own capture, public set, ...)

    @classmethod
    def from_dict(cls, d: dict | None) -> Provenance:
        d = d or {}
        layers = {name: LayerProvenance() for name in LAYERS}
        for name, lp in (d.get("layers") or {}).items():
            layers[name] = _from_dict(LayerProvenance, lp)
        return cls(layers=layers, created=d.get("created", ""), updated=d.get("updated", ""),
                   license=d.get("license", ""), source=d.get("source", ""))


# --------------------------------------------------------------------------- the record
@dataclass
class DocAnnotation:
    doc_id: str
    image: str                               # path relative to the annotation file (or absolute)
    page: PageInfo
    layout: list[Region] = field(default_factory=list)
    ocr: list[TextLine] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    kie: list[KIEField] = field(default_factory=list)
    understanding: Understanding = field(default_factory=Understanding)
    provenance: Provenance = field(default_factory=Provenance)
    split: str = "train"                     # train | heldout — decided per IMAGE, never per QA
    schema_version: str = SCHEMA_VERSION

    # ---- lookups
    def index(self) -> dict[str, Any]:
        """Every addressable id -> object. Table cells are addressed as ``<table_id>:<row>,<col>``."""
        idx: dict[str, Any] = {}
        for coll in (self.layout, self.ocr, self.tables, self.kie, self.understanding.guide,
                     self.understanding.qa):
            for obj in coll:
                idx[obj.id] = obj
        for t in self.tables:
            for c in t.cells:
                idx[f"{t.id}:{c.row},{c.col}"] = c
        return idx

    def layer_status(self, layer: str) -> str:
        return self.provenance.layers.get(layer, LayerProvenance()).status

    def has_layer(self, layer: str) -> bool:
        """True when the layer is annotated (status != absent) AND carries content."""
        if self.layer_status(layer) == STATUS_ABSENT:
            return False
        return {
            "page": True, "layout": bool(self.layout), "ocr": bool(self.ocr),
            "table": bool(self.tables), "kie": bool(self.kie),
            "understanding": not self.understanding.is_empty(),
        }[layer]

    def lines_in_reading_order(self) -> list[TextLine]:
        """Lines sorted by their region's reading order, then top-to-bottom, left-to-right."""
        order = {r.id: (r.order if r.order is not None else 10**6) for r in self.layout}
        return sorted(self.ocr, key=lambda ln: (order.get(ln.region, 10**6),
                                                round(ln.bbox[1]), ln.bbox[0]))

    def full_text(self) -> str:
        return "\n".join(ln.text for ln in self.lines_in_reading_order())

    def evidence_text(self, ids: list[str]) -> str:
        """Concatenate the text behind evidence ids (lines, cells, fields)."""
        idx = self.index()
        out = []
        for i in ids:
            o = idx.get(i)
            if isinstance(o, (TextLine, Cell)):
                out.append(o.text)
            elif isinstance(o, KIEField):
                out.append(o.value)
        return " ".join(out)

    def evidence_bbox(self, ids: list[str]) -> list[float] | None:
        """Union box of the evidence ids that carry geometry (lines, regions, cells via lines)."""
        idx = self.index()
        boxes = []
        for i in ids:
            o = idx.get(i)
            if isinstance(o, (TextLine, Region)):
                boxes.append(o.bbox)
            elif isinstance(o, Cell):
                boxes += [idx[ln].bbox for ln in o.lines if isinstance(idx.get(ln), TextLine)]
            elif isinstance(o, KIEField):
                boxes += [idx[ln].bbox for ln in o.value_lines if isinstance(idx.get(ln), TextLine)]
        if not boxes:
            return None
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]

    # ---- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ocr"] = [_drop_none(ln) for ln in d["ocr"]]
        d["layout"] = [_drop_none(r) for r in d["layout"]]
        return d

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> DocAnnotation:
        return cls(
            doc_id=d["doc_id"], image=d["image"], page=_from_dict(PageInfo, d["page"]),
            layout=[_from_dict(Region, r) for r in d.get("layout", [])],
            ocr=[_from_dict(TextLine, ln) for ln in d.get("ocr", [])],
            tables=[Table.from_dict(t) for t in d.get("tables", [])],
            kie=[_from_dict(KIEField, f) for f in d.get("kie", [])],
            understanding=Understanding.from_dict(d.get("understanding")),
            provenance=Provenance.from_dict(d.get("provenance")),
            split=d.get("split", "train"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, s: str) -> DocAnnotation:
        return cls.from_dict(json.loads(s))


def skeleton(doc_id: str, image: str, width: int, height: int, **page_kw) -> DocAnnotation:
    """An empty record for a new image — the starting point of the annotation pipeline."""
    return DocAnnotation(doc_id=doc_id, image=image, page=PageInfo(width=width, height=height,
                                                                   **page_kw))
