"""Model-free derivation of *understanding* ground truth from a render.

The OCR ground truth (what string is where) falls out of the render for free. The harder, more
valuable GT is the **non-OCR understanding** layer — *where is this word / table?*, *how many times
does it appear?*, *what is the total?* — and the **reasoning** that justifies each answer. This
module derives all of that **without any external model**, purely from:

  * the rendered PDF's exact text positions (``RenderResult.search_boxes`` — PyMuPDF), and
  * the structured values the generator already knows (numbers to sum, etc.).

Everything here is deterministic and exact, so the derived answers are gold by construction. Each
deriver returns the answer **and** a human-readable ``rationale`` string, so the same call produces
both the supervision target and the chain-of-thought that explains it (A2).

Design goals the project cares about:
  * **no external model** — geometry + arithmetic only;
  * **validated** — derivers report when a requested word is absent (so GT is never silently wrong);
  * **efficient** — callers cache one search per distinct string (see ``DerivationResolver``);
  * **flexible** — new operations are one small function + one ``OP`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dto import BBox


# --------------------------------------------------------------------------- spatial (from render)
def word_boxes(rr, text: str) -> list[BBox]:
    """All pixel boxes of ``text`` on the page, in reading order (model-free, from the PDF)."""
    return [b for b in (BBox.from_list(x) for x in rr.search_boxes(text)) if b]


def locate(rr, text: str, occurrence: int = 0) -> BBox | None:
    """Box of the ``occurrence``-th hit of ``text`` (None if not found)."""
    boxes = word_boxes(rr, text)
    return boxes[occurrence] if 0 <= occurrence < len(boxes) else None


def count_occurrences(rr, text: str) -> tuple[int, list[BBox]]:
    """How many times ``text`` is rendered, plus the hit boxes (the evidence for the rationale)."""
    boxes = word_boxes(rr, text)
    return len(boxes), boxes


def union_box(boxes: list[BBox | None]) -> BBox | None:
    """Axis-aligned bounding box enclosing all the given boxes — e.g. a whole table/region."""
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return BBox(min(b.x1 for b in bs), min(b.y1 for b in bs),
               max(b.x2 for b in bs), max(b.y2 for b in bs))


def region_box(rr, texts: list[str]) -> BBox | None:
    """Bounding box of a region defined by the strings it contains (header + cells of a table,
    etc.). Model-free: union of every hit of every string."""
    all_boxes: list[BBox | None] = []
    for t in texts:
        all_boxes.extend(word_boxes(rr, t))
    return union_box(all_boxes)


# --------------------------------------------------------------------------- arithmetic (values)
def _fmt(n: float) -> str:
    return f"{n:g}"


def aggregate(values: list[float], op: str = "sum") -> tuple[float, str]:
    """Reduce numeric ``values`` with ``op`` and return ``(result, rationale)``.

    ``rationale`` shows the working (e.g. ``"45 + 80 + 20.5 = 145.5"``) so it doubles as the A2
    chain-of-thought target. Raises on an unknown op or empty input — fail loud, never fabricate."""
    nums = [float(v) for v in values]
    if not nums:
        raise ValueError("aggregate() needs at least one value")
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}; choose from {sorted(_OPS)}")
    result, rationale = _OPS[op](nums)
    return result, rationale


def _op_sum(ns):
    r = sum(ns)
    return r, f"{' + '.join(_fmt(n) for n in ns)} = {_fmt(r)}"


def _op_max(ns):
    r = max(ns)
    return r, f"max({', '.join(_fmt(n) for n in ns)}) = {_fmt(r)}"


def _op_min(ns):
    r = min(ns)
    return r, f"min({', '.join(_fmt(n) for n in ns)}) = {_fmt(r)}"


def _op_mean(ns):
    r = sum(ns) / len(ns)
    return r, f"({' + '.join(_fmt(n) for n in ns)}) / {len(ns)} = {_fmt(r)}"


def _op_count(ns):  # count of values (arithmetic counterpart of count_occurrences)
    return float(len(ns)), f"there are {len(ns)} value(s)"


_OPS: dict[str, Callable[[list[float]], tuple[float, str]]] = {
    "sum": _op_sum, "max": _op_max, "min": _op_min, "mean": _op_mean, "count": _op_count,
}


# --------------------------------------------------------------------------- request → resolved QA
@dataclass
class Derivation:
    """A request for one piece of understanding GT, resolved against a render into a QA dict.

    ``kind`` in {locate, count, region, aggregate}. The resolver fills ``answer``/``rationale``/
    ``box`` and the question text. ``found`` is False when a spatial request matched nothing (the
    caller can warn and skip, so GT is never silently empty)."""

    kind: str
    text: str = ""                       # the word/phrase (locate/count)
    texts: list[str] | None = None       # region member strings
    values: list[float] | None = None    # aggregate inputs
    op: str = "sum"                       # aggregate op
    label: str | None = None             # human label for region/aggregate questions
    occurrence: int = 0
    key: str | None = None
    answer_type: str | None = None       # override the default readable axis tag


_DEFAULT_TYPE = {"locate": "L1-locate", "count": "H-count",
                 "region": "L1-region", "aggregate": "H1-aggregate"}


def resolve(rr, d: Derivation) -> dict | None:
    """Resolve one :class:`Derivation` against an open render into a flat ``qa`` dict
    (``question/answers/metric/answer_type/rationale[/box]``), or None if it found nothing.

    The qa dict is tagged ``derived=True`` so a config can include/exclude the whole understanding
    layer as one ablation switch."""
    W, H = rr.image.size
    atype = d.answer_type or _DEFAULT_TYPE[d.kind]

    if d.kind in ("locate", "region"):
        box = locate(rr, d.text, d.occurrence) if d.kind == "locate" else region_box(rr, d.texts or [])
        if box is None:
            return None
        target = d.label or (f"the text '{d.text}'" if d.kind == "locate" else "the region")
        q = (f"Where is {target} located? Return its bounding box as [x1, y1, x2, y2] in pixel "
             f"coordinates. The image is {W}x{H} pixels.")
        ans = f"{box.x1},{box.y1},{box.x2},{box.y2};{W},{H}"
        rat = f"{target} is rendered at [{box.x1}, {box.y1}, {box.x2}, {box.y2}] on the {W}x{H}px page."
        return {"key": d.key, "question": q, "answers": [ans], "metric": "grounding",
                "answer_type": atype, "rationale": rat, "box": box.to_list(), "derived": True}

    if d.kind == "count":
        n, boxes = count_occurrences(rr, d.text)
        coords = "; ".join(f"[{b.x1},{b.y1}]" for b in boxes) or "nowhere"
        q = f"How many times does '{d.text}' appear in the document? Answer with a number."
        rat = f"'{d.text}' is found {n} time(s), at {coords}."
        return {"key": d.key, "question": q, "answers": [str(n)], "metric": "exact",
                "answer_type": atype, "rationale": rat, "derived": True}

    if d.kind == "aggregate":
        result, working = aggregate(d.values or [], d.op)
        label = d.label or f"the {d.op}"
        q = f"What is {label}? Answer with a number."
        rat = f"{label.capitalize()}: {working}."
        # integers render without a trailing .0; keep a couple of acceptable surface forms
        forms = [_fmt(result)]
        if float(result).is_integer():
            forms.append(str(int(result)))
        return {"key": d.key, "question": q, "answers": forms, "metric": "relaxed_acc",
                "answer_type": atype, "rationale": rat, "derived": True}

    raise ValueError(f"unknown derivation kind {d.kind!r}")
