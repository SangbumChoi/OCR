"""Error attribution: when a model gets a value wrong, WHICH layer broke first?

Because every KIE field and QA cites its evidence lines, one wrong answer can be traced down the
stack with a handful of extra probe questions on the same image:

    orientation probe wrong            -> "orientation"   (it never saw the page upright)
    evidence lines mis-read (CER)      -> "ocr"           (right place, wrong characters)
    evidence not localised (IoU)       -> "localization"  (read the wrong place)
    all of the above fine              -> "reasoning"     (read it right, concluded wrong)

Aggregated over a held-out set this gives the *failure profile* of a model — the evidence for
which annotation layer to invest in next. Ask the model the questions from
:func:`diagnostic_samples`, collect ``{sample_id: prediction}``, then call :func:`attribute`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..metrics.text import score_sample
from ..schema import Sample
from .export import _Emitter, _gbox, _kie, _orientation, _qa, _resolve_image
from .schema import DocAnnotation

ORDER = ("orientation", "ocr", "localization", "reasoning")


def _evidence_lines(ann: DocAnnotation, ids: list[str]) -> list[str]:
    """Expand evidence ids (lines / cells / fields) into the ocr line ids beneath them."""
    idx = ann.index()
    out: list[str] = []
    for i in ids:
        o = idx.get(i)
        if o is None:
            continue
        kind = type(o).__name__
        if kind == "TextLine":
            out.append(i)
        elif kind == "Cell":
            out += o.lines
        elif kind == "KIEField":
            out += o.value_lines
    return list(dict.fromkeys(out))


def _items(ann: DocAnnotation):
    """(item_id, end-task sample suffix, evidence line ids) for every KIE field and QA."""
    for f in ann.kie:
        if f.present:
            yield f.id, f"kie_{f.id}", list(f.value_lines)
    for q in ann.understanding.qa:
        ev = list(q.evidence) + [i for s in q.steps for i in s.evidence]
        yield q.id, f"qa_{q.id}", _evidence_lines(ann, ev)


def diagnostic_samples(ann: DocAnnotation, base_dir: str | Path | None = None) -> list[Sample]:
    """End-task questions + the probes needed to attribute their errors."""
    e = _Emitter(ann, _resolve_image(ann, base_dir))
    _orientation(e)
    _kie(e)
    _qa(e, "answer")
    W, H = ann.page.width, ann.page.height
    idx = ann.index()
    seen = set()
    for _, _, lines in _items(ann):
        for lid in lines:
            ln = idx.get(lid)
            if lid in seen or ln is None:
                continue
            seen.add(lid)
            b = ln.bbox
            e.add("ocr", f"read_{lid}", f"Read the text inside the box {b[0]:.0f},{b[1]:.0f},"
                  f"{b[2]:.0f},{b[3]:.0f} (image is {W}x{H} pixels). Answer with the text only.",
                  [ln.text], task="recognition", metric="cer_sim", ref=lid)
            e.add("line_grounding", f"where_{lid}", f'Where is the text "{ln.text}" in the '
                  "document? Return the bounding box as x1,y1,x2,y2.", [_gbox(b, W, H)],
                  task="localization", metric="grounding", ref=lid)
    return e.out


def attribute(ann: DocAnnotation, preds: dict[str, str], *, ocr_min: float = 0.9,
              iou_min: float = 0.5, correct_min: float = 1.0) -> list[dict]:
    """Per end-task item: is it correct, and if not, the LOWEST layer whose probe failed.

    ``correct_min`` defaults to 1.0: an extracted value is right or wrong — ANLS 0.71 for
    "1289.00" vs "1298.00" is a wrong total, not a partially right one.

    ``preds`` maps sample_id (as produced by :func:`diagnostic_samples`) -> model output. Probes
    without a prediction are skipped (treated as passing), so a partial probe run still works."""
    samples = {s.sample_id: s for s in diagnostic_samples(ann)}

    def score(suffix: str) -> float | None:
        sid = f"{ann.doc_id}:{suffix}"
        if sid not in samples or sid not in preds:
            return None
        s = samples[sid]
        return score_sample(s.metric, preds[sid], s.answers)

    orient = score("orient")
    out = []
    for item_id, suffix, lines in _items(ann):
        end = score(suffix)
        if end is None:
            continue
        row = {"doc_id": ann.doc_id, "item": item_id, "score": end, "correct": end >= correct_min,
               "failed_layer": None}
        if not row["correct"]:
            ocr = [x for x in (score(f"read_{lid}") for lid in lines) if x is not None]
            loc = [x for x in (score(f"where_{lid}") for lid in lines) if x is not None]
            if orient is not None and orient < 1.0:
                row["failed_layer"] = "orientation"
            elif ocr and min(ocr) < ocr_min:
                row["failed_layer"] = "ocr"
            elif loc and min(loc) < iou_min:
                row["failed_layer"] = "localization"
            else:
                row["failed_layer"] = "reasoning"
        out.append(row)
    return out


def failure_profile(rows: list[dict]) -> dict:
    """Aggregate :func:`attribute` rows -> accuracy and the share of errors per failed layer."""
    n = len(rows)
    wrong = [r for r in rows if not r["correct"]]
    c = Counter(r["failed_layer"] for r in wrong)
    return {"n_items": n, "accuracy": (n - len(wrong)) / n if n else 0.0,
            "n_errors": len(wrong),
            "error_share": {k: c.get(k, 0) / len(wrong) if wrong else 0.0 for k in ORDER}}
