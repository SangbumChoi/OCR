"""Turn a :class:`~.schema.DocAnnotation` into training / evaluation data, one LAYER at a time.

Every exporter emits the repo's flat :class:`~docvlm_eval.schema.Sample` (the JSONL the fine-tune
and eval pipelines already read), tagged ``meta.layer`` / ``meta.task`` so a mix can be sliced
afterwards. :func:`export_samples` picks which layers to emit — that switch IS the layer ablation
(:mod:`.ablation`): the images never change, only which annotation layers become targets.

Answer formats follow the existing metrics:
  * grounding answers are ``"x1,y1,x2,y2;W,H"`` in stored-image pixels (metrics/grounding.py);
  * KIE is a JSON object of normalized values (``"null"`` for ``present=False`` fields);
  * reasoning in ``chain`` style is ``<steps>\\nAnswer: <answer>``; ``answer`` style is the bare answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Sample
from .schema import DocAnnotation

# the layer families an ablation arm can switch on (a superset of schema.LAYERS: the
# understanding layer splits into three independently useful targets, KIE into value / spotting)
EXPORT_LAYERS = ("orientation", "doctype", "layout", "ocr", "line_grounding", "table", "kie",
                 "kie_spotting", "qa", "caption", "guide")
REASONING_STYLES = ("answer", "chain", "guided_chain")

_ABSENT = "null"


def _resolve_image(ann: DocAnnotation, base_dir: str | Path | None) -> str:
    p = Path(ann.image)
    if not p.is_absolute() and base_dir is not None:
        p = (Path(base_dir) / p).resolve()
    return str(p)


def _gbox(b, W, H) -> str:
    return f"{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f};{W},{H}"


class _Emitter:
    def __init__(self, ann: DocAnnotation, image: str):
        self.ann, self.image, self.out = ann, image, []

    def add(self, layer: str, suffix: str, question: str, answers: list[str], *, task: str,
            metric: str, **meta):
        answers = [a for a in answers if a is not None and str(a).strip()]
        if not answers:
            return
        self.out.append(Sample(
            sample_id=f"{self.ann.doc_id}:{suffix}", image_path=self.image, question=question,
            answers=answers, answer_type=task, metric=metric,
            meta={"layer": layer, "task": task, "doc_id": self.ann.doc_id,
                  "doc_type": self.ann.page.doc_type, "split": self.ann.split,
                  "status": self.ann.layer_status(_STATUS_LAYER.get(layer, layer)), **meta}))


# export layer -> provenance layer it is read from
_STATUS_LAYER = {"orientation": "page", "doctype": "page", "line_grounding": "ocr",
                 "kie_spotting": "kie", "qa": "understanding", "caption": "understanding",
                 "guide": "understanding"}


# --------------------------------------------------------------------------- per-layer exporters
def _orientation(e: _Emitter):
    e.add("orientation", "orient",
          "By how many degrees clockwise is this document rotated from upright? "
          "Answer with one of 0, 90, 180, 270.",
          [str(e.ann.page.orientation)], task="classification", metric="exact")


def _doctype(e: _Emitter):
    if e.ann.page.doc_type:
        e.add("doctype", "doctype", "What type of document is this? Answer with a short name.",
              [e.ann.page.doc_type], task="classification", metric="anls")


_CLASS_PHRASE = {"text": "text block", "kv_block": "key-value block", "list": "list block",
                 "ui_element": "UI element"}


def _layout(e: _Emitter, max_classes: int = 8):
    W, H = e.ann.page.width, e.ann.page.height
    by_cls: dict[str, list[str]] = {}
    for r in e.ann.layout:
        by_cls.setdefault(r.cls, []).append(_gbox(r.bbox, W, H))
    for k, (cls, golds) in enumerate(list(by_cls.items())[:max_classes]):
        name = _CLASS_PHRASE.get(cls, cls.replace("_", " "))
        e.add("layout", f"layout{k}", f"Where is the {name} in the document? "
                                      "Return the bounding box as x1,y1,x2,y2.",
              golds, task="localization", metric="grounding", label=cls)
    ordered = sorted((r for r in e.ann.layout if r.order is not None), key=lambda r: r.order)
    if len(ordered) >= 2:
        e.add("layout", "readorder", "List the layout blocks of the document in reading order, "
                                     "one class name per line.",
              ["\n".join(r.cls for r in ordered)], task="localization", metric="ned")


def _ocr(e: _Emitter):
    text = e.ann.full_text()
    if text.strip():
        e.add("ocr", "ocr", "Transcribe all the text in the image in reading order. "
                            "Answer with the text only.", [text], task="recognition", metric="ned")


def _line_grounding(e: _Emitter, max_lines: int = 6):
    """Text <-> box in both directions (read-this-box, where-is-this-text): the spotting signal."""
    W, H = e.ann.page.width, e.ann.page.height
    text_count: dict[str, int] = {}
    for ln in e.ann.ocr:
        text_count[ln.text] = text_count.get(ln.text, 0) + 1
    # prefer longer, unique lines: they are unambiguous grounding targets
    lines = sorted((ln for ln in e.ann.ocr if ln.text.strip() and text_count[ln.text] == 1
                    and ln.legibility == "clear"), key=lambda ln: -len(ln.text))[:max_lines]
    for ln in lines:
        e.add("line_grounding", f"where_{ln.id}",
              f'Where is the text "{ln.text}" in the document? Return the bounding box as x1,y1,x2,y2.',
              [_gbox(ln.bbox, W, H)], task="localization", metric="grounding", ref=ln.id)
        b = ln.bbox
        e.add("line_grounding", f"read_{ln.id}",
              f"Read the text inside the box {b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f} "
              f"(image is {W}x{H} pixels). Answer with the text only.",
              [ln.text], task="recognition", metric="ned", ref=ln.id)


def _table(e: _Emitter):
    for t in e.ann.tables:
        e.add("table", f"table_{t.id}", "Convert the table in the image to HTML.",
              [t.to_html()], task="table", metric="teds", ref=t.id)


def _kie_json(ann: DocAnnotation) -> str:
    return json.dumps({f.key: (f.target() if f.present else None) for f in ann.kie},
                      ensure_ascii=False)


def _kie(e: _Emitter):
    if not e.ann.kie:
        return
    keys = ", ".join(f.key for f in e.ann.kie)
    e.add("kie", "kie", f"Extract these fields from the document as JSON (use null when a field is "
                        f"not present): {keys}.", [_kie_json(e.ann)], task="kie", metric="kie_f1")
    for f in e.ann.kie:
        gold = [f.target(), f.value] if f.present else [_ABSENT, "not present"]
        e.add("kie", f"kie_{f.id}", f"What is the value of '{f.key}' in this document? "
                                    "Answer with the value only, or null if it is not present.",
              list(dict.fromkeys(gold)), task="kie", metric="anls", ref=f.id, key=f.key)


def _kie_spotting(e: _Emitter):
    """Value + where it is: the A1 'spot then answer' target, on real documents."""
    W, H = e.ann.page.width, e.ann.page.height
    for f in e.ann.kie:
        if not f.present:
            continue
        box = e.ann.evidence_bbox(f.value_lines)
        if box is None:
            continue
        e.add("kie_spotting", f"spot_{f.id}",
              f"What is the value of '{f.key}' and where is it? Answer as: value | x1,y1,x2,y2",
              [f"{f.target()} | {box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}"],
              task="localization", metric="anls", ref=f.id, key=f.key, bbox=box, size=[W, H])


def _render_chain(ann: DocAnnotation, q, *, guided: bool) -> str:
    """Steps -> text. Evidence boxes are written inline so the chain is grounded, not free-form."""
    idx = ann.index()
    lines = []
    if guided:
        scoped = {i for s in q.steps for i in s.evidence} | set(q.evidence)
        notes = [g.text for g in ann.understanding.guide
                 if not g.scope or scoped & set(g.scope)]
        if notes:
            lines.append("Context: " + " ".join(notes))
    for k, s in enumerate(q.steps, 1):
        where = []
        for i in s.evidence:
            box = ann.evidence_bbox([i])
            if box is not None and i in idx:
                where.append(f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
        tail = (" " + " ".join(where)) if where else ""
        res = f" -> {s.result}" if s.result not in (None, "") else ""
        lines.append(f"{k}. {s.text}{tail}{res}")
    lines.append(f"Answer: {q.answers[0]}")
    return "\n".join(lines)


def _qa(e: _Emitter, style: str):
    for q in e.ann.understanding.qa:
        if not q.answers:
            continue
        if style == "answer" or not q.steps:
            question, golds = q.question + " Answer concisely.", q.answers
        else:
            question = q.question + " Think step by step, then give the final answer after 'Answer:'."
            golds = [_render_chain(e.ann, q, guided=(style == "guided_chain"))]
        e.add("qa", f"qa_{q.id}", question, golds, task="reasoning" if q.steps else "vqa",
              metric=q.metric if style == "answer" else "final_answer", ref=q.id, qa_type=q.qa_type,
              difficulty=q.difficulty, style=style, final_answers=q.answers)


def _caption(e: _Emitter):
    if e.ann.understanding.caption:
        e.add("caption", "caption", "Describe this document: what it is and what it is for.",
              [e.ann.understanding.caption], task="caption", metric="ned")


def _guide(e: _Emitter):
    notes = e.ann.understanding.guide
    if notes:
        e.add("guide", "guide", "Explain how to read this document correctly: its structure, "
                                "conventions and pitfalls.",
              ["\n".join(f"- {g.text}" for g in notes)], task="caption", metric="ned")


_EXPORTERS = {"orientation": _orientation, "doctype": _doctype, "layout": _layout, "ocr": _ocr,
              "line_grounding": _line_grounding, "table": _table, "kie": _kie,
              "kie_spotting": _kie_spotting, "caption": _caption, "guide": _guide}


def export_samples(ann: DocAnnotation, layers=EXPORT_LAYERS, *, reasoning: str = "answer",
                   base_dir: str | Path | None = None, include_pseudo: bool = True) -> list[Sample]:
    """Emit training Samples for the chosen export ``layers`` of one record.

    ``reasoning`` controls the QA target (``answer`` | ``chain`` | ``guided_chain``).
    ``include_pseudo=False`` drops layers whose provenance is model-generated."""
    if reasoning not in REASONING_STYLES:
        raise ValueError(f"reasoning must be one of {REASONING_STYLES}")
    unknown = set(layers) - set(EXPORT_LAYERS)
    if unknown:
        raise ValueError(f"unknown export layers {sorted(unknown)}; choose from {EXPORT_LAYERS}")
    e = _Emitter(ann, _resolve_image(ann, base_dir))
    for layer in EXPORT_LAYERS:                      # fixed order -> deterministic output
        if layer not in layers:
            continue
        src = _STATUS_LAYER.get(layer, layer)
        if not ann.has_layer(src):
            continue
        if not include_pseudo and ann.layer_status(src) == "pseudo":
            continue
        if layer == "qa":
            _qa(e, reasoning)
        else:
            _EXPORTERS[layer](e)
    return e.out


def write_jsonl(samples: list[Sample], path: str | Path) -> int:
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return len(samples)


def to_unified(ann: DocAnnotation, base_dir: str | Path | None = None):
    """Bridge into the UDD record (:class:`~docvlm_eval.unified.UnifiedSample`) so hand-annotated
    documents can be merged with the public corpus. Lossy by design: UDD has no reasoning steps or
    guide; those stay in the DAR file. QAs go in ``qas`` (grouped form)."""
    from ..unified.core import QA, Box, Field, Task, UnifiedSample
    from ..unified.core import Region as URegion

    def box(b):
        return Box(*b, normalized=False) if b else None

    fields = [Field(f.key, f.target(), box(ann.evidence_bbox(f.value_lines)))
              for f in ann.kie if f.present]
    regions = [URegion(r.cls, box(r.bbox)) for r in ann.layout] + \
              [URegion("text_line", box(ln.bbox), ln.text) for ln in ann.ocr]
    qas = [QA(q.question, list(q.answers)) for q in ann.understanding.qa if q.answers]
    task = Task.KIE if fields else Task.REASONING if qas else Task.RECOGNITION
    return UnifiedSample(
        sample_id=ann.doc_id, source="dar", task=task, qas=qas, fields=fields, regions=regions,
        full_text=ann.full_text() or None,
        table_html=ann.tables[0].to_html() if ann.tables else None,
        language=(ann.page.languages or [None])[0], metric="anls",
        image_path=_resolve_image(ann, base_dir), split=ann.split,
        meta={"doc_type": ann.page.doc_type, "orientation": ann.page.orientation,
              "acquisition": ann.page.acquisition, "caption": ann.understanding.caption})
