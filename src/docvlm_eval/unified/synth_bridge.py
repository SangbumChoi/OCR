"""Bridge the SYNTHETIC generator's DTO (``docvlm_eval.synth.dto.DocSample``) into the UDD format.

The synthetic pipeline persists one ``gt.json`` (the :meth:`DocSample.to_dict` superset) + rendered
image per case. This module converts that record into a :class:`~docvlm_eval.unified.core.
UnifiedSample` carrying **every annotation the UDD schema can express**, so synthetic and public
data flow through the same trainset builders, validators and ablation arms:

* ``qa_detailed``           -> the native QA pairing (``qas`` when >1, flat otherwise)
* ``fields_detailed``       -> UDD ``fields`` (pixel boxes normalized by the image size; the A1 signal)
* ``qa.answer_bbox``        -> UDD ``regions`` (grounding targets for ``to_grounding_samples``)
* ``qa.rationale``          -> an EXTRA "… Explain your reasoning." QA whose target is the rationale
                               chain (the A2 signal — kept as data, since UDD has no rationale column)
* ``table_html`` / ``languages`` / doc metadata -> the matching UDD columns / ``meta``

Synthetic corpora built this way are **local-only by design** (they are regenerable and not public
data — do not push them to the UDD Hub repo); ``scripts/build_udd_synthetic.py`` builds + safety-
checks such a corpus next to the public one.
"""

from __future__ import annotations

from typing import Any

from .core import QA, Box, Field, Region, Task, UnifiedSample
from ..benchmarks.trainset import _s, norm_metric


def _norm_box(b: list | None, w: int, h: int) -> Box | None:
    """Synth boxes are PIXEL [x1,y1,x2,y2] in the rendered frame -> normalized UDD Box."""
    if not (b and len(b) >= 4 and w and h):
        return None
    return Box(b[0] / w, b[1] / h, b[2] / w, b[3] / h, True)


def docsample_to_unified(gt: Any, image_path: str, image_size: tuple[int, int],
                         sample_id: str = "") -> UnifiedSample:
    """Convert one synthetic record (a ``DocSample`` or its persisted ``gt.json`` dict) into a
    UDD-compatible :class:`UnifiedSample`. Raises ValueError when nothing trainable is present."""
    if hasattr(gt, "to_dict"):
        gt = gt.to_dict()
    w, h = image_size

    # ---- fields (KIE + A1 boxes): prefer the structured view, fall back to the legacy mirror
    fields: list[Field] = []
    detailed = gt.get("fields_detailed")
    if detailed:
        for f in detailed:
            fields.append(Field(key=_s(f.get("key")), value=_s(f.get("value")),
                                bbox=_norm_box(f.get("bbox"), w, h)))
    else:
        spotting = gt.get("spotting", {}) or {}
        for k, v in (gt.get("fields") or {}).items():
            fields.append(Field(key=_s(k), value=_s(v), bbox=_norm_box(spotting.get(k), w, h)))

    # ---- QAs: every question, plus a derived reasoning QA per rationale (A2 as data)
    qas: list[QA] = []
    regions: list[Region] = []
    metrics: list[str] = []
    for q in (gt.get("qa_detailed") or gt.get("qa") or []):
        question, answers = _s(q.get("question")), [_s(a) for a in (q.get("answers") or []) if _s(a)]
        if not (question and answers):
            continue
        qas.append(QA(question, answers))
        metrics.append(norm_metric(q.get("metric")))
        bb = _norm_box(q.get("answer_bbox"), w, h)
        if bb is not None:                       # grounding target -> region (A1)
            regions.append(Region(label=_s(q.get("key")) or question[:40], bbox=bb,
                                  text=answers[0]))
        rat = _s(q.get("rationale"))
        if rat:                                  # rationale -> explicit reasoning QA (A2)
            qas.append(QA(f"{question} Explain your reasoning.",
                          [f"{rat} So the answer is {answers[0]}."]))

    table_html = _s(gt.get("table_html")) or None
    if not (qas or fields or table_html):
        raise ValueError(f"synthetic record {gt.get('doc_id')!r} has no trainable annotation")

    task = (Task.TABLE if table_html and not qas else
            Task.KIE if fields and not qas else Task.VQA)
    langs = [
        language
        for language in (gt.get("languages") or [])
        if language
    ]
    single = qas[0] if len(qas) == 1 else None
    render = gt.get("render") or {}
    raw_page_count = render.get("rendered_page_count")
    if raw_page_count is None:
        raw_page_count = render.get("page_count")
    page_count = 1 if raw_page_count is None else int(raw_page_count)
    raw_document_count = render.get("document_count")
    document_count = (
        1 if raw_document_count is None else int(raw_document_count)
    )
    if page_count < 1 or document_count < 1:
        raise ValueError("synthetic composition counts must be positive")
    return UnifiedSample(
        sample_id=sample_id or f"synthetic_{_s(gt.get('doc_id')) or 'case'}_0",
        source="synthetic", task=task,
        instruction=single.question if single else "",
        answers=list(single.answers) if single else [],
        qas=qas if len(qas) > 1 else [],
        fields=fields, regions=regions, table_html=table_html,
        language=langs[0] if len(langs) == 1 else None,
        metric=metrics[0] if metrics else norm_metric(gt.get("anchor_metric")),
        image_path=image_path, hf_id=None, split="synthetic",
        meta={"doc_type": _s(gt.get("doc_type") or gt.get("type")),
              "stressors": list(gt.get("stressors") or []),
              "domain": _s(gt.get("domain")) or None,
              "acquisition": _s(gt.get("acquisition")) or None,
              "page_count": page_count,
              "document_count": document_count,
              "synthetic": True})
