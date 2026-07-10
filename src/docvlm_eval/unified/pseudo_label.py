"""Pseudo-labeling pipeline: fill or repair EMPTY UDD columns with model-generated labels.

Much of the corpus has structurally empty annotation that a strong open-source model could fill —
most importantly ``full_text``: only recognition-task rows carry a transcript, yet every document
image HAS one, and a SOTA open-source OCR model (the repo already wraps ``got-ocr2`` and
``paddleocr-vl`` in ``docvlm_eval.models``) can pseudo-label the rest. Same pattern applies to
region texts (layout boxes without content) and table HTML.

Design principles:

* **Pluggable fillers** (:class:`Filler`): each declares the target column, a ``needs(row)``
  predicate (is this row missing the label?), and a ``label(row, image)`` function that produces
  the value. Registered in :data:`FILLERS`.
* **Plan first, infer later** — ``plan(ds)`` reports exactly what each filler WOULD fill (counts
  per source/task) with zero model loading; ``apply(ds, filler, labeler=...)`` runs the actual
  fill. The GPU labelers are intentionally NOT implemented here — filling is future work; the
  pipeline, provenance and tests are not.
* **Provenance is mandatory**: every filled value is recorded in the row's ``pseudo_json`` column
  (``{column: labeler_name}``), so pseudo-labels can always be distinguished from source GT,
  filtered out, or regenerated with a better model. Gold values are NEVER overwritten — fillers
  only touch rows where ``needs(row)`` is true.

    from docvlm_eval.unified.pseudo_label import plan, apply, FILLERS
    print(plan(ds))                                          # counts only, no models
    ds = apply(ds, "full_text", labeler=my_ocr_fn, name="got-ocr2")   # GPU, later
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable


# --------------------------------------------------------------- standardization (the contract)
# A model's raw output is NEVER written to the corpus as-is: every filler owns a normalizer that
# (a) standardizes the value into the column's format and (b) REJECTS degenerate outputs by
# returning None (the row is then simply left unfilled — no provenance, no bad label).
_WRAPPERS = re.compile(
    r"^(the (text|image|document)( in the image)? (reads|says|shows|contains|is)[:\s]*|"
    r"here is the (text|transcription)[:\s]*|transcription[:\s]*)", re.IGNORECASE)


def normalize_text(raw: str, *, max_len: int = 8000) -> str | None:
    """Standardize a model transcript: NFC unicode, chat-wrapper phrases stripped, whitespace
    runs collapsed (newlines kept — reading order is meaningful), surrounding quotes dropped.
    Returns None for degenerate outputs (empty, refusals, over-long ramblings)."""
    t = unicodedata.normalize("NFC", (raw or "").strip())
    t = _WRAPPERS.sub("", t).strip().strip('"\u201c\u201d').strip()
    t = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines())
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    low = t.lower()
    if not t or len(t) > max_len:
        return None
    if low.startswith(("i cannot", "i can't", "i'm sorry", "sorry,", "unable to")):
        return None                                   # refusal, not a transcript
    return t


def normalize_table_html(raw: str) -> str | None:
    """Standardize table output: keep exactly the <table>...</table> block, nothing else."""
    m = re.search(r"<table\b.*?</table>", raw or "", re.IGNORECASE | re.DOTALL)
    return m.group(0) if m else None


@dataclass
class Filler:
    """One pseudo-labelable column: what counts as empty + how a labeler would fill it."""
    column: str
    description: str
    needs: Callable[[dict], bool]
    suggested_models: tuple[str, ...] = ()
    normalize: Callable[[Any], Any] | None = None
    prompt: str = ""                                  # the standardized instruction for VLM labelers


def _no_full_text(row: dict) -> bool:
    return not (row.get("full_text") or "").strip()


def _has_textless_regions(row: dict) -> bool:
    els = json.loads(row.get("elements_json") or "[]")
    return any(e.get("kind") == "region" and e.get("bbox") and not (e.get("value") or "").strip()
               for e in els)


def _no_table_html(row: dict) -> bool:
    return row.get("task") == "table" and not (row.get("table_html") or "").strip()


FILLERS: dict[str, Filler] = {
    "full_text": Filler(
        column="full_text",
        description="full-page transcript from a SOTA open-source OCR model — every document image "
                    "has one, but only recognition-task rows ship it",
        needs=_no_full_text,
        suggested_models=("got-ocr2", "paddleocr-vl"),
        normalize=normalize_text,
        prompt="Read ALL the text in this image, top to bottom. Output ONLY the text exactly as "
               "written, preserving line breaks. No commentary."),
    "region_text": Filler(
        column="elements_json",
        description="text content of layout regions that have a box but no value (DocLayNet/"
                    "PubLayNet boxes are class-only) — crop the box, OCR the crop",
        needs=_has_textless_regions,
        suggested_models=("got-ocr2", "paddleocr-vl"),
        normalize=normalize_text,
        prompt="Read the text inside this cropped document region. Output ONLY the text, "
               "no commentary."),
    "table_html": Filler(
        column="table_html",
        description="table-structure HTML for table-task rows missing it",
        needs=_no_table_html,
        suggested_models=("got-ocr2",),
        normalize=normalize_table_html,
        prompt="Convert this table image to HTML. Output ONLY the <table>...</table> markup."),
}


def plan(ds) -> dict[str, Any]:
    """What WOULD be filled — counts per filler, split by source — without loading any model."""
    report: dict[str, Any] = {"total_rows": len(ds)}
    rows = [{k: ds[k][i] for k in ("full_text", "elements_json", "table_html", "task", "source")
             if k in ds.column_names} for i in range(len(ds))]
    for name, f in FILLERS.items():
        need = [r for r in rows if f.needs(r)]
        by_src: dict[str, int] = {}
        for r in need:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        report[name] = {"column": f.column, "rows_needing_fill": len(need),
                        "share": round(len(need) / max(1, len(ds)), 3),
                        "suggested_models": list(f.suggested_models),
                        "by_source_top": dict(sorted(by_src.items(), key=lambda t: -t[1])[:8])}
    return report


def apply(ds, filler_name: str, *, labeler: Callable[[dict], Any], name: str):
    """Fill one column with ``labeler(row) -> value`` on the rows that need it, recording
    provenance in ``pseudo_json`` (``{column: labeler_name}``). Gold values are never touched.

    ``labeler`` receives the full row (incl. the decoded ``image``) and returns the new column
    value, or None to skip the row. Plug a real OCR model here when running on a GPU — e.g. wrap
    ``docvlm_eval.models`` 'got-ocr2' generate() into a ``lambda row: ocr(row["image"])``."""
    f = FILLERS[filler_name]

    def _fill(row):
        out = {}
        pseudo = json.loads(row.get("pseudo_json") or "{}")
        if f.needs(row):
            val = labeler(row)
            if val is not None and f.normalize is not None:
                val = f.normalize(val)                # standardize; None = reject, leave unfilled
            if val is not None:
                out[f.column] = val
                pseudo[f.column] = name
        out["pseudo_json"] = json.dumps(pseudo, ensure_ascii=False)
        return out

    return ds.map(_fill, desc=f"pseudo-label: {filler_name} <- {name}",
                  load_from_cache_file=False)


def vlm_labeler(model_key: str, filler_name: str, *, device: str = "cpu",
                max_new_tokens: int = 256) -> Callable[[dict], str | None]:
    """Wrap any model from ``docvlm_eval.models`` into a ``labeler(row) -> raw text``.

    The filler's standardized ``prompt`` is used, so every model answers the SAME instruction —
    the output contract (and rejection rules) live in the filler's ``normalize``, keeping labels
    comparable across labeler models. Works with any registered VLM ('smolvlm-256m' runs on CPU;
    'got-ocr2' / 'paddleocr-vl' / bigger Qwen/LFM need a GPU)."""
    import tempfile
    from ..models import build_model
    from ..models.base import GenConfig

    adapter = build_model(model_key, device=device,
                          dtype="float32" if device == "cpu" else "bfloat16",
                          gen=GenConfig(max_new_tokens=max_new_tokens))
    adapter.load()
    prompt = FILLERS[filler_name].prompt

    def _label(row: dict) -> str | None:
        img = row.get("image")
        if img is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
            img.convert("RGB").save(tf.name)
            text, _conf = adapter.generate(tf.name, prompt)
        return text

    return _label
