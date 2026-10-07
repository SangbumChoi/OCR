"""Scorers for STRUCTURED answers emitted by the layered annotation exporters.

* ``kie_f1``       — field-level F1 between a predicted JSON object and the gold JSON object.
  A field counts as correct when its value matches under :func:`bank.semantic_match` (so "2,305.00"
  == "2305.00"); a gold ``null`` (field not on the document) is matched only by a missing key or
  ``null``/empty — so hallucinating a value for an absent field is penalised.
* ``final_answer`` — for chain-of-thought targets: take the text after the LAST ``Answer:`` in the
  prediction and in each gold chain, then score with ``semantic_match`` (relaxed-acc for numbers).
  The chain itself is not scored — it is the training signal, the final answer is the outcome.
"""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(r"answer\s*[:：]", re.IGNORECASE)


def _parse_json_obj(s: str) -> dict | None:
    s = (s or "").strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _is_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a"))


def _field_f1(pred: dict, gold: dict) -> float:
    from .bank import semantic_match
    keys = set(gold)
    tp = fp = fn = 0
    for k in keys:
        g, p = gold.get(k), pred.get(k)
        if _is_null(g):
            if not _is_null(p):
                fp += 1                     # hallucinated value for an absent field
            continue
        if _is_null(p):
            fn += 1
        elif semantic_match(str(p), [str(g)]) >= 1.0:
            tp += 1
        else:
            fp += 1
            fn += 1
    fp += sum(1 for k in set(pred) - keys if not _is_null(pred[k]))
    if tp == 0:
        return 1.0 if fp == fn == 0 else 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def kie_f1(pred: str, golds: list[str]) -> float:
    p = _parse_json_obj(pred)
    if p is None:
        return 0.0
    best = 0.0
    for g in golds:
        go = _parse_json_obj(g)
        if go is not None:
            best = max(best, _field_f1(p, go))
    return best


def extract_final_answer(text: str) -> str:
    """First line after the LAST 'Answer:' marker; the whole (stripped) text when there is none."""
    parts = _ANSWER_RE.split(text or "")
    if len(parts) == 1:
        return (text or "").strip()
    tail = parts[-1].strip()
    return tail.splitlines()[0].strip() if tail else ""


_NUMERIC_RE = re.compile(r"^[\s$€£¥₩+-]*\d[\d,]*(\.\d+)?\s*%?$")


def final_answer(pred: str, golds: list[str]) -> float:
    from .bank import semantic_match
    from .text import relaxed_accuracy
    p = extract_final_answer(pred)
    g = [extract_final_answer(x) for x in golds]
    g = [x for x in g if x]
    if not (p and g):
        return 0.0
    score = semantic_match(p, g)
    # relaxed (5%) numeric tolerance only for purely numeric golds — never for dates / ids, where
    # "2024-03-04" would otherwise pass as the number 2024
    numeric = [x for x in g if _NUMERIC_RE.match(x)]
    if numeric:
        score = max(score, relaxed_accuracy(p, numeric))
    return score
