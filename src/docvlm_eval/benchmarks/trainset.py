"""Adapt heterogeneous public benchmark records into **our** training DTO.

Every benchmark in ``configs/benchmark_catalog.yaml`` ships its own raw schema (DocVQA has
``question``/``answers``; IAM has a ``text`` transcription; CORD packs key-values in a JSON
``ground_truth`` string; ChartQA uses ``query``/``label``; ...). To fine-tune on a *uniform* signal
we normalise each record into the canonical :class:`~docvlm_eval.schema.Sample` shape — the same
``{sample_id, image_path, question, answers, answer_type, metric, meta}`` JSONL that
``docvlm_eval.finetune.lora_vlm`` and ``scripts/run_ablation.py`` already train and eval on.

This module is the **pure** half: ``extract_qa(key, ex, entry)`` maps one raw example dict to a list
of QA dicts ``{question, answers, answer_type, metric}`` — no network, no image I/O — so the
per-benchmark mapping decisions are unit-testable offline (``tests/test_benchmark_trainset.py``).
The streaming / image-caching / HF-packaging half lives in ``scripts/build_benchmark_trainset.py``.

Add a benchmark by registering one function in ``_ADAPTERS``; anything unregistered falls through to
``_auto`` which probes the common field names.
"""

from __future__ import annotations

import json
from typing import Any, Callable

# Catalog metric strings -> the scorer vocabulary our eval understands. Anything unknown -> "anls".
_METRIC_ALIASES = {
    "anls": "anls", "exact": "exact", "relaxed_acc": "relaxed_acc", "ocrbench": "ocrbench",
    "ned": "ned", "cer": "ned", "wer": "ned", "teds": "anls", "f1": "anls",
}


def norm_metric(raw: str | None, default: str = "anls") -> str:
    """Map a free-form catalog metric to a supported scorer key (best-effort, lowercased prefix)."""
    if not raw:
        return default
    low = str(raw).strip().lower()
    for token, norm in _METRIC_ALIASES.items():
        if low.startswith(token):
            return norm
    return default


# --------------------------------------------------------------------------- small helpers
def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _as_answer_list(x: Any) -> list[str]:
    """Coerce an answer field (str | number | list | dict) into a non-empty list[str]."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = [_s(v) for v in x if _s(v)]
        return out
    if isinstance(x, dict):                       # e.g. {"text": [...]} style answer dicts
        for k in ("text", "answer", "answers", "value"):
            if k in x:
                return _as_answer_list(x[k])
        return []
    s = _s(x)
    return [s] if s else []


def _first(ex: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in ex and ex[k] not in (None, "", [], {}):
            return ex[k]
    return None


def _from_conversations(ex: dict) -> list[str]:
    """Pull the assistant/gpt turn(s) out of a LLaVA-style ``conversations`` list."""
    conv = ex.get("conversations")
    if not isinstance(conv, list):
        return []
    out = []
    for t in conv:
        if isinstance(t, dict) and str(t.get("from", "")).lower() in ("gpt", "assistant"):
            v = _s(t.get("value"))
            if v:
                out.append(v)
    return out


_TRANSCRIBE = "Transcribe all the text in the image. Answer with the text only."
_LATEX = "Convert the formula in the image to LaTeX. Answer with the LaTeX only."
_TABLE = "Convert the table in the image to HTML."


def _qa(question: str, answers: list[str], answer_type: str, metric: str) -> list[dict]:
    answers = [a for a in answers if _s(a)]
    if not _s(question) or not answers:
        return []
    return [{"question": question, "answers": answers, "answer_type": answer_type, "metric": metric}]


# --------------------------------------------------------------------------- per-benchmark adapters
def _vqa(ex: dict, e: dict, *, atype: str | None = None, metric: str | None = None) -> list[dict]:
    """Generic visual-QA: a question field + an answer/answers field (the most common shape)."""
    q = _s(_first(ex, ("question", "query", "instruction", "prompt", "text_input")))
    ans = _as_answer_list(_first(ex, ("answers", "answer", "label", "labels", "gt_answer", "value")))
    if not ans:
        ans = _from_conversations(ex)
    return _qa(q, ans, atype or e.get("key", "vqa"), norm_metric(metric or e.get("metric")))


def _transcribe(ex: dict, e: dict, *, keys=("text", "label", "transcription", "gt", "ground_truth"),
                question=_TRANSCRIBE, atype="recognition", metric="ned") -> list[dict]:
    ans = _as_answer_list(_first(ex, keys))
    return _qa(question, ans, atype, norm_metric(metric))


def _mcq(ex: dict, e: dict) -> list[dict]:
    """Multiple-choice (AI2D-style): question + options[], answer is an index or the option text."""
    q = _s(_first(ex, ("question", "query")))
    opts = _first(ex, ("options", "choices"))
    raw_ans = _first(ex, ("answer", "label", "correct"))
    if isinstance(opts, list) and opts:
        # answer may be an int index, a digit string, or the literal option text
        ans_text = None
        if isinstance(raw_ans, int) and 0 <= raw_ans < len(opts):
            ans_text = opts[raw_ans]
        elif isinstance(raw_ans, str) and raw_ans.isdigit() and int(raw_ans) < len(opts):
            ans_text = opts[int(raw_ans)]
        else:
            ans_text = _s(raw_ans)
        q = q + "\nOptions: " + "; ".join(f"{i+1}) {_s(o)}" for i, o in enumerate(opts))
        return _qa(q, [_s(ans_text)], "diagram-mcq", "exact")
    return _vqa(ex, e, atype="diagram-mcq", metric="exact")


def _cord(ex: dict, e: dict) -> list[dict]:
    """CORD receipts: a JSON ``ground_truth`` string with ``gt_parse`` of fine-grained key-values."""
    gt = ex.get("ground_truth")
    parse = None
    if isinstance(gt, str):
        try:
            parse = json.loads(gt).get("gt_parse")
        except Exception:
            parse = None
    elif isinstance(gt, dict):
        parse = gt.get("gt_parse", gt)
    if not isinstance(parse, dict):
        return _vqa(ex, e, atype="kie", metric="anls")
    flat = json.dumps(parse, ensure_ascii=False, sort_keys=True)
    return _qa("Extract the receipt's key fields as JSON.", [flat], "kie", "anls")


def _table(ex: dict, e: dict) -> list[dict]:
    ans = _as_answer_list(_first(ex, ("html_table", "html", "table", "text", "label")))
    return _qa(_TABLE, ans, "table", "anls")


def _ai2d(ex: dict, e: dict) -> list[dict]:
    return _mcq(ex, e)


def _ocrvqa(ex: dict, e: dict) -> list[dict]:
    """OCR-VQA packs several QA per book-cover image as parallel ``questions`` / ``answers`` lists."""
    qs = ex.get("questions") or ex.get("question")
    ans = ex.get("answers") or ex.get("answer")
    qs = qs if isinstance(qs, list) else [qs]
    ans = ans if isinstance(ans, list) else [ans]
    out: list[dict] = []
    for q, a in zip(qs, ans):
        out += _qa(_s(q), _as_answer_list(a), "doc-vqa", "exact")
    return out


def _charxiv(ex: dict, e: dict) -> list[dict]:
    """CharXiv: ``reasoning_q``/``reasoning_a`` are free text (the descriptive_q* are integer
    template ids with no in-row decoder, so we only take the reasoning pair)."""
    return _qa(_s(ex.get("reasoning_q")), _as_answer_list(ex.get("reasoning_a")),
               "sci-figure", "exact")


# key -> adapter. Unregistered benchmarks fall through to ``_auto``.
_ADAPTERS: dict[str, Callable[[dict, dict], list[dict]]] = {
    "docvqa": lambda ex, e: _vqa(ex, e, atype="doc-vqa", metric="anls"),
    "infovqa": lambda ex, e: _vqa(ex, e, atype="infographic", metric="anls"),
    "textvqa": lambda ex, e: _vqa(ex, e, atype="scene-text-vqa", metric="exact"),
    "stvqa": lambda ex, e: _vqa(ex, e, atype="scene-text-vqa", metric="anls"),
    "ocrvqa": _ocrvqa,
    "ai2d": _ai2d,
    "chartqa": lambda ex, e: _vqa(ex, e, atype="chart", metric="relaxed_acc"),
    "mathvista": lambda ex, e: _vqa(ex, e, atype="figure-math", metric="exact"),
    "plotqa": lambda ex, e: _vqa(ex, e, atype="chart", metric="relaxed_acc"),
    "dvqa": lambda ex, e: _vqa(ex, e, atype="chart", metric="exact"),
    "ocrbench": lambda ex, e: _vqa(ex, e, atype="ocr", metric="ocrbench"),
    "ocrbench_v2": lambda ex, e: _vqa(ex, e, atype="ocr", metric="exact"),
    "charxiv": _charxiv,
    "pope": lambda ex, e: _vqa(ex, e, atype="hallucination", metric="exact"),
    "hallusionbench": lambda ex, e: _vqa(ex, e, atype="hallucination", metric="exact"),
    "iam": lambda ex, e: _transcribe(ex, e, atype="handwriting"),
    "sroie": lambda ex, e: _transcribe(ex, e, keys=("label", "text", "objects"), atype="receipt-line"),
    "funsd": lambda ex, e: _transcribe(ex, e, keys=("words", "tokens", "text"),
                                       question=_TRANSCRIBE, atype="form", metric="ned"),
    "im2latex": lambda ex, e: _transcribe(ex, e, keys=("latex_formula", "latex", "formula", "text"),
                                          question=_LATEX, atype="formula", metric="ned"),
    "latexocr": lambda ex, e: _transcribe(ex, e, keys=("text", "latex", "formula"),
                                          question=_LATEX, atype="formula", metric="ned"),
    "cord": _cord,
    "pubtabnet": _table,
    "fintabnet": _table,
}


def _auto(ex: dict, e: dict) -> list[dict]:
    """Fallback for unregistered benchmarks: try VQA, then a transcription target, then conversations."""
    out = _vqa(ex, e)
    if out:
        return out
    out = _transcribe(ex, e, metric=str(e.get("metric") or "ned"))
    if out:
        return out
    conv = _from_conversations(ex)
    return _qa("What does the image show? Answer concisely.", conv, e.get("key", "auto"),
               norm_metric(e.get("metric")))


def extract_qa(key: str, ex: dict, entry: dict | None = None) -> list[dict]:
    """Map one raw benchmark example to a list of QA dicts in our DTO.

    Returns ``[]`` when no trainable (question, answer) pair can be derived (e.g. detection-only
    sets); callers skip those rows.
    """
    entry = entry or {"key": key}
    adapter = _ADAPTERS.get(key, _auto)
    try:
        return adapter(ex, entry)
    except Exception:
        # never let one malformed record abort a whole-benchmark stream
        return []
