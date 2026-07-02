"""Correctness metrics for text answers.

These follow the conventions used by the original benchmark authors so that numbers are
comparable to published results:

* **ANLS** (Average Normalized Levenshtein Similarity) - the official DocVQA / InfoVQA
  metric. Robust to minor OCR/formatting differences; scores 0 below a 0.5 similarity
  threshold. Ref: Biten et al., "Scene Text VQA" / Mathew et al., "DocVQA" (ICDAR'19/WACV'21).
* **Relaxed accuracy** - the official ChartQA metric: a numeric prediction counts as
  correct if within 5% of the gold value; otherwise exact (normalised) match. Ref:
  Masry et al., "ChartQA" (ACL'22 Findings).
* **Exact match** - normalised string equality (used for the robustness probe's
  short-answer items).
* **OCRBench** - OCRBench scores each item 0/1 by checking whether the gold string is
  contained in the (normalised) prediction, summed to a score out of 1000 over 1000
  items. Ref: Liu et al., "OCRBench" (2023).
"""

from __future__ import annotations

import re


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/articles and collapse whitespace.

    Mirrors the VQA-style normalisation so that "the Total: $1,200." and
    "total $1200" compare equal where appropriate.
    """
    s = s.strip().lower()
    s = s.replace("\n", " ")
    # remove thousands separators inside numbers (1,200 -> 1200)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w\s.%/-]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # strip surrounding punctuation (e.g. a trailing period from "Top-right.") while keeping
    # internal decimals intact ("0.28" stays "0.28")
    s = s.strip(" .,;:!?")
    return s


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance (iterative, O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _nls(pred: str, gold: str) -> float:
    """Normalized Levenshtein similarity for a single (pred, gold) pair."""
    p, g = pred.strip().lower(), gold.strip().lower()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    dist = levenshtein(p, g)
    return 1.0 - dist / max(len(p), len(g))


def anls(pred: str, golds: list[str], threshold: float = 0.5) -> float:
    """ANLS for one question against its set of gold answers.

    Takes the best-matching gold; similarities below ``threshold`` score 0 (the standard
    DocVQA behaviour that prevents partial credit for near-misses).
    """
    if not golds:
        return 0.0
    best = max(_nls(pred, g) for g in golds)
    return best if best >= threshold else 0.0


def _to_float(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def relaxed_accuracy(pred: str, golds: list[str], tol: float = 0.05) -> float:
    """ChartQA relaxed accuracy: numeric within ``tol`` relative error, else exact match."""
    np_ = normalize_text(pred)
    for g in golds:
        gf, pf = _to_float(g), _to_float(pred)
        if gf is not None and pf is not None:
            denom = abs(gf) if gf != 0 else 1.0
            if abs(pf - gf) / denom <= tol:
                return 1.0
        if np_ == normalize_text(g):
            return 1.0
    return 0.0


def exact_match(pred: str, golds: list[str]) -> float:
    np_ = normalize_text(pred)
    return 1.0 if any(np_ == normalize_text(g) for g in golds) else 0.0


def ocrbench_score(pred: str, golds: list[str]) -> float:
    """OCRBench item score: 1 if any (normalised) gold is a substring of the prediction."""
    np_ = normalize_text(pred)
    return 1.0 if any(normalize_text(g) in np_ for g in golds if g) else 0.0


def cer(pred: str, gold: str) -> float:
    """Character Error Rate = edit_distance / len(gold). Lower is better; can exceed 1."""
    g = gold.strip()
    if not g:
        return 0.0 if not pred.strip() else 1.0
    return levenshtein(pred.strip(), g) / len(g)


def wer(pred: str, gold: str) -> float:
    """Word Error Rate via word-level edit distance / #gold words."""
    gw, pw = gold.split(), pred.split()
    if not gw:
        return 0.0 if not pw else 1.0
    # word-level Levenshtein
    prev = list(range(len(gw) + 1))
    for i, a in enumerate(pw, 1):
        cur = [i] + [0] * len(gw)
        for j, b in enumerate(gw, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1] / len(gw)


def ned_similarity(pred: str, golds: list[str]) -> float:
    """1 - normalized edit distance against the best gold (higher better). Unlike ANLS this has
    no 0.5 cliff, so it rewards partially-correct long transcriptions."""
    if not golds:
        return 0.0
    return max(_nls(pred, g) for g in golds)


def _grounding(pred: str, golds: list[str]) -> float:
    from .grounding import grounding_score  # local import to avoid a cycle

    return grounding_score(pred, golds)


def _teds(pred: str, golds: list[str]) -> float:
    from .tables import teds_score  # local import to avoid a cycle

    return teds_score(pred, golds)


def _bank(name: str):
    """Late-bound dispatch into metrics.bank (module imports text, so avoid a cycle)."""
    def scorer(pred: str, golds: list[str]) -> float:
        from . import bank
        return bank.METRIC_BANK[name](pred, golds)
    return scorer


_SCORERS = {
    "anls": anls,
    "relaxed_acc": relaxed_accuracy,
    "exact": exact_match,
    "ocrbench": ocrbench_score,
    "grounding": _grounding,
    "ned": ned_similarity,
    "teds": _teds,
    # metric-bank additions (metrics/bank.py): usable as a Sample.metric directly
    "token_f1": _bank("token_f1"),
    "drop_em": _bank("drop_em"),
    "cer_sim": _bank("cer_sim"),
    "semantic_match": _bank("semantic_match"),
}


def score_sample(metric: str, pred: str, golds: list[str]) -> float:
    """Dispatch to the per-sample scorer named by ``metric``."""
    try:
        return _SCORERS[metric](pred, golds)
    except KeyError as exc:  # pragma: no cover - guards typos in loaders
        raise ValueError(f"Unknown metric '{metric}'. Known: {list(_SCORERS)}") from exc
