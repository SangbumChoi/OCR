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


_SCORERS = {
    "anls": anls,
    "relaxed_acc": relaxed_accuracy,
    "exact": exact_match,
    "ocrbench": ocrbench_score,
}


def score_sample(metric: str, pred: str, golds: list[str]) -> float:
    """Dispatch to the per-sample scorer named by ``metric``."""
    try:
        return _SCORERS[metric](pred, golds)
    except KeyError as exc:  # pragma: no cover - guards typos in loaders
        raise ValueError(f"Unknown metric '{metric}'. Known: {list(_SCORERS)}") from exc
