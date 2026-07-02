"""The METRIC BANK: every text-answer scorer behind one (pred, golds) -> [0,1] signature.

Different benchmarks canonised different scoring rules (SQuAD token-F1, DROP/TAT-QA normalized EM,
OCR's CER, DocVQA's ANLS, ChartQA's relaxed accuracy, …), and each embodies a different *tolerance*:
what surface variation it forgives and what it punishes. This module collects them as a uniform
bank so the SAME predictions can be scored under EVERY metric and their tendencies compared
(``scripts/compare_metrics.py`` builds the perturbation-tolerance matrix and the per-prediction
correlation/disagreement report).

Added here (the classics the repo lacked):

* ``token_f1``  — SQuAD-style: VQA-normalise both sides, token-overlap F1, best over golds.
  Gives partial credit for span-vs-sentence answers ("5 days" vs "The default period is 5 days").
* ``drop_em``   — DROP / TAT-QA-style normalized exact match: number WORDS -> digits
  ("five" == "5"), thousands separators, currency/percent symbols and month-name dates
  canonicalised, then numeric equality if both sides are numbers, else string EM.
* ``cer_sim``   — 1 - CER clipped to [0,1] (best over golds): the recognition metric expressed as
  a similarity so it can sit in the same table as the others.
* ``semantic_match`` — the layered scorer: exact/canon EM -> drop_em -> token_f1 as partial credit.
  Deterministic and offline; the recommended default when one number must summarise "did it mean
  the same thing".

All are registered in ``text._SCORERS`` too, so a ``Sample.metric`` can name them directly.
"""

from __future__ import annotations

import re

from .text import (anls, cer, exact_match, ned_similarity, normalize_text, ocrbench_score,
                   relaxed_accuracy)

# ---------------------------------------------------------------------------- normalizers
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}
_MONTHS.update({m[:3]: v for m, v in _MONTHS.items()})


def _words_to_digits(s: str) -> str:
    """Replace standalone number words (incl. hyphenated tens-units like twenty-one) with digits."""
    def hyph(m: re.Match) -> str:
        tens, unit = m.group(1), m.group(2)
        return str(_NUM_WORDS[tens] + _NUM_WORDS[unit])
    s = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-"
               r"(one|two|three|four|five|six|seven|eight|nine)\b", hyph, s)
    return re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b",
                  lambda m: str(_NUM_WORDS[m.group(1)]), s)


def _norm_dates(s: str) -> str:
    """'january 5, 2020' / '5 jan 2020' -> '2020-01-05' (month-name forms only — unambiguous)."""
    def md_y(m: re.Match) -> str:
        return f"{m.group(3)}-{_MONTHS[m.group(1)[:3]]:02d}-{int(m.group(2)):02d}"
    def dm_y(m: re.Match) -> str:
        return f"{m.group(3)}-{_MONTHS[m.group(2)[:3]]:02d}-{int(m.group(1)):02d}"
    mon = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
    s = re.sub(mon + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", md_y, s)
    s = re.sub(r"(\d{1,2})(?:st|nd|rd|th)?\s+" + mon + r",?\s+(\d{4})", dm_y, s)
    return s


def drop_normalize(s: str) -> str:
    """DROP/TAT-QA-style canonical form: VQA normalize + number-words->digits + date + unit strip."""
    s = normalize_text(s)
    s = _norm_dates(s)
    s = _words_to_digits(s)
    s = re.sub(r"[%$€£¥₩]", " ", s)                      # units/currency: compare the quantity
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_number(s: str) -> float | None:
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s.strip())
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------- new scorers
def token_f1(pred: str, golds: list[str]) -> float:
    """SQuAD-style token-overlap F1 after VQA normalization; best over golds."""
    pt = normalize_text(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_text(g).split()
        if not pt and not gt:
            return 1.0
        if not pt or not gt:
            continue
        common: dict[str, int] = {}
        for t in pt:
            common[t] = common.get(t, 0) + 1
        overlap = sum(min(c, gt.count(t)) for t, c in common.items())
        if overlap == 0:
            continue
        prec, rec = overlap / len(pt), overlap / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def drop_em(pred: str, golds: list[str]) -> float:
    """DROP/TAT-QA-style EM: canonicalise (number words, dates, separators, units), then numeric
    equality when both sides are numbers, else string equality. Best over golds."""
    p = drop_normalize(pred)
    pn = _as_number(p)
    for g in golds:
        gq = drop_normalize(g)
        gn = _as_number(gq)
        if pn is not None and gn is not None:
            if abs(pn - gn) < 1e-9:
                return 1.0
        elif p == gq:
            return 1.0
    return 0.0


def cer_sim(pred: str, golds: list[str]) -> float:
    """Recognition CER as a similarity: max over golds of 1 - CER, clipped to [0,1]."""
    if not golds:
        return 0.0
    return max(0.0, max(1.0 - cer(pred, g) for g in golds))


def semantic_match(pred: str, golds: list[str]) -> float:
    """Layered semantic scorer (deterministic, offline):
    1. normalized exact match -> 1.0
    2. DROP-style canonical/numeric match ("five" == "5", "Jan 5, 2020" == "5 January 2020") -> 1.0
    3. otherwise token-F1 as partial credit (span-vs-sentence answers)."""
    if exact_match(pred, golds):
        return 1.0
    if drop_em(pred, golds):
        return 1.0
    return token_f1(pred, golds)


# ---------------------------------------------------------------------------- the bank
METRIC_BANK: dict = {
    # existing conventions (benchmark-official)
    "exact": exact_match,
    "anls": anls,
    "ned": ned_similarity,
    "relaxed_acc": relaxed_accuracy,
    "ocrbench": ocrbench_score,
    # classic families added for comparison
    "token_f1": token_f1,
    "drop_em": drop_em,
    "cer_sim": cer_sim,
    "semantic_match": semantic_match,
}


def score_all(pred: str, golds: list[str]) -> dict[str, float]:
    """Score one prediction under EVERY bank metric -> {metric: score}. The comparison primitive."""
    return {name: float(fn(pred, golds)) for name, fn in METRIC_BANK.items()}
