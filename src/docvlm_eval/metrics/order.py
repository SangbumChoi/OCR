"""Reading-order metrics: score ORDER separately from CONTENT.

A plain transcript metric (NED/CER) conflates two failures: a model that reads every element
perfectly but in the wrong order (tategaki read row-wise, columns read across) looks identical to
one that misreads half the words. These two metrics split them:

* ``content_bag``  — order-INSENSITIVE: what fraction of the gold elements appear in the
  prediction at all (normalized substring / token-overlap match). "Did it read everything?"
* ``order_tau``    — content-INSENSITIVE: over the elements that were found, the normalized
  Kendall rank correlation between their predicted positions and the gold reading order,
  mapped to [0,1] (1 = perfect order, 0.5 = random, 0 = exactly reversed).
  "Did it read them in the right sequence?"

The GAP between the two isolates reading-order failure: content 0.95 / order 0.60 is an
order problem; content 0.60 / order 0.95 is a recognition problem.

Gold format (``Sample.answers[0]``): the ordered elements joined by ``\\n`` — exactly what the
OmniDocBench rows in UDD carry (``regions`` are stored in reading order) and what the synthetic
reading-order probe emits (``scripts/make_reading_order_probe.py``).
"""

from __future__ import annotations

from .text import normalize_text


def _match_positions(pred: str, elements: list[str]) -> list[int | None]:
    """Position of each gold element in the (normalized) prediction — None when not found.

    Elements are matched left-to-right as substrings after VQA normalization, each search starting
    at the string head so positions reflect the PREDICTION's ordering, not the gold's. Elements
    shorter than 3 normalized chars fall back to token matching to avoid spurious hits."""
    np_ = normalize_text(pred)
    out: list[int | None] = []
    for el in elements:
        ne = normalize_text(el)
        if not ne:
            out.append(None)
            continue
        pos = np_.find(ne)
        if pos < 0 and len(ne) >= 6:                     # tolerate small OCR noise on long elements
            head, tail = ne[: len(ne) // 2], ne[len(ne) // 2:]
            p1, p2 = np_.find(head), np_.find(tail)
            pos = p1 if p1 >= 0 else p2
        out.append(pos if pos >= 0 else None)
    return out


def content_bag(pred: str, golds: list[str]) -> float:
    """Fraction of gold elements present in the prediction, order ignored. Best over golds."""
    best = 0.0
    for g in golds:
        elements = [e for e in g.split("\n") if e.strip()]
        if not elements:
            continue
        found = sum(1 for p in _match_positions(pred, elements) if p is not None)
        best = max(best, found / len(elements))
    return best


def order_tau(pred: str, golds: list[str]) -> float:
    """Normalized Kendall tau of the found elements' predicted positions vs the gold order,
    mapped to [0,1]. Needs >=2 found elements (else 0 — no order evidence). Best over golds."""
    best = 0.0
    for g in golds:
        elements = [e for e in g.split("\n") if e.strip()]
        pos = [p for p in _match_positions(pred, elements) if p is not None]
        n = len(pos)
        if n < 2:
            continue
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pos[i] < pos[j]:
                    concordant += 1
                elif pos[i] > pos[j]:
                    discordant += 1
        pairs = n * (n - 1) / 2
        tau = (concordant - discordant) / pairs          # [-1, 1]
        best = max(best, (tau + 1) / 2)
    return best
