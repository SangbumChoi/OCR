"""Calibration metrics.

A deployable document reader should *know when it is unsure* - if it answers a field on
an invoice with low confidence, a downstream system can route to human review. We measure
this with **Expected Calibration Error (ECE)**: bin predictions by confidence, and compare
the average confidence in each bin to the empirical correctness in that bin.

ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

Ref: Guo et al., "On Calibration of Modern Neural Networks" (ICML'17); Naeini et al. (AAAI'15).

We treat a sample as "correct" using its task metric thresholded at ``correct_threshold``
(e.g. ANLS >= 0.5), which makes calibration comparable across DocVQA-style and exact-match
tasks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    avg_confidence: float
    accuracy: float


def reliability_table(
    confidences: list[float],
    correctness: list[float],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin (confidence, correctness) pairs into equal-width confidence bins."""
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            # last bin is closed on the right so confidence==1.0 lands somewhere
            if (lo < c <= hi) or (b == 0 and c == 0.0)
        ]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correctness[i] for i in idx) / len(idx)
        bins.append(ReliabilityBin(lo, hi, len(idx), avg_conf, acc))
    return bins


def expected_calibration_error(
    confidences: list[float],
    correctness: list[float],
    n_bins: int = 10,
) -> float | None:
    """Return ECE in [0, 1], or ``None`` if no confidences were available."""
    pairs = [
        (c, k)
        for c, k in zip(confidences, correctness)
        if c is not None
    ]
    if not pairs:
        return None
    conf, corr = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(conf)
    ece = 0.0
    for b in reliability_table(conf, corr, n_bins):
        if b.count:
            ece += (b.count / n) * abs(b.accuracy - b.avg_confidence)
    return ece
