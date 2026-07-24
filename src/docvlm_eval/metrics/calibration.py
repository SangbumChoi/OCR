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

import math
from dataclasses import dataclass


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass(frozen=True)
class TemperatureScalingResult:
    status: str
    temperature: float | None
    n_samples: int
    n_correct: int
    n_incorrect: int
    raw_nll: float | None
    calibrated_nll: float | None
    reason: str | None = None


def temperature_scale_confidence(
    confidence: float,
    temperature: float,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Apply scalar temperature scaling to a probability-like confidence."""

    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    clipped = min(max(float(confidence), epsilon), 1.0 - epsilon)
    logit = math.log(clipped) - math.log1p(-clipped)
    scaled = logit / float(temperature)
    if scaled >= 0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exp_scaled = math.exp(scaled)
    return exp_scaled / (1.0 + exp_scaled)


def _binary_nll(
    confidences: list[float],
    correctness: list[float],
    temperature: float,
) -> float:
    values = [
        temperature_scale_confidence(confidence, temperature)
        for confidence in confidences
    ]
    epsilon = 1e-12
    return sum(
        -(
            target * math.log(max(value, epsilon))
            + (1.0 - target) * math.log(max(1.0 - value, epsilon))
        )
        for value, target in zip(values, correctness)
    ) / len(values)


def fit_temperature_scaling(
    confidences: list[float],
    correctness: list[float],
    *,
    min_samples: int = 20,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    iterations: int = 96,
) -> TemperatureScalingResult:
    """Fit one temperature by bounded golden-section search on binary NLL."""

    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have equal length")
    if min_samples <= 0 or iterations <= 0:
        raise ValueError("min_samples and iterations must be positive")
    if (
        not math.isfinite(min_temperature)
        or not math.isfinite(max_temperature)
        or not 0 < min_temperature < max_temperature
    ):
        raise ValueError("temperature bounds must be positive, finite, and ordered")
    pairs = [
        (float(confidence), float(target))
        for confidence, target in zip(confidences, correctness)
        if confidence is not None
    ]
    for confidence, target in pairs:
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if target not in {0.0, 1.0}:
            raise ValueError("correctness values must be binary")
    n_samples = len(pairs)
    n_correct = sum(int(target) for _, target in pairs)
    n_incorrect = n_samples - n_correct
    if n_samples < min_samples:
        return TemperatureScalingResult(
            status="insufficient_evidence",
            temperature=None,
            n_samples=n_samples,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            raw_nll=None,
            calibrated_nll=None,
            reason=f"requires at least {min_samples} confidence-bearing samples",
        )
    if n_correct == 0 or n_incorrect == 0:
        return TemperatureScalingResult(
            status="insufficient_evidence",
            temperature=None,
            n_samples=n_samples,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            raw_nll=None,
            calibrated_nll=None,
            reason="requires at least one correct and one incorrect sample",
        )
    confidence_values = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    lo = math.log(min_temperature)
    hi = math.log(max_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)
    left_loss = _binary_nll(confidence_values, targets, math.exp(left))
    right_loss = _binary_nll(confidence_values, targets, math.exp(right))
    for _ in range(iterations):
        if left_loss <= right_loss:
            hi, right, right_loss = right, left, left_loss
            left = hi - ratio * (hi - lo)
            left_loss = _binary_nll(
                confidence_values,
                targets,
                math.exp(left),
            )
        else:
            lo, left, left_loss = left, right, right_loss
            right = lo + ratio * (hi - lo)
            right_loss = _binary_nll(
                confidence_values,
                targets,
                math.exp(right),
            )
    temperature = math.exp((lo + hi) / 2.0)
    return TemperatureScalingResult(
        status="fitted",
        temperature=temperature,
        n_samples=n_samples,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        raw_nll=_binary_nll(confidence_values, targets, 1.0),
        calibrated_nll=_binary_nll(
            confidence_values,
            targets,
            temperature,
        ),
    )


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
