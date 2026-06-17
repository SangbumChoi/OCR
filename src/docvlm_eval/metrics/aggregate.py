"""Aggregate per-sample scores into a model/benchmark summary.

Produces the row that lands in the comparison table: headline score, calibration, and a
per-slice breakdown (by ``answer_type``) that powers the knowledge-gap analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..schema import Prediction, Sample
from .text import score_sample
from .calibration import expected_calibration_error


def aggregate(
    samples: list[Sample],
    predictions: dict[str, Prediction],
    correct_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute headline + sliced metrics for one (model, benchmark) run.

    Parameters
    ----------
    samples:
        The benchmark items that were evaluated.
    predictions:
        Mapping ``sample_id -> Prediction``.
    correct_threshold:
        A sample counts as "correct" (for accuracy & calibration) when its task score is
        at or above this value. ANLS uses 0.5 by convention; binary metrics use 1.0-style
        scores so the threshold is harmless.
    """
    per_sample: list[dict[str, Any]] = []
    scores: list[float] = []
    confidences: list[float] = []
    correctness: list[float] = []
    by_slice: dict[str, list[float]] = defaultdict(list)
    n_answered = 0

    for s in samples:
        pred = predictions.get(s.sample_id)
        pred_text = pred.prediction if pred else ""
        conf = pred.confidence if pred else None
        sc = score_sample(s.metric, pred_text, s.answers)
        is_correct = 1.0 if sc >= correct_threshold else 0.0

        scores.append(sc)
        correctness.append(is_correct)
        by_slice[s.answer_type].append(sc)
        if pred and pred.prediction.strip():
            n_answered += 1
        if conf is not None:
            confidences.append(conf)

        per_sample.append(
            {
                "sample_id": s.sample_id,
                "answer_type": s.answer_type,
                "metric": s.metric,
                "score": round(sc, 4),
                "correct": is_correct,
                "confidence": conf,
                "prediction": pred_text,
                "answers": s.answers,
                **({"perturbation": s.meta["perturbation"]} if "perturbation" in s.meta else {}),
            }
        )

    n = len(samples)
    summary: dict[str, Any] = {
        "n_samples": n,
        "primary_metric": _dominant_metric(samples),
        "score": round(sum(scores) / n, 4) if n else 0.0,
        "accuracy": round(sum(correctness) / n, 4) if n else 0.0,
        "answer_rate": round(n_answered / n, 4) if n else 0.0,
        "ece": (
            round(e, 4)
            if (e := expected_calibration_error(
                [predictions[s.sample_id].confidence if predictions.get(s.sample_id) else None for s in samples],
                correctness,
            )) is not None
            else None
        ),
        "by_answer_type": {
            k: {"n": len(v), "score": round(sum(v) / len(v), 4)} for k, v in sorted(by_slice.items())
        },
    }
    return {"summary": summary, "per_sample": per_sample}


def _dominant_metric(samples: list[Sample]) -> str:
    if not samples:
        return "anls"
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.metric] += 1
    return max(counts, key=counts.get)
