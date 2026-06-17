"""Unified data structures shared across loaders, models and metrics.

Every benchmark - whether it comes from VLMEvalKit, a HuggingFace dataset, or our own
robustness probe - is normalised into a list of :class:`Sample`. Keeping one schema is
what lets a single evaluation script run any model on any benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """A single document-understanding question.

    Attributes
    ----------
    sample_id:
        Stable identifier (used to join predictions back to the dataset).
    image_path:
        Absolute or repo-relative path to the document image.
    question:
        The natural-language question / instruction shown to the model.
    answers:
        One or more gold answers. DocVQA/InfoVQA ship several human answers per
        question; ANLS scores against the best-matching one.
    answer_type:
        Free-form tag used for *slice* analysis (e.g. "table", "handwritten",
        "numeric", "free-text"). Drives the knowledge-gap breakdown.
    metric:
        Which scoring rule applies to this sample: one of
        {"anls", "relaxed_acc", "exact", "ocrbench"}.
    meta:
        Anything else a loader wants to carry through (perturbation name for the
        robustness probe, chart type, document category, ...).
    """

    sample_id: str
    image_path: str
    question: str
    answers: list[str]
    answer_type: str = "default"
    metric: str = "anls"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A model's answer to a :class:`Sample`, plus a confidence for calibration."""

    sample_id: str
    prediction: str
    # Sequence confidence in [0, 1]; mean token probability of the generated answer.
    # Used for Expected Calibration Error. ``None`` if the backend cannot expose logits.
    confidence: float | None = None
    raw: str = ""  # untrimmed model output, kept for debugging
