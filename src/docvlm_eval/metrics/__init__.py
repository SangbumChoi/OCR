"""Scoring functions for document-understanding evaluation.

We deliberately separate *correctness* metrics (ANLS, relaxed accuracy, exact match,
OCRBench) from *reliability* metrics (calibration ECE, robustness delta), because the
task asks us to go beyond generic VQA accuracy and probe properties that matter for
deploying a document reader: does it know when it is wrong (calibration), and does it
hold up under realistic document noise / domain terms (robustness)?
"""

from .text import (
    anls,
    cer,
    exact_match,
    ned_similarity,
    normalize_text,
    relaxed_accuracy,
    ocrbench_score,
    score_sample,
    wer,
)
from .calibration import expected_calibration_error, reliability_table
from .aggregate import aggregate
from .tables import teds, teds_struct

__all__ = [
    "anls",
    "cer",
    "wer",
    "ned_similarity",
    "exact_match",
    "normalize_text",
    "relaxed_accuracy",
    "ocrbench_score",
    "score_sample",
    "expected_calibration_error",
    "reliability_table",
    "aggregate",
    "teds",
    "teds_struct",
]
