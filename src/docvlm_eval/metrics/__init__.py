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
from .bank import (METRIC_BANK, cer_sim, drop_em, drop_normalize, score_all, semantic_match,
                   token_f1)
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
    "METRIC_BANK",
    "score_all",
    "token_f1",
    "drop_em",
    "cer_sim",
    "semantic_match",
    "drop_normalize",
    "expected_calibration_error",
    "reliability_table",
    "aggregate",
    "teds",
    "teds_struct",
]
