"""Unified, task-typed data loading for every OCR/document dataset.

One schema (:class:`UnifiedSample`) + one loader (:class:`UnifiedLoader`) that normalises every
benchmark into a task-typed record preserving its structured payload (KIE fields, localization
boxes, table HTML, full text). See ``docs/report/unified_loader.md``.
"""

from .core import (QA, Box, Field, Region, Task, TASK_BY_BENCHMARK, UnifiedLoader, UnifiedSample,
                   canon_answers, canon_key, derive_spatial_reasoning, extract_unified,
                   merge_by_image, register, to_training_samples)
from .enrich import detect_language, dhash, enrich_dataset, enrich_record, hamming
from .hf import (dedupe_by_phash, push, safety_check, to_hf_dataset, udd_features,
                 unified_from_hf_row)
from .synth_bridge import docsample_to_unified
from .visualize import render_grid

__all__ = ["Task", "TASK_BY_BENCHMARK", "Box", "Field", "Region", "QA", "UnifiedSample",
           "UnifiedLoader", "extract_unified", "register", "merge_by_image", "to_training_samples",
           "render_grid", "udd_features", "to_hf_dataset", "safety_check", "push",
           "unified_from_hf_row", "detect_language", "enrich_dataset", "enrich_record",
           "dhash", "hamming", "canon_answers", "canon_key", "derive_spatial_reasoning",
           "dedupe_by_phash", "docsample_to_unified"]
