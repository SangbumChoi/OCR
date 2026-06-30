"""Unified, task-typed data loading for every OCR/document dataset.

One schema (:class:`UnifiedSample`) + one loader (:class:`UnifiedLoader`) that normalises every
benchmark into a task-typed record preserving its structured payload (KIE fields, localization
boxes, table HTML, full text). See ``docs/report/unified_loader.md``.
"""

from .core import (Box, Field, Region, Task, TASK_BY_BENCHMARK, UnifiedLoader, UnifiedSample,
                   extract_unified, register, to_training_samples)
from .hf import push, safety_check, to_hf_dataset, udd_features
from .visualize import render_grid

__all__ = ["Task", "TASK_BY_BENCHMARK", "Box", "Field", "Region", "UnifiedSample",
           "UnifiedLoader", "extract_unified", "register", "to_training_samples", "render_grid",
           "udd_features", "to_hf_dataset", "safety_check", "push"]
