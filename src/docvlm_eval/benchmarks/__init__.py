"""Benchmark loading + construction.

All benchmarks are materialised to a normalised JSONL of :class:`~docvlm_eval.schema.Sample`
records (see ``loaders.load_jsonl``). ``hf_builders`` converts public HuggingFace datasets
(DocVQA, InfoVQA, ChartQA, OCRBench) into that format; ``robustness`` builds our custom
probe on top of any base benchmark.
"""

from .loaders import load_jsonl, save_jsonl

__all__ = ["load_jsonl", "save_jsonl"]
