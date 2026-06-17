#!/usr/bin/env python3
"""Thin shim -> docvlm_eval.cli.fetch_samples (also installed as `docvlm-fetch`).

Catalog of all benchmarks lives in configs/benchmark_catalog.yaml.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.cli import fetch_samples

if __name__ == "__main__":
    fetch_samples()
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort
