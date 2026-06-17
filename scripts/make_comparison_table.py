#!/usr/bin/env python3
"""Thin shim -> docvlm_eval.cli.comparison_table (also installed as `docvlm-table`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.cli import comparison_table

if __name__ == "__main__":
    comparison_table()
