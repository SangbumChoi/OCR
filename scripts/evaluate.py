#!/usr/bin/env python3
"""Thin shim -> docvlm_eval.cli.evaluate (also installed as `docvlm-eval`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.cli import evaluate

if __name__ == "__main__":
    evaluate()
