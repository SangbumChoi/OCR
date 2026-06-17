#!/usr/bin/env python3
"""Thin shim -> docvlm_eval.cli.build_robustness (also installed as `docvlm-robustness`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.cli import build_robustness

if __name__ == "__main__":
    build_robustness()
