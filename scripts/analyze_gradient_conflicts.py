#!/usr/bin/env python3
"""Aggregate a completed replicated gradient-conflict audit."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.gradient_audit import (
    write_gradient_conflict_audit,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_gradient_conflict_audit.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-measurements", type=int, default=20)
    parser.add_argument("--negative-fraction", type=float, default=0.25)
    parser.add_argument("--mean-cosine", type=float, default=-0.05)
    parser.add_argument("--material-mean-delta", type=float, default=0.05)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="docvlm-gradient-audit-") as temporary:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        audit = write_gradient_conflict_audit(
            plan,
            output_dir=args.output_dir,
            minimum_measurements=args.minimum_measurements,
            negative_fraction_threshold=args.negative_fraction,
            mean_cosine_threshold=args.mean_cosine,
            material_mean_delta=args.material_mean_delta,
        )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
