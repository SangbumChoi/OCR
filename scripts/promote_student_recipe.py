#!/usr/bin/env python3
"""Materialize one statistically promoted sweep arm as a canonical recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docvlm_eval.student.promotion import materialize_promoted_recipe


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_sweep.yaml",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Defaults to the compiled sweep output's comparison.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = materialize_promoted_recipe(
        args.sweep,
        args.output,
        repo_root=ROOT,
        python=sys.executable,
        comparison_path=args.comparison,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
