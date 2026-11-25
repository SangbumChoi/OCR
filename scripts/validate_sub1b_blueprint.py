#!/usr/bin/env python3
"""Validate the adjustable sub-1B architecture and training blueprint."""

from __future__ import annotations

import argparse
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    args = parser.parse_args()
    estimates, errors = validate_blueprint(load_blueprint(args.config))

    if estimates:
        for name in ("vision", "language", "connector", "task_heads", "total"):
            print(f"{name:>10}: {estimates[name]:>12,} parameters")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Blueprint is valid and below the deployment budget.")


if __name__ == "__main__":
    main()
