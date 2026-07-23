#!/usr/bin/env python3
"""Validate the 100-method catalog and build its Markdown research report."""

from __future__ import annotations

import argparse
from pathlib import Path

from docvlm_eval.method_catalog import (
    load_method_catalog,
    render_method_survey,
    validate_method_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "frontier_method_catalog.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "report" / "frontier_method_survey.md",
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing the report.")
    args = parser.parse_args()

    rows = load_method_catalog(args.catalog)
    errors = validate_method_catalog(rows)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Validated {len(rows)} methods from {args.catalog}")
    if args.check:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_method_survey(rows), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
