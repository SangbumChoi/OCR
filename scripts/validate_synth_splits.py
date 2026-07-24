#!/usr/bin/env python3
"""Validate semantic leakage across generated synthetic train/validation/heldout roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.synth.splits import validate_split_leakage  # noqa: E402


def _parse_split(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--split must use NAME=PATH")
    name, path = raw.split("=", 1)
    if name not in {"train", "validation", "heldout"}:
        raise argparse.ArgumentTypeError("split NAME must be train, validation, or heldout")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        type=_parse_split,
        required=True,
        metavar="NAME=PATH",
        help="generated split root; repeat for every split",
    )
    parser.add_argument(
        "--require-template-isolation",
        action="store_true",
        help="also reject the same graph-program template across splits",
    )
    parser.add_argument(
        "--require-layout-isolation",
        action="store_true",
        help="also require and isolate visual layout families across splits",
    )
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()

    records: list[dict] = []
    roots: dict[str, str] = {}
    for split, root in args.split:
        roots[split] = str(root.resolve())
        for path in sorted(root.rglob("gt.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["split"] = split
            record["_path"] = str(path)
            records.append(record)
    if not records:
        raise SystemExit("no gt.json records found")

    report = validate_split_leakage(
        records,
        require_template_isolation=args.require_template_isolation,
        require_layout_isolation=args.require_layout_isolation,
    )
    report["roots"] = roots
    report["records"] = len(records)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
