#!/usr/bin/env python3
"""Audit adopted frontier methods against live implementation and test anchors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from docvlm_eval.method_catalog import (
    audit_adopted_method_evidence,
    load_method_catalog,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "frontier_method_catalog.jsonl",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "configs" / "frontier_method_evidence.yaml",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = yaml.safe_load(args.evidence.read_text(encoding="utf-8"))
    report = audit_adopted_method_evidence(
        load_method_catalog(args.catalog),
        evidence,
        repo_root=args.repo_root,
    )
    _write_json(args.output, report)
    print(
        f"method evidence audit: {report['status']} "
        f"({report['certified_methods']}/{report['adopted_methods']} adopted)"
    )
    if report["status"] != "pass":
        raise SystemExit(f"method evidence audit failed; inspect {args.output}")


if __name__ == "__main__":
    main()
