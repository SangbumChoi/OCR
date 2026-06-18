#!/usr/bin/env python3
"""Prettify (align) markdown tables in-place across the repo's generated docs.

    python scripts/prettify_md.py            # results/ + report/ + data/benchmarks/README.md
    python scripts/prettify_md.py path.md ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.report_md import prettify_tables  # noqa: E402


def targets() -> list[Path]:
    out = list((ROOT / "results").glob("*.md")) + list((ROOT / "report").glob("*.md"))
    out += list((ROOT / "data" / "benchmarks").glob("**/README.md"))
    return out


def main():
    paths = [Path(a) for a in sys.argv[1:]] or targets()
    changed = 0
    for p in paths:
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8")
        new = prettify_tables(s)
        if new != s:
            p.write_text(new, encoding="utf-8")
            changed += 1
            print(f"[pretty] {p.relative_to(ROOT)}")
    print(f"[done] prettified {changed} file(s)")


if __name__ == "__main__":
    main()
