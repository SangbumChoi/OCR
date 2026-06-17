#!/usr/bin/env python3
"""Build normalised benchmark JSONL from public HuggingFace datasets.

Run on a machine with `datasets` + network (e.g. Colab). Produces
``data/benchmarks/<name>.jsonl`` + an ``images/`` folder, ready for ``scripts/evaluate.py``.

Examples
--------
    python scripts/build_benchmarks.py --benchmark docvqa --limit 500
    python scripts/build_benchmarks.py --benchmark all --limit 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks.hf_builders import BUILDERS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", choices=list(BUILDERS) + ["all"], required=True)
    p.add_argument("--out-dir", default="data/benchmarks")
    p.add_argument("--limit", type=int, default=None, help="cap #samples (fast subset)")
    args = p.parse_args()

    names = list(BUILDERS) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        print(f"[build] {name} ...")
        path = BUILDERS[name](args.out_dir, limit=args.limit)
        print(f"[done]  {path}")


if __name__ == "__main__":
    main()
