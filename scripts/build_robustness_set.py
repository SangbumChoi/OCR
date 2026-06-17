#!/usr/bin/env python3
"""Build the custom robustness probe from a base benchmark JSONL.

Takes (a subset of) e.g. DocVQA and emits a paired clean/perturbed JSONL that
``scripts/evaluate.py`` can run like any other benchmark. ``scripts/make_comparison_table.py``
then reads the per-perturbation slices to compute retention.

Example
-------
    python scripts/build_robustness_set.py \
        --base data/benchmarks/docvqa.jsonl \
        --out-dir data/robustness/docvqa --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks.loaders import load_jsonl, save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.robustness import VISUAL, build_robustness_set  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base benchmark JSONL to perturb")
    p.add_argument("--out-dir", default="data/robustness/docvqa")
    p.add_argument("--limit", type=int, default=100, help="#base samples to perturb")
    p.add_argument(
        "--perturbations",
        nargs="+",
        default=VISUAL + ["term_paraphrase"],
        help="subset of perturbations to apply",
    )
    args = p.parse_args()

    base = load_jsonl(args.base)[: args.limit]
    expanded = build_robustness_set(base, args.out_dir, perturbations=args.perturbations)
    out_jsonl = Path(args.out_dir) / "robustness.jsonl"
    save_jsonl(expanded, out_jsonl)
    print(f"[done] {len(base)} base -> {len(expanded)} probe samples -> {out_jsonl}")


if __name__ == "__main__":
    main()
