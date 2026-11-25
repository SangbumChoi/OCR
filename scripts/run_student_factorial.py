#!/usr/bin/env python3
"""Run the initialization-by-data-scale native-student factorial."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.factorial import (
    FactorialRunner,
    compile_factorial_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorial",
        type=Path,
        default=ROOT / "configs" / "sub1b_initialization_data_scale.yaml",
    )
    parser.add_argument("--scale", action="append", dest="scales")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--replicate", action="append", dest="replicates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--to-stage")
    args = parser.parse_args()

    if args.dry_run:
        with tempfile.TemporaryDirectory(
            prefix="docvlm-factorial-"
        ) as temporary:
            plan = compile_factorial_plan(
                args.factorial,
                repo_root=ROOT,
                python=sys.executable,
                compile_root=temporary,
            )
            result = FactorialRunner(plan, repo_root=ROOT).run(
                dry_run=True,
                resume=not args.no_resume,
                scale_ids=set(args.scales) if args.scales else None,
                variant_ids=set(args.variants) if args.variants else None,
                replicate_ids=(
                    set(args.replicates) if args.replicates else None
                ),
                start=args.from_stage,
                stop=args.to_stage,
            )
    else:
        plan = compile_factorial_plan(
            args.factorial,
            repo_root=ROOT,
            python=sys.executable,
        )
        result = FactorialRunner(plan, repo_root=ROOT).run(
            resume=not args.no_resume,
            scale_ids=set(args.scales) if args.scales else None,
            variant_ids=set(args.variants) if args.variants else None,
            replicate_ids=(
                set(args.replicates) if args.replicates else None
            ),
            start=args.from_stage,
            stop=args.to_stage,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
