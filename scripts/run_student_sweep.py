#!/usr/bin/env python3
"""Compile and run matched native-student experiment variants."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.sweep import SweepRunner, compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_sweep.yaml",
    )
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--replicate", action="append", dest="replicates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full-dry-run",
        action="store_true",
        help="include every per-run command instead of the compact topology summary",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--to-stage")
    args = parser.parse_args()
    if args.full_dry_run and not args.dry_run:
        parser.error("--full-dry-run requires --dry-run")

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="docvlm-sweep-") as temporary:
            plan = compile_sweep_plan(
                args.sweep,
                repo_root=ROOT,
                python=sys.executable,
                compile_root=temporary,
            )
            result = SweepRunner(plan, repo_root=ROOT).run(
                dry_run=True,
                dry_run_detail="full" if args.full_dry_run else "compact",
                resume=not args.no_resume,
                variant_ids=set(args.variants) if args.variants else None,
                replicate_ids=set(args.replicates) if args.replicates else None,
                start=args.from_stage,
                stop=args.to_stage,
            )
    else:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
        )
        result = SweepRunner(plan, repo_root=ROOT).run(
            resume=not args.no_resume,
            variant_ids=set(args.variants) if args.variants else None,
            replicate_ids=set(args.replicates) if args.replicates else None,
            start=args.from_stage,
            stop=args.to_stage,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
