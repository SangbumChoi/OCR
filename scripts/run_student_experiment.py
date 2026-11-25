#!/usr/bin/env python3
"""Run the complete synthetic-data to heldout-evaluation native student DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docvlm_eval.student.experiment import ExperimentRunner, build_experiment_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "configs" / "sub1b_experiment.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--to-stage")
    args = parser.parse_args()

    plan = build_experiment_plan(
        args.experiment,
        repo_root=ROOT,
        python=sys.executable,
    )
    result = ExperimentRunner(plan, repo_root=ROOT).run(
        dry_run=args.dry_run,
        resume=not args.no_resume,
        start=args.from_stage,
        stop=args.to_stage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
