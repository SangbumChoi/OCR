#!/usr/bin/env python3
"""Plan the next exact synthetic batch from structured validation failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.student.synthesis_policy import (
    file_fingerprint,
    load_evaluation_rows,
    load_synthesis_policy_config,
    plan_synthesis_batch,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument(
        "--baseline-per-sample",
        type=Path,
        help=(
            "matched baseline per-sample rows from the same evaluation split"
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--allow-heldout-analysis",
        action="store_true",
        help="emit a non-executable analysis plan from heldout rows",
    )
    args = parser.parse_args()

    rows = load_evaluation_rows(args.per_sample)
    baseline_rows = (
        load_evaluation_rows(args.baseline_per_sample)
        if args.baseline_per_sample is not None
        else None
    )
    config = load_synthesis_policy_config(args.config)
    plan = plan_synthesis_batch(
        rows,
        config,
        source_fingerprint=file_fingerprint(args.per_sample),
        source_path=str(args.per_sample.resolve()),
        baseline_rows=baseline_rows,
        baseline_source_fingerprint=(
            file_fingerprint(args.baseline_per_sample)
            if args.baseline_per_sample is not None
            else None
        ),
        baseline_source_path=(
            str(args.baseline_per_sample.resolve())
            if args.baseline_per_sample is not None
            else None
        ),
        budget=args.budget,
        seed=args.seed,
        allow_heldout_analysis=args.allow_heldout_analysis,
    )
    write_json_atomic(args.output, plan)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "training_authorized": plan["training_authorized"],
                "budget": plan["budget"],
                "candidate_count": plan["candidate_count"],
                "jobs": len(plan["jobs"]),
                "plan_fingerprint": plan["plan_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
