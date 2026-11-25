#!/usr/bin/env python3
"""Prepare the next failure-driven student experiment from a completed parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docvlm_eval.student.continuation import (
    prepare_next_round_spec,
    write_round_spec,
)
from docvlm_eval.student.experiment import build_experiment_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-compile-check", action="store_true")
    args = parser.parse_args()

    spec = prepare_next_round_spec(
        parent_root=args.parent_root,
        output_root=args.output_root,
        round_index=args.round_index,
        replay_fraction=args.replay_fraction,
        replay_seed=(
            args.replay_seed
            if args.replay_seed is not None
            else 20_000 + args.round_index
        ),
    )
    write_round_spec(spec, args.output)
    result: dict[str, object] = {
        "experiment": str(args.output.resolve()),
        "output_root": str(args.output_root.resolve()),
        "round_index": args.round_index,
    }
    if not args.no_compile_check:
        plan = build_experiment_plan(
            args.output,
            repo_root=ROOT,
            python=sys.executable,
        )
        result.update(
            {
                "fingerprint": plan.fingerprint,
                "stages": plan.stage_names,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
