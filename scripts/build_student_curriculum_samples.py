#!/usr/bin/env python3
"""Merge all new failure-driven samples with deterministic parent replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.student.continuation import build_curriculum_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-samples", type=Path, required=True)
    parser.add_argument("--replay-samples", type=Path, required=True)
    parser.add_argument("--replay-fraction", type=float, required=True)
    parser.add_argument("--replay-seed", type=int, required=True)
    parser.add_argument("--parent-round-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_curriculum_samples(
        current_samples=args.current_samples,
        replay_samples=args.replay_samples,
        replay_fraction=args.replay_fraction,
        replay_seed=args.replay_seed,
        parent_round_index=args.parent_round_index,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
