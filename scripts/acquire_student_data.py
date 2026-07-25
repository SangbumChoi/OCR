#!/usr/bin/env python3
"""Acquire and validate one pinned Hugging Face UDD component."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docvlm_eval.student.acquisition import HubComponentSpec, acquire_hub_component


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name")
    parser.add_argument("--fold", default="train")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decode-checks", type=int, default=16)
    parser.add_argument(
        "--sampling-strategy",
        choices=("global_hash", "task_stratified"),
        default="global_hash",
    )
    parser.add_argument("--min-rows-per-task", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = acquire_hub_component(
        HubComponentSpec(
            repo_id=args.repo_id,
            revision=args.revision,
            split=args.split,
            config_name=args.config_name,
            fold=(args.fold or None),
            sources=tuple(args.source),
            tasks=tuple(args.task),
            languages=tuple(args.language),
            max_rows=args.max_rows,
            seed=args.seed,
            decode_checks=args.decode_checks,
            sampling_strategy=args.sampling_strategy,
            min_rows_per_task=args.min_rows_per_task,
        ),
        args.output,
        token=os.environ.get("HF_TOKEN"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
