#!/usr/bin/env python3
"""Acquire one immutable Hugging Face model checkpoint for selective transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docvlm_eval.student.checkpoint_acquisition import (
    HubCheckpointSpec,
    acquire_hub_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--family",
        required=True,
        choices=["student", "siglip", "llama", "lfm2"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = acquire_hub_checkpoint(
        HubCheckpointSpec(
            repo_id=args.repo_id,
            revision=args.revision,
            family=args.family,
        ),
        args.output,
        token=os.environ.get("HF_TOKEN"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
