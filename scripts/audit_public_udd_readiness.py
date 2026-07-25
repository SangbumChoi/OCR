#!/usr/bin/env python3
"""Capture compact immutable readiness evidence for the public UDD component."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from docvlm_eval.student.dataset_readiness import (
    build_public_dataset_readiness,
    load_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def _viewer_info(repo_id: str) -> dict:
    query = urllib.parse.urlencode({"dataset": repo_id})
    with urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/info?{query}",
        timeout=60,
    ) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "configs" / "sub1b_experiment.yaml",
    )
    parser.add_argument("--repo-id", default="danelcsb/UDD")
    parser.add_argument(
        "--revision",
        default="f5eb52104627d20ddd1eab2130ad78f87cb0d7c9",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "results" / "public_udd_training_readiness.json",
    )
    args = parser.parse_args()

    api = HfApi()
    pinned = api.dataset_info(
        args.repo_id,
        revision=args.revision,
        files_metadata=True,
    )
    current = api.dataset_info(args.repo_id)
    card_path = hf_hub_download(
        args.repo_id,
        "README.md",
        repo_type="dataset",
        revision=args.revision,
    )
    files = [
        {
            "path": sibling.rfilename,
            "size": sibling.size,
            "sha256": (
                None if sibling.lfs is None else sibling.lfs.sha256
            ),
        }
        for sibling in pinned.siblings
    ]
    result = build_public_dataset_readiness(
        load_experiment(args.experiment),
        repo_id=args.repo_id,
        requested_revision=args.revision,
        resolved_revision=str(pinned.sha),
        main_revision=str(current.sha),
        card=Path(card_path).read_text(encoding="utf-8"),
        files=files,
        viewer_info=_viewer_info(args.repo_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fingerprint": result["fingerprint"],
                "output": str(args.output.resolve()),
                "rows": result["dataset"]["rows"],
                "status": result["overall_status"],
                "tasks": result["dataset"]["task_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
