#!/usr/bin/env python3
"""Capture compact immutable readiness evidence for the public UDD component."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from docvlm_eval.student.dataset_readiness import (
    PILOT_SELECTION_POLICY,
    REQUIRED_UDD_TASKS,
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


def _filter_count(repo_id: str, where: str) -> int:
    query = urllib.parse.urlencode(
        {
            "dataset": repo_id,
            "config": "default",
            "split": "train",
            "where": where,
            "offset": 0,
            "length": 1,
        }
    )
    url = f"https://datasets-server.huggingface.co/filter?{query}"
    for attempt in range(10):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "docvlm-udd-readiness/1"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return int(json.load(response)["num_rows_total"])
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (
                exc.code in {429, 500, 502, 503, 504}
                and attempt < 9
            ):
                time.sleep(min(5 * (attempt + 1), 20))
                continue
            raise RuntimeError(
                f"Dataset Viewer filter failed for {where!r}: "
                f"HTTP {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < 9:
                time.sleep(min(5 * (attempt + 1), 20))
                continue
            raise RuntimeError(
                f"Dataset Viewer filter failed for {where!r}: {exc}"
            ) from exc
    raise RuntimeError("Dataset Viewer filter retry loop exhausted")


def _viewer_training_counts(
    repo_id: str,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    tasks = sorted(REQUIRED_UDD_TASKS)
    languages = list(PILOT_SELECTION_POLICY["coverage_languages"])
    requests = [
        ("task", task, None)
        for task in tasks
    ] + [
        ("intersection", task, language)
        for language in languages
        for task in tasks
    ]

    def fetch(
        request: tuple[str, str, str | None],
    ) -> tuple[str, str, str | None, int]:
        kind, task, language = request
        predicates = ['"fold"=\'train\'', f'"task"=\'{task}\'']
        if language is not None:
            predicates.append(f'"language"=\'{language}\'')
        count = _filter_count(repo_id, " AND ".join(predicates))
        return kind, task, language, count

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, requests))
    task_counts: dict[str, int] = {}
    intersections = {
        language: {task: 0 for task in tasks}
        for language in languages
    }
    for kind, task, language, count in results:
        if kind == "task":
            task_counts[task] = count
        else:
            assert language is not None
            intersections[language][task] = count
    return task_counts, intersections


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
    task_counts, task_language_counts = _viewer_training_counts(
        args.repo_id
    )
    result = build_public_dataset_readiness(
        load_experiment(args.experiment),
        repo_id=args.repo_id,
        requested_revision=args.revision,
        resolved_revision=str(pinned.sha),
        main_revision=str(current.sha),
        card=Path(card_path).read_text(encoding="utf-8"),
        files=files,
        viewer_info=_viewer_info(args.repo_id),
        train_fold_task_counts=task_counts,
        train_fold_task_language_counts=task_language_counts,
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
                "pilot_selection": result["pilot_selection_plan"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
