#!/usr/bin/env python3
"""Migrate an existing UDD snapshot to the fine-grained annotation schema.

The migration preserves source content, QA lists, full_text, and table_html. Existing rationale is
kept; missing rationale is represented by aligned empty strings, never generated. Upload is opt-in.

    python scripts/migrate_udd_schema.py --src examples/udd/hf/_all --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --push --token $HF_TOKEN
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import upgrade_udd_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", help="local Dataset or DatasetDict saved with save_to_disk")
    parser.add_argument("--repo", default="danelcsb/UDD", help="Hub source/target dataset repo")
    parser.add_argument("--out", help="optional local destination for the migrated snapshot")
    parser.add_argument("--push", action="store_true", help="publish the migrated dataset to --repo")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    if args.push and not args.token:
        parser.error("--push requires --token or HF_TOKEN")
    if args.src and not Path(args.src).exists():
        parser.error(f"local source does not exist: {args.src}")
    if args.out and Path(args.out).exists():
        parser.error(f"output path already exists; choose a new path: {args.out}")

    from datasets import load_dataset, load_from_disk

    dataset = load_from_disk(args.src) if args.src else load_dataset(args.repo)
    migrated = upgrade_udd_dataset(dataset)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        migrated.save_to_disk(args.out)
        print(f"[ok] migrated snapshot saved to {args.out}")
    for split, rows in migrated.items() if isinstance(migrated, dict) else [("train", migrated)]:
        visual_counts = Counter(rows["visual_type"])
        job_counts = Counter(detail for details in rows["task_detail"] for detail in details)
        rationale_count = sum(bool(value.strip()) for values in rows["reasoning"] for value in values)
        print(f"[{split}] rows={len(rows)} rationale_entries={rationale_count} "
              f"visual_types={dict(visual_counts)} "
              f"top_task_details={job_counts.most_common(8)}")
    if args.push:
        migrated.push_to_hub(args.repo, token=args.token)
        print(f"[ok] pushed migrated dataset to {args.repo}")


if __name__ == "__main__":
    main()
