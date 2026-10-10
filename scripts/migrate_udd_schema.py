#!/usr/bin/env python3
"""Migrate an existing UDD snapshot to the fine-grained annotation schema.

The migration preserves source content and scored gold answers, adding visual labels, task details,
and source-provided rationales. Legacy HallusionBench explanation QAs are paired with their original
questions. Missing rationales stay empty; none are generated. Upload is opt-in and targets a new
Hub branch so immutable training snapshots remain intact.

    python scripts/migrate_udd_schema.py --src examples/udd/hf/_all --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --revision fine-grained-v2 --push
"""
from __future__ import annotations

import argparse
import os
import re
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
    parser.add_argument("--revision", default="fine-grained-v2",
                        help="new Hub branch to publish; the existing default branch is never changed")
    parser.add_argument("--out", help="optional local destination for the migrated snapshot")
    parser.add_argument("--push", action="store_true", help="publish the migrated dataset to --repo")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    if args.push and args.revision in {"main", "master"}:
        parser.error("--push requires a new version branch; main/master are protected")
    if args.src and not Path(args.src).exists():
        parser.error(f"local source does not exist: {args.src}")
    if args.out and Path(args.out).exists():
        parser.error(f"output path already exists; choose a new path: {args.out}")

    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, get_token, hf_hub_download

    if args.push and not args.token:
        args.token = get_token()
        if not args.token:
            parser.error("--push requires --token, HF_TOKEN, or `hf auth login`")

    api = HfApi(token=args.token) if args.push else None
    base_revision = None
    if args.push:
        refs = api.list_repo_refs(args.repo, repo_type="dataset")
        if any(ref.name == args.revision for ref in refs.branches):
            parser.error(f"target branch {args.revision!r} already exists; choose a new branch")
        base_revision = api.dataset_info(args.repo, revision="main", files_metadata=False).sha
    dataset = (load_from_disk(args.src) if args.src else
               load_dataset(args.repo, revision=base_revision) if base_revision else
               load_dataset(args.repo))
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
        api.create_branch(args.repo, branch=args.revision, revision=base_revision,
                          repo_type="dataset", token=args.token)
        card_path = hf_hub_download(args.repo, "README.md", repo_type="dataset",
                                    revision=base_revision, token=args.token)
        card = _upgrade_dataset_card(Path(card_path).read_text(encoding="utf-8"))
        commit = migrated.push_to_hub(args.repo, token=args.token, revision=args.revision,
                                      commit_message="Add fine-grained UDD annotations")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset", revision=args.revision,
                        commit_message="Document fine-grained UDD schema")
        print(f"[ok] pushed migrated dataset to {args.repo}@{args.revision} "
              f"(based on {base_revision}; data commit {commit.oid})")


def _upgrade_dataset_card(card: str) -> str:
    """Update the existing Hub card without discarding its project description or citations."""
    feature_additions = (
        "  - name: visual_type\n    dtype: string\n"
        "  - name: visual_subtype\n    dtype: string\n"
        "  - name: task_detail\n    list: string\n"
    )
    marker = "  - name: task\n    dtype: string\n"
    if "  - name: visual_type\n" not in card:
        if marker not in card:
            raise ValueError("cannot update dataset card: task feature declaration not found")
        card = card.replace(marker, marker + feature_additions, 1)
    marker = "  - name: answers\n    list:\n      list: string\n"
    if "  - name: reasoning\n" not in card:
        if marker not in card:
            raise ValueError("cannot update dataset card: answers feature declaration not found")
        card = card.replace(marker, marker + "  - name: reasoning\n    list: string\n", 1)
    if "| `visual_type` |" not in card:
        table_rows = (
            "| `visual_type` | document / chart / table / diagram / scene_text / interface / webpage / formula / natural_image / mixed / other |\n"
            "| `visual_subtype` | source-supported finer visual category |\n"
            "| `task_detail` | list[string], fine-grained job aligned by QA index |\n"
        )
        lines = card.splitlines(keepends=True)
        task_index = next((i for i, line in enumerate(lines)
                           if line.startswith("| `task` |")), None)
        if task_index is None:
            raise ValueError("cannot update dataset card: task schema row not found")
        lines.insert(task_index + 1, table_rows)
        card = "".join(lines)
    if "| `reasoning` |" not in card:
        lines = card.splitlines(keepends=True)
        answer_index = next((i for i, line in enumerate(lines)
                             if line.startswith("| `answers` |")), None)
        if answer_index is None:
            raise ValueError("cannot update dataset card: answers schema row not found")
        lines.insert(answer_index + 1,
                     "| `reasoning` | list[string] | Optional source-provided rationale aligned with each QA; empty when unavailable |\n")
        card = "".join(lines)
    card, replacements = re.subn(
        r"HallusionBench is a `reasoning` source:.*?answers, and each question carries a paired rationale QA .*?\)\. POPE was removed by design:",
        "HallusionBench is a `reasoning` source: its raw \"0\"/\"1\" labels are normalized to pure **yes/no** answers, paired with the source explanation in `reasoning`; legacy synthetic explanation QAs are folded into that field. POPE was removed by design:",
        card, count=1,
    )
    if replacements != 1:
        raise ValueError("cannot update dataset card: legacy HallusionBench description not found")
    card = card.replace(
        "**Current release:** **39,837 image-rows / 77,063 QAs**",
        "**Fine-grained v2 release:** **39,837 image-rows / 76,730 scored QAs**",
        1,
    )
    card += (
        "\n### Fine-grained v2 schema\n"
        "This branch adds `visual_type` and `visual_subtype` for image content, plus "
        "`task_detail[i]` and `reasoning[i]` aligned with each instruction and answer. "
        "Rationales are source-provided only; missing rationales are empty. Existing scored gold "
        "answers and payload columns are preserved. The 333 legacy synthetic HallusionBench "
        "explanation QAs are folded into the rationale field, leaving 76,730 scored QAs. "
        "`full_text` is a linear reading-order "
        "transcript; `table_html` is structural table markup. A table may provide both, and "
        "they are not interchangeable. The default `main` revision remains unchanged.\n\n"
        "Load this version with `load_dataset(\"danelcsb/UDD\", revision=\"fine-grained-v2\")`.\n"
    )
    return card


if __name__ == "__main__":
    main()
