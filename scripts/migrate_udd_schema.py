#!/usr/bin/env python3
"""Migrate the fine-grained v2 UDD snapshot to the compact v3 public schema.

The migration preserves source content and scored gold answers, adding visual labels, task details,
and source-provided rationales. Legacy HallusionBench explanation QAs are paired with their original
questions. Missing rationales stay empty; none are generated. Upload is opt-in and targets a new
Hub branch so immutable training snapshots remain intact.

    python scripts/migrate_udd_schema.py --src examples/udd/hf/_all --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --out examples/udd/hf/_all_v2
    python scripts/migrate_udd_schema.py --repo danelcsb/UDD --push
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
    parser.add_argument("--revision", default="fine-grained-v3",
                        help="new Hub branch to publish; the existing default branch is never changed")
    parser.add_argument("--source-revision", default="fine-grained-v2",
                        help="Hub revision to migrate (default: fine-grained-v2)")
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
        base_revision = api.dataset_info(args.repo, revision=args.source_revision,
                                         files_metadata=False).sha
    dataset = (load_from_disk(args.src) if args.src else
               load_dataset(args.repo, revision=args.source_revision))
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
                                      commit_message="Publish compact fine-grained UDD v3 schema")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset", revision=args.revision,
                        commit_message="Document fine-grained UDD v3 schema")
        print(f"[ok] pushed migrated dataset to {args.repo}@{args.revision} "
              f"(based on {args.source_revision}@{base_revision}; data commit {commit.oid})")


def _upgrade_dataset_card(card: str) -> str:
    """Update the inherited v2 card to exactly describe the v3 table."""
    lines = card.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line == "  - name: answers\n"), None)
    end = next((i for i, line in enumerate(lines[start + 1:], start + 1)
                if line.startswith("  - name: ")), None) if start is not None else None
    if start is None or end is None:
        raise ValueError("cannot update dataset card: answer feature declaration not found")
    lines[start:end] = [
        "  - name: answers\n", "    list: string\n",
        "  - name: answer_variants\n", "    list:\n", "      list: string\n",
    ]
    card = "".join(lines)
    card = re.sub(r"(?m)^  - name: (?:full_text|table_html|n_fields|n_regions)\n(?:    .*\n)+", "", card)
    reasoning = "  - name: reasoning\n    list: string\n"
    card = card.replace(reasoning, reasoning + "  - name: full_annotation\n    dtype: string\n", 1)
    table_start = card.index("| `answers` |")
    table_end = card.find("\n", table_start) + 1
    card = card[:table_start] + (
        "| `answers` | list[string] | One primary answer per instruction, aligned by index |\n"
        "| `answer_variants` | list[list[string]] | All accepted gold variants for each QA, aligned by index |\n"
    ) + card[table_end:]
    card = re.sub(r"(?m)^\| `(?:full_text|table_html|n_fields|n_regions)` \|.*\n", "", card)
    reasoning_row = "| `reasoning` | list[string] | Optional source-provided rationale aligned with each QA; empty when unavailable |\n"
    if reasoning_row in card:
        card = card.replace(reasoning_row, reasoning_row +
                            "| `full_annotation` | string(JSON) | Versioned object containing the full text transcript and/or table HTML |\n", 1)
    card = card.replace("**Fine-grained v2 release:**", "**Fine-grained v3 release:**", 1)
    card = re.sub(r"### Fine-grained v2 schema.*$", "", card, flags=re.DOTALL)
    card += (
        "\n### Fine-grained v3 schema\n"
        "This branch keeps `instructions`, primary `answers`, `answer_variants`, `reasoning`, and "
        "`task_detail` aligned one-to-one. The former nested answers held alternate valid gold "
        "phrases; these now live explicitly in `answer_variants`, preserving scorer behavior while "
        "making the primary answer unambiguous. `full_annotation` is a versioned JSON object with "
        "`text` and `html` members, so transcript and table markup are retained together. The "
        "derived `n_fields` and `n_regions` columns are removed; counts can be derived from "
        "`elements_json`. The default `main` and `fine-grained-v2` revisions remain unchanged.\n\n"
        "Load this version with `load_dataset(\"danelcsb/UDD\", revision=\"fine-grained-v3\")`.\n"
    )
    return card


if __name__ == "__main__":
    main()
