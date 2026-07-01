#!/usr/bin/env python3
"""Split UDD into **equal-sized per-task training sets** for the task-value ablation.

To decide whether adding a task (vqa / kie / recognition / table / reasoning / localization) is
*worth it*, we fine-tune on **one task at a time with the SAME number of samples** and compare the
effect on a fixed validation suite. This script produces those equal-N per-task training jsonls from
the merged UDD (``data/udd/hf/_all`` on disk, or a Hub repo), fully offline:

  <out>/images/<task>/<sample_id>.jpg     decoded images (HF rows store bytes; training wants paths)
  <out>/task_<task>.jsonl                 flat training Samples for ONE task (== N samples)
  <out>/all.jsonl                         the union (mixed-task control)
  <out>/summary.json                      per-task available vs used counts, the balanced N

The balanced N defaults to the smallest task's size (so every task gets an equal budget — the fair
"same amount" comparison); override with ``--per-task``. ``--merge-qa`` first merges duplicate-image
QAs into one record (a Q/A list) before counting, so VQA "samples" are counted per question but share
one image decode.

    python scripts/build_task_trainsets.py                       # equal N = smallest task
    python scripts/build_task_trainsets.py --per-task 30         # fixed budget per task
    python scripts/build_task_trainsets.py --repo danelcsb/UDD   # pull from the Hub instead
Then feed each task_<task>.jsonl to scripts/run_task_value.py (GPU) to get the value comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.benchmarks.loaders import save_jsonl  # noqa: E402
from docvlm_eval.unified import merge_by_image, to_training_samples, unified_from_hf_row  # noqa: E402


def _load(src: str | None, repo: str | None):
    if repo:
        from datasets import load_dataset
        return load_dataset(repo, split="train")
    from datasets import load_from_disk
    return load_from_disk(src)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"),
                   help="local load_from_disk path (the merged UDD)")
    p.add_argument("--repo", default=None, help="pull UDD from this Hub repo instead of --src")
    p.add_argument("--out", default=str(ROOT / "data" / "udd_tasks"))
    p.add_argument("--per-task", type=int, default=0,
                   help="samples per task (0 = the smallest task's size, i.e. equal budget for all)")
    p.add_argument("--tasks", default=None, help="comma-separated tasks to include (default: all present)")
    p.add_argument("--merge-qa", action="store_true",
                   help="merge duplicate-image QAs into a Q/A list before counting (one image decode "
                        "per group; VQA counted per question)")
    p.add_argument("--seed", type=int, default=7, help="shuffle seed for the balanced subsample")
    args = p.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = _load(args.src, args.repo)
    want = {t.strip() for t in args.tasks.split(",")} if args.tasks else None

    # 1) reconstruct UnifiedSamples, decoding each image to disk (training needs a path, not bytes)
    by_task: dict[str, list] = defaultdict(list)
    for i in range(len(ds)):
        row = {k: ds[k][i] for k in ds.column_names if k != "image"}
        task = row.get("task")
        if want and task not in want:
            continue
        tdir = out / "images" / task
        tdir.mkdir(parents=True, exist_ok=True)
        # name by IMAGE identity (sample_id minus the trailing QA index) so same-image QAs share one
        # file — dedups disk and lets merge_by_image (keyed on image_path) group them.
        img_key = row["sample_id"].rsplit("_", 1)[0] or row["sample_id"]
        ip = tdir / f"{img_key}.jpg"
        if not ip.exists():
            ds[i]["image"].convert("RGB").save(ip, quality=90)
        by_task[task].append(unified_from_hf_row(row, image_path=str(ip)))

    # 2) optional merge of duplicate-image QAs (a Q/A list per image) BEFORE we count samples
    if args.merge_qa:
        by_task = {t: merge_by_image(rows) for t, rows in by_task.items()}

    # 3) expand to flat training Samples per task, then balance to an EQUAL N across tasks
    import random as _random
    rng = _random.Random(args.seed)
    samples_by_task = {t: to_training_samples(rows) for t, rows in by_task.items()}
    samples_by_task = {t: s for t, s in samples_by_task.items() if s}   # drop empty tasks
    if not samples_by_task:
        print("[task-value] no trainable samples found — check --src / --tasks."); return
    n_balanced = args.per_task or min(len(s) for s in samples_by_task.values())

    summary, all_samples = {}, []
    print(f"[task-value] equal budget N={n_balanced} samples/task  (from {args.repo or args.src})\n")
    for task in sorted(samples_by_task):
        pool = list(samples_by_task[task])
        rng.shuffle(pool)
        used = pool[:n_balanced]
        save_jsonl(used, out / f"task_{task}.jsonl")
        all_samples += used
        summary[task] = {"available": len(pool), "used": len(used),
                         "images": len({s.image_path for s in used})}
        print(f"[ok]   {task:14} available={len(pool):4} used={len(used):4} "
              f"images={summary[task]['images']}")

    save_jsonl(all_samples, out / "all.jsonl")
    (out / "summary.json").write_text(
        json.dumps({"n_balanced": n_balanced, "merge_qa": args.merge_qa, "tasks": summary},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== built {len(summary)} per-task sets (N={n_balanced} each) -> {out} ===")
    print(f"  per-task jsonl : {out}/task_<task>.jsonl   (feed to run_task_value.py)")
    print(f"  mixed control  : {out}/all.jsonl")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort
