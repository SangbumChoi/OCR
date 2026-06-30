#!/usr/bin/env python3
"""Load EVERY catalog benchmark through the unified loader and materialise one standardized dataset.

Streams a subset (<200 images/benchmark) of each streamable dataset, normalises it into the
task-typed :class:`~docvlm_eval.unified.UnifiedSample` (recognition / kie / vqa /
localization / table / reasoning) **preserving the structured payload** (KIE fields, localization
boxes, table HTML), caches images offline, and writes:

  <out>/images/<key>/NNNN.jpg     cached images
  <out>/unified.jsonl             rich task-typed records (fields/regions/boxes/table preserved)
  <out>/train.jsonl               flat trainable Samples (UnifiedSample.to_sample())
  <out>/by_task/<task>.jsonl      unified records grouped by task (easy per-task consumption)
  <out>/summary.json              per-benchmark + per-task counts

Examples
--------
    python scripts/build_unified_dataset.py --per-bench 50
    python scripts/build_unified_dataset.py --only cord,funsd,ocrvqa --per-bench 100
    python scripts/build_unified_dataset.py --task kie          # only KIE-yielding benchmarks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.benchmarks.loaders import save_jsonl  # noqa: E402
from docvlm_eval.unified import (TASK_BY_BENCHMARK, Task, UnifiedLoader,  # noqa: E402
                                            to_training_samples)

HARD_CAP = 200


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "unified_dataset"))
    p.add_argument("--per-bench", type=int, default=50, help=f"images per benchmark (< {HARD_CAP})")
    p.add_argument("--only", default=None, help="comma-separated benchmark keys to include")
    p.add_argument("--skip", default=None, help="comma-separated benchmark keys to skip")
    p.add_argument("--task", default=None,
                   help=f"only benchmarks whose default task is this ({'/'.join(Task.ALL)})")
    p.add_argument("--max-scan", type=int, default=3000)
    p.add_argument("--max-px", type=int, default=1000)
    args = p.parse_args()

    per = max(1, min(args.per_bench, HARD_CAP - 1))
    out = Path(args.out)
    (out / "by_task").mkdir(parents=True, exist_ok=True)

    loader = UnifiedLoader()
    only = {k.strip() for k in args.only.split(",")} if args.only else None
    skip = {k.strip() for k in args.skip.split(",")} if args.skip else set()
    keys = [k for k in loader.streamable_keys()
            if (only is None or k in only) and k not in skip
            and (args.task is None or TASK_BY_BENCHMARK.get(k) == args.task)]
    if not keys:
        print("No benchmarks matched the filter."); return

    print(f"Unified-loading <{per} images/benchmark from {len(keys)} datasets -> {out}\n")
    all_rows = []
    summary: dict[str, dict] = {}
    for k in keys:
        rows = loader.load(k, limit=per, max_scan=args.max_scan, max_px=args.max_px,
                           cache_dir=str(out / "images"))
        tasks = Counter(r.task for r in rows)
        summary[k] = {"records": len(rows), "images": len({r.image_path for r in rows}),
                      "tasks": dict(tasks),
                      "with_boxes": sum(1 for r in rows if any(f.bbox for f in r.fields) or
                                        any(rg.bbox for rg in r.regions))}
        all_rows.extend(rows)
        if rows:
            print(f"[ok] {k:14} {len(rows):4} records  tasks={dict(tasks)}")

    # rich unified jsonl
    (out / "unified.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in all_rows) + "\n",
        encoding="utf-8")
    # grouped by task
    by_task: dict[str, list] = {}
    for r in all_rows:
        by_task.setdefault(r.task, []).append(r)
    for task, rows in by_task.items():
        (out / "by_task" / f"{task}.jsonl").write_text(
            "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
    # flat trainable samples
    samples = to_training_samples(all_rows)
    save_jsonl(samples, out / "train.jsonl")

    task_totals = Counter(r.task for r in all_rows)
    (out / "summary.json").write_text(
        json.dumps({"per_benchmark": summary, "per_task": dict(task_totals),
                    "n_records": len(all_rows), "n_trainable": len(samples)}, indent=2,
                   ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {len(all_rows)} unified records from {len(summary)} benchmarks ===")
    print(f"  by task     : {dict(task_totals)}")
    print(f"  unified     : {out/'unified.jsonl'}  (task-typed, structured payload preserved)")
    print(f"  by_task/    : {out/'by_task'}/<task>.jsonl")
    print(f"  trainable   : {out/'train.jsonl'}  ({len(samples)} flat Samples)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort
