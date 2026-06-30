#!/usr/bin/env python3
"""Build & upload the **UDD — Unified Document Dataset** to the Hub.

Per-dataset converter → uniform UDD schema → **safety-checked** → sharded → uploaded as one config
per benchmark (plus a combined ``all`` config) under a single HF dataset repo.

Memory/scale notes: each benchmark is streamed and converted **independently** (bounded by
``--per-bench``), safety-checked, saved to disk, and — with ``--push`` — uploaded and then **freed**
before the next one, so the combined corpus never has to live in RAM at once. Uploads are sharded
(``--max-shard-size``) so re-runs/downloads are resumable and partial.

MOCKUP (default): ``--per-bench 10`` — ten examples per dataset, saved locally + safety-checked +
visualized. Add ``--push --repo <user>/UDD --token <hf_token>`` to actually upload.

    # local mockup (no upload): 10 examples/dataset, safety-check, visualize
    python scripts/build_udd.py --only cord,funsd,ocrvqa,docvqa --per-bench 10
    # upload the mockup to your HF
    python scripts/build_udd.py --per-bench 10 --push --repo <user>/UDD --token $HF_TOKEN
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import (UnifiedLoader, push, render_grid,  # noqa: E402
                                 safety_check, to_hf_dataset)

HARD_CAP = 200


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "udd"))
    p.add_argument("--per-bench", type=int, default=10, help=f"examples/dataset (mockup=10; < {HARD_CAP})")
    p.add_argument("--only", default=None, help="comma-separated benchmark keys")
    p.add_argument("--skip", default=None, help="comma-separated benchmark keys to skip")
    p.add_argument("--max-scan", type=int, default=400)
    p.add_argument("--max-px", type=int, default=1000)
    p.add_argument("--push", action="store_true", help="upload to the Hub (needs --repo + token)")
    p.add_argument("--repo", default=None, help="target HF dataset repo, e.g. <user>/UDD")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--public", action="store_true", help="push as a public dataset (default private)")
    p.add_argument("--max-shard-size", default="500MB")
    p.add_argument("--no-combined", action="store_true", help="skip the combined 'all' config")
    p.add_argument("--viz", default=str(ROOT / "docs" / "report" / "figures" / "udd_examples.png"))
    args = p.parse_args()

    if args.push and not (args.repo and args.token):
        sys.exit("[udd] --push needs --repo <user>/UDD and a token (--token or $HF_TOKEN). "
                 "Run `hf auth login` or pass --token.")

    per = max(1, min(args.per_bench, HARD_CAP - 1))
    out = Path(args.out)
    loader = UnifiedLoader()
    only = [k.strip() for k in args.only.split(",")] if args.only else None
    skip = {k.strip() for k in args.skip.split(",")} if args.skip else set()
    keys = [k for k in loader.streamable_keys()
            if (only is None or k in only) and k not in skip]
    if not keys:
        print("No benchmarks matched the filter."); return

    mode = "PUSH" if args.push else "LOCAL MOCKUP"
    print(f"[udd] {mode}: {per} examples/dataset from {len(keys)} datasets -> {out}"
          + (f" -> {args.repo}" if args.push else "") + "\n")

    viz_rows = []
    report: dict[str, dict] = {}
    for k in keys:
        rows = loader.load(k, limit=per, max_scan=args.max_scan, max_px=args.max_px,
                           cache_dir=str(out / "images"))
        if not rows:
            print(f"[skip] {k}: no records"); continue
        # 1) safety-check the converter BEFORE trusting/uploading it
        try:
            rep = safety_check(rows, str(out / "hf" / k))
        except Exception as exc:
            print(f"[FAIL] {k}: safety check failed -> {type(exc).__name__}: {exc}"); continue
        report[k] = {"records": len(rows), **rep}
        print(f"[ok]   {k:14} {rep['rows']:4} rows  fields={rep['fields']} regions={rep['regions']} "
              f"image_ok={rep['image_ok']}")
        viz_rows.append(rows[0])
        # 2) upload this dataset as its own config, then free it
        if args.push:
            ds = to_hf_dataset(rows)
            push(ds, args.repo, config_name=k, token=args.token,
                 private=not args.public, max_shard_size=args.max_shard_size)
            print(f"       pushed config '{k}' -> {args.repo}")
            del ds

    # combined "all" config (concat) — built once at the end
    if not args.no_combined and report:
        all_rows = []
        for k in report:                      # reload from the per-dataset save (cheap, on-disk)
            from datasets import load_from_disk
            all_rows.append(load_from_disk(str(out / "hf" / k)))
        if all_rows:
            from datasets import concatenate_datasets
            combined = concatenate_datasets(all_rows)
            combined.save_to_disk(str(out / "hf" / "_all"))
            print(f"\n[ok] combined 'all': {len(combined)} rows -> {out/'hf'/'_all'}")
            if args.push:
                push(combined, args.repo, config_name="all", token=args.token,
                     private=not args.public, max_shard_size=args.max_shard_size)
                print(f"     pushed config 'all' -> {args.repo}")

    # 3) visualize the unified mockup
    if viz_rows:
        render_grid(viz_rows, args.viz, title=f"UDD mockup — {len(viz_rows)} datasets")

    print(f"\n=== UDD {'uploaded' if args.push else 'mockup built'} for {len(report)} datasets ===")
    if not args.push:
        print(f"  local HF datasets: {out/'hf'}/<dataset>  (load_from_disk)")
        print(f"  to upload:  python scripts/build_udd.py --per-bench {per} --push "
              f"--repo <user>/UDD --token $HF_TOKEN" + (f" --only {args.only}" if args.only else ""))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort
