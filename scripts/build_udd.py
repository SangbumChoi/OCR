#!/usr/bin/env python3
"""Build & upload the **UDD — Universal Document Dataset** to the Hub.

Per-dataset converter → uniform UDD schema → **safety-checked** → merged into **ONE sharded dataset**
(a single default config). The origin (source dataset name, split, hf_id, hf_config) is kept in
**columns**, not in the repo layout — so it's one dataset you filter by `source`/`task`, not a pile of
per-benchmark folders.

Memory/scale notes: each benchmark is streamed + converted **independently** (bounded by
``--per-bench``), safety-checked, saved to disk, then all sources are merged by **memory-mapped
concat** (never all in RAM) and pushed as one dataset sharded by ``--max-shard-size`` (resumable).

INCREMENTAL ADDS + DEDUP CACHE: per-source builds are cached on disk (``out/hf/<key>``) and a
persistent **image-hash index** (``out/hash_index.json``, md5 of the downscaled image → owner source)
is maintained across runs. ``--skip-existing`` skips re-streaming already-built sources, the index
skips images that already exist under a *different* source (COCO pages recur across scene-text sets),
and the merge always concatenates **everything on disk** — so adding one new dataset costs one
dataset: ``--only <newkey> --skip-existing`` streams just the newcomer and re-merges the full corpus.
The merge also runs the **enrichment pass** (language fill + n_fields/n_regions + image dims).

MOCKUP (default): ``--per-bench 10`` — ten examples per dataset, saved locally + safety-checked +
visualized. Add ``--push --repo <user>/UDD --token <hf_token> --public`` to upload.

    # local mockup (no upload): 10 examples/dataset, safety-check, visualize
    python scripts/build_udd.py --only cord,funsd,ocrvqa,docvqa --per-bench 10
    # upload the merged sharded dataset to your HF
    python scripts/build_udd.py --per-bench 10 --push --repo <user>/UDD --token $HF_TOKEN --public
    # INCREMENTAL: add one new benchmark to the existing corpus and re-push
    python scripts/build_udd.py --only <newkey> --per-bench 100 --skip-existing --push --repo <user>/UDD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import (UnifiedLoader, enrich_dataset, push,  # noqa: E402
                                 render_grid, safety_check, to_hf_dataset)

HARD_CAP = 1500     # was 500 (300/source era), 200 before that; raised for the 1000/source release


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
    p.add_argument("--skip-existing", action="store_true",
                   help="INCREMENTAL ADD: don't re-stream sources whose on-disk build already exists "
                        "(out/hf/<key>) — only new sources are converted; the merge still includes "
                        "everything on disk. Adding one dataset costs one dataset, not 21.")
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

    # persistent image-hash index: md5 -> owner source. Dedups ACROSS sources (COCO images recur in
    # scene-text sets) and across runs; a source's own hashes never block its rebuild.
    index_path = out / "hash_index.json"
    hash_index: dict[str, str] = json.loads(index_path.read_text()) if index_path.exists() else {}

    viz_rows = []
    report: dict[str, dict] = {}
    for k in keys:
        if args.skip_existing and (out / "hf" / k).exists():
            print(f"[cache] {k}: on-disk build exists, skipping stream (merged below)")
            continue
        rows = loader.load(k, limit=per, max_scan=args.max_scan, max_px=args.max_px,
                           cache_dir=str(out / "images"), global_index=hash_index)
        if not rows:
            print(f"[skip] {k}: no records"); continue
        # 1) safety-check the converter BEFORE trusting/uploading it
        try:
            rep = safety_check(rows, str(out / "hf" / k))
        except Exception as exc:
            print(f"[FAIL] {k}: safety check failed -> {type(exc).__name__}: {exc}"); continue
        report[k] = {"records": len(rows), **rep}
        print(f"[ok]   {k:14} {rep['rows']:4} rows  fields={rep['fields']} regions={rep['regions']} "
              f"image_ok={rep['image_ok']}  (split={rows[0].split})")
        viz_rows.append(rows[0])
        index_path.write_text(json.dumps(hash_index))     # checkpoint the dedup cache per source
        # each source is saved to disk (out/hf/<key>) for safety + memory-mapped concat below;
        # the origin (source/split/hf_config) lives in COLUMNS, so we don't need per-benchmark folders.

    # ONE merged, sharded dataset (default config) — origin kept in the `source`/`split` columns, not
    # in the repo layout. Built by memory-mapped concat of ALL on-disk per-source saves (not just this
    # run's — so an incremental `--only newkey` build still merges the full corpus), never all in RAM.
    # underscore-prefixed dirs are pipeline outputs (_all, plus _all_tmp if a merge crashed
    # mid-save) — never treat them as sources
    on_disk = sorted(d.name for d in (out / "hf").glob("*")
                     if d.is_dir() and not d.name.startswith("_")) if (out / "hf").exists() else []
    if on_disk and not args.no_combined:
        from datasets import concatenate_datasets, load_from_disk
        parts = [load_from_disk(str(out / "hf" / k)) for k in on_disk]
        combined = concatenate_datasets(parts)
        combined = enrich_dataset(combined)               # language + n_fields/n_regions + image dims
        from docvlm_eval.unified import dedupe_by_phash
        combined = dedupe_by_phash(combined)              # same phash+dims = same image: store once,
                                                          # QAs gathered into instructions/answers
        combined.save_to_disk(str(out / "hf" / "_all_tmp"))
        import shutil
        if (out / "hf" / "_all").exists():
            shutil.rmtree(out / "hf" / "_all")
        (out / "hf" / "_all_tmp").rename(out / "hf" / "_all")
        print(f"\n[ok] merged dataset: {len(combined)} rows from {len(on_disk)} sources "
              f"({len(report)} built now, {len(on_disk) - len(report)} reused) -> {out/'hf'/'_all'}")
        if args.push:
            # default config (no per-benchmark configs); datasets shards parquet by --max-shard-size
            combined.push_to_hub(args.repo, token=args.token, private=not args.public,
                                 max_shard_size=args.max_shard_size)
            print(f"     pushed merged dataset (sharded) -> {args.repo}")

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
