#!/usr/bin/env python3
"""Enrich the built UDD: fill the empty ``language`` column + add queryable derived columns.

The converters only carry what each source ships, so ``language`` came out 0% filled and the
structured payload was only reachable through ``fields_json``/``regions_json`` decodes. This applies
:mod:`docvlm_eval.unified.enrich` to the merged on-disk dataset:

* ``language``   — Unicode-script heuristic over the row's own text, per-source prior for Latin
                   (CORD→id, formula sets→und, rest→en). Deterministic, offline.
* ``n_fields``   — #KIE fields (int column: filter KIE-with-payload without JSON decode)
* ``n_regions``  — #localization regions (int column: "rows with boxes" = ``n_regions > 0``)
* ``image_width``/``image_height`` — stored image dims (resolution slicing / curriculum)

Saves next to the input (``<src>_enriched``) then atomically replaces ``<src>`` so downstream paths
keep working. ``--push`` re-uploads to the Hub.

    python scripts/enrich_udd.py                                  # enrich data/udd/hf/_all in place
    python scripts/enrich_udd.py --push --repo danelcsb/UDD --token $HF_TOKEN --public
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import enrich_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo", default=None, help="target HF dataset repo, e.g. <user>/UDD")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--public", action="store_true")
    p.add_argument("--max-shard-size", default="500MB")
    args = p.parse_args()
    if args.push and not (args.repo and args.token):
        sys.exit("[enrich] --push needs --repo and a token (--token or $HF_TOKEN)")

    from datasets import load_from_disk

    src = Path(args.src)
    ds = load_from_disk(str(src))
    ds = enrich_dataset(ds)

    langs = Counter(ds["language"])
    boxed = sum(1 for n in ds["n_regions"] if n) + sum(1 for n in ds["n_fields"] if n)
    print(f"[enrich] {len(ds)} rows  language filled: {sum(c for l, c in langs.items() if l)}"
          f"/{len(ds)}  distribution={dict(langs.most_common())}")
    print(f"[enrich] rows with structured payload (n_fields|n_regions > 0): {boxed}")

    tmp = src.with_name(src.name + "_enriched")
    ds.save_to_disk(str(tmp))
    shutil.rmtree(src)
    tmp.rename(src)
    print(f"[ok] enriched dataset -> {src} (replaced in place)")

    if args.push:
        ds.push_to_hub(args.repo, token=args.token, private=not args.public,
                       max_shard_size=args.max_shard_size)
        print(f"[ok] pushed -> {args.repo}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort
