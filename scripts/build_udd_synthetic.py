#!/usr/bin/env python3
"""Build a LOCAL-ONLY UDD-format dataset from the synthetic generator's output.

Walks a synthetic case tree (``<root>/<key>/[NNNN/]gt.json`` + ``clean.png``/``degraded.png``, as
written by ``scripts/make_realistic_cases.py``), converts every case through the DTO bridge
(``docvlm_eval.unified.synth_bridge``) into UnifiedSamples carrying ALL annotations (QAs, KIE
boxes, grounding regions, rationale-as-reasoning-QA, table HTML), then runs the SAME
``safety_check`` (schema + payload-DTO + QA-pairing validation) the public sources go through and
saves to disk.

**Never uploaded**: synthetic data is regenerable and not public benchmark data — this script has
no push path on purpose; the output lands under ``data/udd_synthetic/`` (git-ignored) and can be
merged into training mixes locally.

    python scripts/build_udd_synthetic.py                          # from data/probes/realistic_cases
    python scripts/build_udd_synthetic.py --root <cases> --variant degraded
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified import docsample_to_unified, safety_check  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "data" / "probes" / "realistic_cases"))
    p.add_argument("--out", default=str(ROOT / "data" / "udd_synthetic"))
    p.add_argument("--variant", choices=["clean", "degraded"], default="clean")
    p.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    args = p.parse_args()

    root = Path(args.root)
    gt_paths = sorted(root.rglob("gt.json"))
    if args.limit:
        gt_paths = gt_paths[: args.limit]
    if not gt_paths:
        sys.exit(f"[udd-synth] no gt.json under {root} — generate cases first "
                 "(python scripts/make_realistic_cases.py)")

    from PIL import Image
    rows, skipped = [], 0
    for i, gp in enumerate(gt_paths):
        case = gp.parent
        img = case / f"{args.variant}.png"
        if not img.exists():
            img = case / "clean.png"
        if not img.exists():
            skipped += 1
            continue
        try:
            with Image.open(img) as im:
                size = im.size
            rec = docsample_to_unified(json.loads(gp.read_text(encoding="utf-8")),
                                       image_path=str(img), image_size=size,
                                       sample_id=f"synthetic_{case.parent.name}_{case.name}_{i}")
            rows.append(rec)
        except ValueError:
            skipped += 1

    if not rows:
        sys.exit("[udd-synth] nothing trainable found")
    out = Path(args.out)
    rep = safety_check(rows, str(out / "hf"))   # SAME validation the public sources pass
    n_qas = sum(max(1, len(r.qas)) for r in rows)
    print(f"[ok] synthetic UDD: {rep['rows']} image-rows / {n_qas} QAs  fields={rep['fields']} "
          f"regions={rep['regions']}  (skipped {skipped})")
    print(f"     LOCAL ONLY -> {out/'hf'}  (load_from_disk; no push path by design)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort
