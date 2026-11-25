#!/usr/bin/env python
"""CLI for the layered document annotation (DAR) pipeline.

    # 1) start: one skeleton JSON per image (page size read from the file)
    python scripts/annotate_dar.py init data/annotations/mine/*.png --doc-type invoice
    # 2) annotate the JSONs (by hand / your tool of choice), then check them
    python scripts/annotate_dar.py validate data/annotations/mine
    python scripts/annotate_dar.py stats data/annotations/mine
    # 3) export a training mix for one ablation arm (+ the held-out end-task eval set)
    python scripts/annotate_dar.py arms
    python scripts/annotate_dar.py export data/annotations/mine --arm L1_+ocr --out data/dar_arms/L1_ocr
    # ...or any explicit layer set
    python scripts/annotate_dar.py export data/annotations/mine --layers kie qa ocr --reasoning chain

See docs/report/annotation_format.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.annotation import (
    ARMS,
    EXPORT_LAYERS,
    LAYERS,
    REASONING_STYLES,
    DocAnnotation,
    annotation_cost,
    build_arm,
    eval_samples,
    export_samples,
    skeleton,
    validate,
    write_jsonl,
)
from docvlm_eval.annotation.validate import ERROR


def _json_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    return out


def _load(paths: list[str]) -> list[tuple[Path, DocAnnotation]]:
    recs = []
    for f in _json_files(paths):
        try:
            recs.append((f, DocAnnotation.from_json(f.read_text(encoding="utf-8"))))
        except (KeyError, TypeError, ValueError) as e:
            print(f"[skip] {f}: not a DAR record ({e})", file=sys.stderr)
    return recs


def cmd_init(a) -> int:
    from PIL import Image
    n = 0
    for img in map(Path, a.images):
        dst = img.with_suffix(".json")
        if dst.exists() and not a.force:
            print(f"[keep] {dst} exists (use --force to overwrite)")
            continue
        with Image.open(img) as im:
            W, H = im.size
        ann = skeleton(f"{a.prefix}{img.stem}", img.name, W, H, doc_type=a.doc_type,
                       domain=a.domain, acquisition=a.acquisition,
                       languages=a.languages or [])
        ann.provenance.source = a.source
        ann.provenance.license = a.license
        dst.write_text(ann.to_json() + "\n", encoding="utf-8")
        n += 1
    print(f"wrote {n} skeleton(s)")
    return 0


def cmd_validate(a) -> int:
    bad = 0
    for f, ann in _load(a.paths):
        issues = validate(ann)
        errs = [i for i in issues if i.level == ERROR]
        bad += bool(errs)
        if issues:
            print(f"{f}: {len(errs)} error(s), {len(issues) - len(errs)} warning(s)")
            for i in issues:
                print("   ", i)
        elif a.verbose:
            print(f"{f}: ok")
    print(f"{'FAIL' if bad else 'OK'}: {bad} record(s) with errors")
    return 1 if bad else 0


def cmd_stats(a) -> int:
    recs = [ann for _, ann in _load(a.paths)]
    status = {name: Counter(r.layer_status(name) for r in recs) for name in LAYERS}
    report = {
        "n_records": len(recs),
        "split": Counter(r.split for r in recs),
        "doc_type": Counter(r.page.doc_type or "?" for r in recs),
        "orientation": Counter(r.page.orientation for r in recs),
        "layer_status": status,
        "counts": {
            "regions": sum(len(r.layout) for r in recs), "lines": sum(len(r.ocr) for r in recs),
            "tables": sum(len(r.tables) for r in recs), "kie_fields": sum(len(r.kie) for r in recs),
            "kie_absent": sum(1 for r in recs for f in r.kie if not f.present),
            "guide_notes": sum(len(r.understanding.guide) for r in recs),
            "qa": sum(len(r.understanding.qa) for r in recs),
            "qa_with_steps": sum(1 for r in recs for q in r.understanding.qa if q.steps),
        },
        "qa_type": Counter(q.qa_type for r in recs for q in r.understanding.qa),
        "annotation_hours": {k: round(v, 3) for k, v in annotation_cost(recs).items()},
    }
    print(json.dumps(report, indent=1, default=dict))
    return 0


def cmd_arms(a) -> int:
    for name, arm in ARMS.items():
        print(f"{name:24s} layers={','.join(arm.layers):60s} qa={arm.reasoning:12s} "
              f"rotate={arm.rotate}")
    return 0


def cmd_export(a) -> int:
    recs = _load(a.paths)
    bad = [(f, i) for f, ann in recs for i in validate(ann) if i.level == ERROR]
    if bad and not a.allow_invalid:
        for f, i in bad:
            print(f"{f}: {i}", file=sys.stderr)
        print("refusing to export records with errors (--allow-invalid to override)",
              file=sys.stderr)
        return 1
    out = Path(a.out)
    train, heldout = [], []
    for f, ann in recs:
        if a.arm:
            train += build_arm([ann], a.arm, base_dir=f.parent, rot_dir=out / "rotated",
                               include_pseudo=not a.human_only)
            heldout += eval_samples([ann], a.arm, base_dir=f.parent)
        else:
            if ann.split == "train":
                train += export_samples(ann, a.layers, reasoning=a.reasoning, base_dir=f.parent,
                                        include_pseudo=not a.human_only)
            else:
                heldout += export_samples(ann, ("kie", "qa"), reasoning=a.reasoning,
                                          base_dir=f.parent)
    n_tr = write_jsonl(train, out / "train.jsonl")
    n_ho = write_jsonl(heldout, out / "heldout.jsonl")
    by_layer = Counter(s.meta["layer"] for s in train)
    print(f"train: {n_tr} samples {dict(by_layer)} -> {out / 'train.jsonl'}")
    print(f"heldout (end task): {n_ho} samples -> {out / 'heldout.jsonl'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write a skeleton JSON next to each image")
    p.add_argument("images", nargs="+")
    p.add_argument("--prefix", default="", help="doc_id prefix, e.g. 'mine/'")
    p.add_argument("--doc-type", default="")
    p.add_argument("--domain", default="")
    p.add_argument("--acquisition", default="unknown")
    p.add_argument("--languages", nargs="*")
    p.add_argument("--source", default="")
    p.add_argument("--license", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("validate", help="check ids, boxes, evidence and layer status")
    p.add_argument("paths", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("stats", help="corpus summary: layer coverage, counts, annotation hours")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("arms", help="list the layer-ablation arms")
    p.set_defaults(fn=cmd_arms)

    p = sub.add_parser("export", help="write train.jsonl / heldout.jsonl (Sample format)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--arm", choices=list(ARMS))
    g.add_argument("--layers", nargs="+", choices=EXPORT_LAYERS, default=list(EXPORT_LAYERS))
    p.add_argument("--reasoning", choices=REASONING_STYLES, default="answer",
                   help="QA target style when --layers is used (an --arm sets its own)")
    p.add_argument("--human-only", action="store_true", help="drop pseudo-labelled layers")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(fn=cmd_export)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
