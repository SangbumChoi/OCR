#!/usr/bin/env python3
"""Visualise the synthetic ground truth for every realistic case.

For each case it draws the GT boxes on the clean render:
  * green  = field / spotting boxes (OCR-KIE localisation)
  * red    = L1-locate  (understanding: "where is this word?")
  * blue   = L1-region  (understanding: "where is this table/region?")
and labels each case with the understanding QAs it carries. Writes a montage to
docs/report/figures/synthetic_gt_overlays.png (+ per-case PNGs alongside).

    python scripts/visualize_synth_gt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "probes" / "realistic_cases"
FIG = ROOT / "docs" / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _boxes(gt):
    """Yield (box, color, label) for every GT box in a case."""
    for f in gt.get("fields_detailed", []):
        if f.get("bbox"):
            yield f["bbox"], "#1a9641", f["key"]
    for q in gt.get("qa_detailed", []):
        if q.get("answer_bbox"):
            if q["answer_type"] == "L1-locate":
                yield q["answer_bbox"], "#d7191c", "locate"
            elif q["answer_type"] == "L1-region":
                yield q["answer_bbox"], "#2c7bb6", "region"


def _draw(ax, key, gt, img):
    ax.imshow(img); ax.set_axis_off()
    for b, color, label in _boxes(gt):
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                     fill=False, edgecolor=color, linewidth=1.6))
        ax.text(b[0], max(0, b[1]-3), label, color=color, fontsize=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.4))
    und = [q["answer_type"].replace("L1-", "").replace("H-", "").replace("H1-", "")
           for q in gt.get("qa_detailed", []) if q["answer_type"].startswith(("L1-", "H-", "H1-"))]
    ax.set_title(f"{key}\n{gt.get('type','')}  ·  GT: {', '.join(und) or '—'}", fontsize=8)


def main():
    cases = sorted(p.name for p in CASES.iterdir() if p.is_dir())
    cols = 4
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.3 * rows))
    for ax, key in zip(axes.ravel(), cases):
        gt = json.loads((CASES / key / "gt.json").read_text())
        img = Image.open(CASES / key / "clean.png").convert("RGB")
        _draw(ax, key, gt, img)
        # also write a standalone per-case overlay
        f1, a1 = plt.subplots(figsize=(7, 7 * img.height / img.width))
        _draw(a1, key, gt, img); f1.tight_layout()
        f1.savefig(FIG / f"synth_gt_{key}.png", dpi=90); plt.close(f1)
    for ax in axes.ravel()[len(cases):]:
        ax.set_axis_off()
    fig.suptitle("Synthetic GT overlays — green=field/spot · red=locate · blue=region", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = FIG / "synthetic_gt_overlays.png"
    fig.savefig(out, dpi=95); plt.close(fig)
    print(f"[done] montage -> {out}  (+ per-case synth_gt_<key>.png in {FIG})")
    for key in cases:
        gt = json.loads((CASES / key / "gt.json").read_text())
        nb = sum(1 for _ in _boxes(gt))
        print(f"   {key:16} boxes={nb}")


if __name__ == "__main__":
    sys.exit(main())
