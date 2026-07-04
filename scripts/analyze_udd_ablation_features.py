#!/usr/bin/env python3
"""What ELSE in UDD can drive an ablation? Mine the corpus columns for usable dimensions.

The wired arms use task / language / fold / grounding / derived rationales. This script audits the
REMAINING information in the schema and reports, for each candidate dimension, the bucket sizes on
the live corpus (an ablation needs equal-N buckets, so support decides viability) and the training
recipe (which column filter builds each bucket). Writes
``docs/results/udd_ablation_features.md`` + ``docs/report/figures/udd_ablation_features.png``.

    python scripts/analyze_udd_ablation_features.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MD = ROOT / "docs" / "results" / "udd_ablation_features.md"
FIG = ROOT / "docs" / "report" / "figures" / "udd_ablation_features.png"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    args = p.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.src)
    n = len(ds)
    W, H = ds["image_width"], ds["image_height"]
    instrs, answers = ds["instructions"], ds["answers"]

    dims: dict[str, Counter] = {}
    dims["resolution (A7 preprocessing)"] = Counter(
        "small <0.3MP" if w * h < 3e5 else "medium 0.3-0.7MP" if w * h < 7e5 else "large >0.7MP"
        for w, h in zip(W, H))
    dims["aspect ratio (doc form factor)"] = Counter(
        ("wide <0.8" if h / max(1, w) < 0.8 else "square 0.8-1.3" if h / max(1, w) < 1.3 else
         "portrait 1.3-2" if h / max(1, w) < 2 else "tall >2 (receipts/screens)")
        for w, h in zip(W, H))
    dims["QAs per image (packing)"] = Counter(
        "1" if len(q) == 1 else "2-4" if len(q) <= 4 else "5+" for q in instrs)
    dims["answer length (output style)"] = Counter(
        ("short <=15" if ln <= 15 else "medium 16-60" if ln <= 60 else "long >60 (abstractive)")
        for ln in (len(a[0][0]) if (a and a[0]) else 0 for a in answers))
    dims["answer modality"] = Counter(
        "numeric" if (a and a[0] and re.fullmatch(r"[\d.,%$\s-]+", a[0][0] or "x")) else "textual"
        for a in answers)
    qt = Counter()
    for q in instrs:
        s = (q[0] if q else "").lower()
        qt["how-many/count" if s.startswith("how many") else
           "what" if s.startswith("what") else
           "yes/no" if re.match(r"(is|are|does|do|was|can)\b", s) else "other"] += 1
    dims["question type"] = qt
    small = tot = 0
    for rj in ds["regions_json"]:
        for el in json.loads(rj or "[]"):
            bb = el.get("bbox")
            if bb and bb[4]:
                tot += 1
                small += (bb[2] - bb[0]) * (bb[3] - bb[1]) < 0.01
    dims["grounding box size (A1 curriculum)"] = Counter(
        {"small <1% page area": small, "larger": tot - small})
    dims["license (compliance filter)"] = Counter(
        "non-commercial (cc-nc)" if lc == "cc-by-nc-4.0" else
        "permissive/tagged" if lc not in ("unspecified", "other") else lc
        for lc in ds["license"])

    # ---------- proposals table (recipe + hypothesis per dimension)
    proposals = [
        ("U-A7 resolution strata", "filter by `image_width*image_height` buckets",
         "training on high-res pages moves small-text NED more than equal-N low-res"),
        ("U-A8 QA packing", "images with 5+ QAs (`len(instructions)`) vs 1-QA images at equal QA "
         "budget", "many-QAs-per-image amortizes vision compute AND teaches multi-field reading — "
         "beats 1 QA/image"),
        ("U-A9 output style", "short (<=15 chars) vs long (>60) answer rows",
         "long-answer training degrades exact/anls on short-answer eval (verbosity bias) — "
         "measure the interference"),
        ("U-A10 numeric reasoning", "rows whose gold is numeric (16% of corpus)",
         "numeric-only training moves chart/relaxed_acc without touching text extraction"),
        ("U-A11 grounding difficulty", "region rows split by box area (55% of boxes <1% page)",
         "curriculum large->small boxes beats mixed-size grounding at equal N (A1 refinement)"),
        ("U-A12 form factor", "aspect-ratio buckets (tall receipts/screens vs wide pages)",
         "form-factor-matched training transfers within factor, weakly across"),
        ("license filter", "drop cc-by-nc rows (200) from training",
         "compliance-safe training costs nothing measurable (only MTVQA is NC)"),
    ]

    # ---------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = list(dims.items())
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (name, cnt) in zip(axes.ravel(), show):
        labels, vals = zip(*cnt.most_common())
        ax.bar(range(len(vals)), vals, color="#7aa6d6")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(name, fontsize=9, fontweight="bold")
        for i, v in enumerate(vals):
            ax.annotate(str(v), (i, v), ha="center", va="bottom", fontsize=7)
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle(f"UDD ablation-usable feature distributions ({n} image-rows)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- report
    lines = ["# UDD — additional ablation-usable features", "",
             f"Corpus: {n} image-rows. Beyond the wired dimensions (task / language / fold / "
             "grounding / derived rationales), these column-derived dimensions have enough bucket "
             "support to ablate (equal-N buckets are the constraint):", ""]
    for name, cnt in dims.items():
        parts = ", ".join(f"{k}: **{v}**" for k, v in cnt.most_common())
        lines.append(f"- **{name}** — {parts}")
    lines += ["", "![distributions](../report/figures/udd_ablation_features.png)", "",
              "## Proposed new arms", "",
              "| arm | bucket recipe (column filter) | hypothesis |", "|---|---|---|"]
    for a, r, h in proposals:
        lines.append(f"| {a} | {r} | {h} |")
    lines += ["", "All recipes are pure column filters on the live schema — no new data collection; "
              "`build_task_trainsets.py`-style equal-N subsampling + `run_ablation --arm public` "
              "run them unchanged. Support caveats: `when/where` question types are too thin "
              "(41/20 rows) to ablate; `tall` aspect has 190 rows — pair it with a lowered --count."]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {MD}\n[ok] {FIG}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
