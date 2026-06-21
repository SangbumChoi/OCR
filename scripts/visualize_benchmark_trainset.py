#!/usr/bin/env python3
"""Visualize the benchmark training set + its ground truth as an image grid.

Reads the merged ``train.jsonl`` produced by ``scripts/build_benchmark_trainset.py`` and renders a
montage: one (or more) cell per benchmark showing the cached image with its normalised
question / answer / metric overlaid — a quick eyeball check that each benchmark mapped into our DTO
correctly. Writes a PNG under ``docs/report/figures/`` (where the other report figures live).

    python scripts/visualize_benchmark_trainset.py --per-bench 1
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _wrap(s: str, width: int, max_lines: int) -> str:
    s = " ".join(str(s).split())
    lines = textwrap.wrap(s, width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", default=str(ROOT / "data" / "benchmark_trainset" / "train.jsonl"))
    p.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" /
                                        "benchmark_trainset_preview.png"))
    p.add_argument("--per-bench", type=int, default=1, help="cells to show per benchmark")
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if not rows:
        print(f"No rows in {args.jsonl} — run scripts/build_benchmark_trainset.py first."); return

    # pick up to --per-bench rows per benchmark, preserving first-seen order
    picked: list[dict] = []
    per: dict[str, int] = {}
    for r in rows:
        b = r["meta"]["benchmark"]
        if per.get(b, 0) >= args.per_bench:
            continue
        per[b] = per.get(b, 0) + 1
        picked.append(r)

    n = len(picked)
    cols = max(1, args.cols)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.6, rows_n * 4.2))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, r in zip(axes, picked):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception as exc:
            ax.text(0.5, 0.5, f"<no image>\n{type(exc).__name__}", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        b = r["meta"]["benchmark"]
        ax.set_title(f"{b}  [{r['answer_type']} · {r['metric']}]", fontsize=8, fontweight="bold")
        q = _wrap("Q: " + r["question"], width=46, max_lines=3)
        a = _wrap("A: " + " | ".join(r["answers"]), width=46, max_lines=2)
        ax.set_xlabel(q + "\n" + a, fontsize=7, loc="left")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Benchmark training set — {n} samples, {len(per)} benchmarks "
                 f"(image + normalised GT)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out}  ({n} cells, {len(per)} benchmarks)")


if __name__ == "__main__":
    main()
