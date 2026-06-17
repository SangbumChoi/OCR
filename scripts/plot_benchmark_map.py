#!/usr/bin/env python3
"""Visualise the benchmark landscape:
  1. report/figures/benchmark_class_matrix.png - what VISUAL CLASSES (beyond text) each
     benchmark probes (charts, tables, formulas, diagrams, handwriting, seals, ...).
  2. report/figures/benchmark_priority.png - benchmarks grouped by evaluation "nature" into
     priority tiers, with prerequisite arrows (recognition -> VQA -> reasoning; structure;
     reliability), sized by how universally model papers report them.

    python scripts/plot_benchmark_map.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- class matrix
CLASSES = [
    "Printed text", "Handwriting", "Scene/natural", "Chart/plot", "Table",
    "Formula", "Diagram", "Infographic", "Sci-figure", "Web/UI", "Seal/stamp",
    "Icon/symbol", "Book cover",
]
# benchmark -> classes present (1)
MATRIX = {
    "docvqa":      ["Printed text", "Table"],
    "infovqa":     ["Printed text", "Chart/plot", "Infographic", "Icon/symbol"],
    "textvqa":     ["Printed text", "Scene/natural"],
    "stvqa":       ["Scene/natural"],
    "ocrvqa":      ["Printed text", "Book cover"],
    "ai2d":        ["Diagram", "Icon/symbol"],
    "chartqa":     ["Chart/plot"],
    "mathvista":   ["Chart/plot", "Diagram", "Formula", "Sci-figure"],
    "iam":         ["Handwriting"],
    "latexocr":    ["Formula"],
    "im2latex":    ["Formula"],
    "sroie":       ["Printed text"],
    "cord":        ["Printed text", "Table"],
    "funsd":       ["Printed text", "Table"],
    "pubtabnet":   ["Table"],
    "pubtables1m": ["Table"],
    "ocrbench":    ["Printed text", "Handwriting", "Scene/natural", "Formula"],
    "ocrbench_v2": ["Printed text", "Handwriting", "Scene/natural", "Formula", "Table", "Diagram"],
    "charxiv":     ["Chart/plot", "Sci-figure"],
    "omnidocbench":["Printed text", "Table", "Formula", "Handwriting"],
    "doclaynet":   ["Printed text", "Table", "Diagram"],
    "scenetext":   ["Scene/natural"],
    "recognition_fullpage": ["Printed text"],
    "pope":        ["Scene/natural"],
    "hallusionbench": ["Scene/natural", "Chart/plot", "Diagram"],
    "robustness":  ["Printed text", "Table"],
}


def plot_class_matrix() -> None:
    rows = list(MATRIX)
    data = [[1 if c in MATRIX[r] else 0 for c in CLASSES] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1.4)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(CLASSES)):
            if data[i][j]:
                ax.text(j, i, "●", ha="center", va="center", color="#1a4a8a", fontsize=9)
    ax.set_title("Visual classes probed by each benchmark (beyond plain text)", fontsize=13, pad=12)
    ax.set_xticks([x - 0.5 for x in range(len(CLASSES) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(rows) + 1)], minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "benchmark_class_matrix.png", dpi=130)
    plt.close(fig)
    print(f"[ok] {FIG / 'benchmark_class_matrix.png'}")


# ---------------------------------------------------------------- priority graph
# tier -> (label, color, [(benchmark, importance 1-3)])
TIERS = [
    ("Tier 1 · Core doc-VLM suite\n(reported by ~all VLM papers)", "#cfe3ff",
     [("DocVQA", 3), ("InfoVQA", 3), ("ChartQA", 3), ("OCRBench", 3), ("TextVQA", 2), ("AI2D", 2)]),
    ("Tier 2 · Parsing & structure\n(OCR-engine comparison)", "#d7f0d7",
     [("OmniDocBench", 3), ("PubTabNet/TEDS", 2), ("im2latex/LaTeX-OCR", 2), ("DocLayNet", 1)]),
    ("Tier 3 · Key-Info Extraction\n(production field value)", "#ffe6c7",
     [("FUNSD", 2), ("CORD", 2), ("SROIE", 2), ("DocILE/XFUND", 1)]),
    ("Tier 4 · Reliability\n(deployment gating)", "#f6d6e0",
     [("Robustness+ECE", 2), ("POPE", 1), ("HallusionBench", 1)]),
]


def plot_priority() -> None:
    fig, ax = plt.subplots(figsize=(18, 8.2))
    ax.axis("off")
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 9)
    y0 = 8.1
    row_h = 1.85
    box_centers = {}
    for ti, (label, color, items) in enumerate(TIERS):
        y = y0 - ti * row_h
        ax.text(0.15, y + 0.15, label, fontsize=10.5, fontweight="bold", va="center")
        x = 3.7
        for name, imp in items:
            w = 1.6 + 0.2 * imp
            box = FancyBboxPatch(
                (x, y - 0.32), w, 0.64, boxstyle="round,pad=0.04",
                linewidth=1.1, edgecolor="#33506e", facecolor=color,
            )
            ax.add_patch(box)
            fs = 8 + imp  # bigger = more universally reported
            ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=fs)
            box_centers[name] = (x + w / 2, y)
            x += w + 0.3

    # prerequisite / flow arrows between tiers (recognition -> VQA -> reasoning ...)
    def arrow(a, b, txt=""):
        xa, ya = box_centers[a]
        xb, yb = box_centers[b]
        ax.add_patch(FancyArrowPatch((xa, ya - 0.34), (xb, yb + 0.34),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#7a7a7a", linewidth=1.1, linestyle="--"))

    arrow("OCRBench", "OmniDocBench")     # recognition underpins parsing
    arrow("DocVQA", "FUNSD")              # doc QA -> field extraction
    arrow("OmniDocBench", "SROIE")        # parsing -> KIE
    arrow("DocVQA", "Robustness+ECE")     # accuracy -> reliability overlay

    ax.text(9.25, 8.85, "Benchmark priority & grouping  (box size ∝ how universally model papers report it; "
            "dashed = prerequisite flow)", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(0.15, 0.25, "Reading: recognition fidelity (OCRBench/transcription) is a prerequisite for "
            "document VQA (ANLS), which feeds field extraction (KIE F1); structure (TEDS) runs parallel; "
            "reliability overlays everything.", fontsize=8.5, color="#444", wrap=True)
    fig.savefig(FIG / "benchmark_priority.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {FIG / 'benchmark_priority.png'}")


if __name__ == "__main__":
    plot_class_matrix()
    plot_priority()
