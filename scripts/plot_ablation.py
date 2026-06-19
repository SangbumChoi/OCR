#!/usr/bin/env python3
"""Visualise the Part-2 ablation study (see docs/report/part2_ablation_plan.md):
  1. ablation_staircase.png     - cumulative gain as winners are stacked (the "staircase")
  2. ablation_deltas.png        - marginal Δ per ablation (which factor helps how much)
  3. ablation_lang_transfer.png - language-pair transfer heatmap (A4)
  4. ablation_lora_placement.png- capability gain by LoRA target group (A5)
  5. ablation_relationship.png  - how the ablations compose (dependency diagram)

Reads docs/results/ablation_results.json (a committed DEMO renders the format; replace with real
ablation numbers, same schema). `python scripts/plot_ablation.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "docs" / "results" / "ablation_results.json"


def _demo_tag(d):
    return "  [DEMO — illustrative]" if d.get("demo") else ""


def staircase(d):
    steps = d["staircase"]
    x = list(range(len(steps)))
    y = [s["score"] for s in steps]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.step(x, y, where="post", color="#1f5fa8", linewidth=2)
    ax.plot(x, y, "o", color="#1f5fa8")
    for i, s in enumerate(steps):
        ax.annotate(f"{s['score']:.1f}", (x[i], y[i]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
        if i > 0:
            dlt = y[i] - y[i - 1]
            ax.annotate(f"+{dlt:.1f}", ((x[i] + x[i-1]) / 2, (y[i] + y[i-1]) / 2),
                        color="#2a8a2a", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([s["step"] for s in steps], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(d.get("metric", "score"))
    ax.set_title("Cumulative ablation staircase" + _demo_tag(d), fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "ablation_staircase.png", dpi=130); plt.close(fig)
    print("[ok] ablation_staircase.png")


def deltas(d):
    items = d["deltas"]
    labels = [f"{i['ablation']}\n({i['axis']})" for i in items]
    vals = [i["delta"] for i in items]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2a8a2a" if v >= 0 else "#b03030" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_ylabel("Δ vs control")
    ax.set_title("Marginal gain per ablation (which factor helps)" + _demo_tag(d), fontsize=13)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "ablation_deltas.png", dpi=130); plt.close(fig)
    print("[ok] ablation_deltas.png")


def lang_transfer(d):
    lt = d.get("language_transfer")
    if not lt:
        return
    langs, M = lt["langs"], lt["matrix"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-3, vmax=3)
    ax.set_xticks(range(len(langs))); ax.set_xticklabels(langs)
    ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs)
    ax.set_xlabel("added language L2"); ax.set_ylabel("target language L1")
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, f"{M[i][j]:+.1f}", ha="center", va="center", fontsize=9)
    ax.set_title("Language transfer: score(L1 | train L1+L2) − score(L1 | L1)" + _demo_tag(d), fontsize=11)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "ablation_lang_transfer.png", dpi=130); plt.close(fig)
    print("[ok] ablation_lang_transfer.png")


def lora_placement(d):
    lp = d.get("lora_placement")
    if not lp:
        return
    groups = lp["groups"]
    caps = [k for k in ("grounding", "reasoning", "language") if k in lp]
    x = range(len(groups))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, cap in enumerate(caps):
        ax.bar([i + (k - 1) * w for i in x], lp[cap], width=w, label=cap)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylabel("capability Δ"); ax.legend()
    ax.set_title("LoRA placement: which modules move which capability" + _demo_tag(d), fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "ablation_lora_placement.png", dpi=130); plt.close(fig)
    print("[ok] ablation_lora_placement.png")


def relationship():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    facecolor=c, edgecolor="#33506e", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    def arr(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color="#777", linewidth=1.1))
    box(0.3, 2.6, 2.0, 0.8, "A7 preprocessing\n(resize/tiling)", "#cfe3ff")
    box(2.9, 3.6, 1.9, 0.8, "A1 spotting", "#d7f0d7")
    box(2.9, 1.6, 1.9, 0.8, "A2 reasoning", "#d7f0d7")
    box(5.3, 2.6, 1.9, 0.8, "composite\nmodel", "#ffe6c7")
    box(5.3, 4.7, 1.9, 0.8, "A4 multilingual\n(orthogonal data)", "#f6e0a8")
    box(8.0, 3.6, 1.8, 0.8, "A5 placement", "#f6d6e0")
    box(8.0, 1.6, 1.8, 0.8, "A6 HPO", "#f6d6e0")
    box(10.2, 2.6, 1.6, 0.8, "final model", "#cfe3ff")
    arr((2.3, 3.0), (2.9, 3.9)); arr((2.3, 3.0), (2.9, 2.0))
    arr((4.8, 4.0), (5.3, 3.2)); arr((4.8, 2.0), (5.3, 2.8))
    arr((6.2, 4.7), (6.2, 3.4))
    arr((7.2, 3.0), (8.0, 3.9)); arr((7.2, 3.0), (8.0, 2.0))
    arr((9.8, 3.9), (10.2, 3.2)); arr((9.8, 2.0), (10.2, 2.8))
    ax.text(6, 5.8, "Ablation relationship: preprocessing gates recognition; spotting+reasoning are "
            "parallel signals; placement/HPO are how to train; multilingual is orthogonal data",
            ha="center", fontsize=10, fontweight="bold")
    fig.savefig(FIG / "ablation_relationship.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[ok] ablation_relationship.png")


def main():
    d = json.loads(DATA.read_text())
    staircase(d); deltas(d); lang_transfer(d); lora_placement(d); relationship()
    print(f"[done] figures -> {FIG}")


if __name__ == "__main__":
    main()
