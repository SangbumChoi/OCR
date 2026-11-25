#!/usr/bin/env python
"""Build the public-benchmark leaderboard page and charts from docs/results/public_leaderboard.yaml.

    python scripts/build_public_leaderboard.py

Writes docs/report/public_leaderboard.md and docs/report/figures/public_leaderboard/*.png. Never
edit those by hand: change the YAML and re-run.

Before writing anything it re-derives each Overall that has a published formula from its own
columns, so a transcription slip in the YAML fails the build instead of reaching the page.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "public_leaderboard.yaml"
PAGE = ROOT / "docs" / "report" / "public_leaderboard.md"
FIG = ROOT / "docs" / "report" / "figures" / "public_leaderboard"

# Chart palette (validated with the dataviz skill's validate_palette.js, light surface):
# categorical slots 1-2 for the two license groups, a one-hue blue ramp for ranks.
PERMISSIVE, OTHER = "#257de0", "#f47230"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
RAMP = ["#0e3d7e", "#1757ac", "#2370cd", "#338ced", "#6aabf1", "#9dc8f7", "#cce3fd"]  # best -> worst
MARK = "‡"


# --------------------------------------------------------------------------- consistency checks
def _overall_checks(bench: dict) -> list[str]:
    """Re-derive Overall where the benchmark defines it from the shown columns."""
    keys = [c["key"] for c in bench["columns"]]
    problems = []
    for row in bench["rows"]:
        s = dict(zip(keys, row["scores"]))
        if s.get("overall") is None:
            continue
        expect = None
        if bench["id"] in ("omnidocbench_v16", "omnidocbench_v15"):
            expect = ((1 - s["text_edit"]) * 100 + s["formula_cdm"] + s["table_teds"]) / 3
        elif bench["id"] == "real5_omnidocbench":
            expect = sum(s[k] for k in ("scanning", "warping", "screen_photo", "illumination", "skew")) / 5
        elif bench["id"] == "mdpbench_ko":
            expect = (40 * s["digital"] + 120 * s["photo"]) / 160
        elif bench["id"] == "kdoc_ocrbench_v2":
            expect = sum(s[k] for k in ("baseline", "no_header_footer", "body", "tables")) / 4
        if expect is not None and abs(expect - s["overall"]) > 0.16:
            problems.append(f"{bench['id']}: {row['model']} Overall {s['overall']} but columns give {expect:.2f}")
    return problems


# MDPBench's Overall is a page mean; a model whose pages are not all scorable can differ from the
# 40/120 digital/photographed weighting by more than rounding. Reported, not fatal.
SOFT_CHECKS = {"mdpbench_ko"}


def validate(data: dict) -> None:
    problems = []
    models = data["models"]
    for b in data["benchmarks"]:
        n = len(b["columns"])
        seen = set()
        for r in b["rows"]:
            if r["model"] not in models:
                problems.append(f"{b['id']}: unknown model {r['model']!r}")
            if r["model"] in seen:
                problems.append(f"{b['id']}: duplicate row {r['model']!r}")
            seen.add(r["model"])
            if len(r["scores"]) != n:
                problems.append(f"{b['id']}: {r['model']} has {len(r['scores'])} scores for {n} columns")
        found = _overall_checks(b)
        if b["id"] in SOFT_CHECKS:
            for w in found:
                print(f"[warn] {w} (page-mean benchmark; kept as published)")
        else:
            problems += found
    if problems:
        sys.exit("public_leaderboard.yaml failed validation:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- helpers
def overall(row: dict, bench: dict):
    keys = [c["key"] for c in bench["columns"]]
    return dict(zip(keys, row["scores"])).get("overall")


def sorted_rows(bench: dict) -> list[dict]:
    """Overall descending; rows without an Overall last (in source order)."""
    with_o = [r for r in bench["rows"] if overall(r, bench) is not None]
    without = [r for r in bench["rows"] if overall(r, bench) is None]
    return sorted(with_o, key=lambda r: -overall(r, bench)) + without


def label(model: str, data: dict) -> str:
    return model + ("" if data["models"][model]["permissive"] else MARK)


def column_formatter(bench: dict):
    """Per column: 3 decimals for 0-1 scales (Edit, KoOCR), else 1 decimal."""
    out = []
    for j, _ in enumerate(bench["columns"]):
        vals = [r["scores"][j] for r in bench["rows"] if r["scores"][j] is not None]
        unit = all(v <= 1.0 for v in vals)
        out.append((lambda v, u=unit: "–" if v is None else (f"{v:.3f}" if u else f"{v:.1f}")))
    return out


# --------------------------------------------------------------------------- figures
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                         "text.color": INK, "savefig.dpi": 160})
    return plt


def dot_chart(bench: dict, data: dict, path: Path) -> None:
    """Ranked Cleveland dot plot of Overall. Dots (not bars) because the scores cluster far from
    zero: a truncated bar axis would exaggerate the gaps, a zero-based one would hide them."""
    plt = _setup_mpl()
    rows = [r for r in sorted_rows(bench) if overall(r, bench) is not None]
    vals = [overall(r, bench) for r in rows]
    names = [label(r["model"], data) for r in rows]
    colors = [PERMISSIVE if data["models"][r["model"]]["permissive"] else OTHER for r in rows]
    unit = max(vals) <= 1.0
    step = 0.05 if unit else 5.0
    lo = math.floor((min(vals) - step * 0.3) / step) * step
    hi = math.ceil((max(vals) + step * 0.3) / step) * step
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(rows) + 1.25))
    y = list(range(len(rows)))[::-1]
    ax.hlines(y, lo, hi, color=GRID, linewidth=0.8, zorder=1)
    ax.scatter(vals, y, s=64, c=colors, edgecolors="white", linewidths=1.5, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + (hi - lo) * 0.012, yi + 0.02, f"{v:.3f}" if unit else f"{v:.1f}", va="center",
                ha="left", fontsize=8.5, color=INK2, zorder=4)
    ax.set_xlim(lo, hi + (hi - lo) * 0.06)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Overall (higher is better; axis does not start at zero)")
    ax.set_title(bench["title"], loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, bench["pages"], transform=ax.transAxes, fontsize=9, color=INK2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=PERMISSIVE, label="Apache-2.0 / MIT / BSD"),
                       Line2D([], [], marker="o", ls="", ms=8, color=OTHER, label=f"{MARK} other license")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rank_heatmap(data: dict, path: Path) -> list[tuple[str, float, int]]:
    """Models x benchmarks, colored by rank within each benchmark (1 = best). Returns the order."""
    plt = _setup_mpl()
    benches = data["benchmarks"]
    ranks: dict[str, dict[str, tuple[int, float]]] = {m: {} for m in data["models"]}
    for b in benches:
        for i, r in enumerate([r for r in sorted_rows(b) if overall(r, b) is not None], 1):
            ranks[r["model"]][b["id"]] = (i, overall(r, b))
    order = []
    for m, d in ranks.items():
        if d:
            order.append((m, sum(i for i, _ in d.values()) / len(d), len(d)))
    order.sort(key=lambda t: (t[1], -t[2]))
    n_models = len(order)
    fig, ax = plt.subplots(figsize=(10.5, 0.34 * n_models + 1.9))
    for yi, (m, _, _) in enumerate(order):
        for xi, b in enumerate(benches):
            cell = ranks[m].get(b["id"])
            if cell is None:
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor="#f4f4f4", edgecolor="white", lw=2))
                ax.text(xi + 0.5, yi + 0.5, "–", ha="center", va="center", fontsize=8, color=INK2)
                continue
            rank, val = cell
            k = min(len(RAMP) - 1, int((rank - 1) / max(1, n_models - 1) * len(RAMP)))
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=RAMP[k], edgecolor="white", lw=2))
            txt = f"#{rank}  {val:.3f}" if val <= 1.0 else f"#{rank}  {val:.1f}"
            ax.text(xi + 0.5, yi + 0.5, txt, ha="center", va="center", fontsize=8,
                    color="white" if k <= 3 else INK)
    ax.set_xlim(0, len(benches))
    ax.set_ylim(n_models, 0)
    ax.set_xticks([i + 0.5 for i in range(len(benches))],
                  [b["title"].replace(" (Korean)", "\n(Korean)").replace(", Korean", "\n(Korean)")
                   for b in benches], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(n_models)],
                  [f"{label(m, data)}  (mean rank {r:.1f})" for m, r, _ in order], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Rank by Overall on each public benchmark (1 = best; darker = higher rank)",
                 loc="left", fontsize=11, fontweight="bold", pad=44)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return order


# --------------------------------------------------------------------------- page
def table_md(bench: dict, data: dict) -> list[str]:
    cols = bench["columns"]
    fmts = column_formatter(bench)
    rows = sorted_rows(bench)
    best = []
    for j, c in enumerate(cols):
        vals = [r["scores"][j] for r in rows if r["scores"][j] is not None]
        best.append((min if c["better"] == "low" else max)(vals) if vals else None)
    head = ["Model", "Size (B)", "License"] + [c["label"] + (" ↓" if c["better"] == "low" else "") for c in cols]
    out = ["| " + " | ".join(head) + " |", "| --- | ---: | --- | " + " | ".join("---:" for _ in cols) + " |"]
    for r in rows:
        m = data["models"][r["model"]]
        cells = []
        for j, v in enumerate(r["scores"]):
            s = fmts[j](v)
            cells.append(f"**{s}**" if v is not None and math.isclose(v, best[j]) else s)
        out.append(f"| {label(r['model'], data)} | {m['size_b']} | {m['license']} | " + " | ".join(cells) + " |")
    notes = []
    for r in rows:
        parts = []
        if r.get("reported"):
            parts.append(f"reported {r['reported']}")
        if r.get("note"):
            parts.append(r["note"])
        if parts:
            notes.append(f"- {label(r['model'], data)}: " + " · ".join(parts))
    if notes:
        out += ["", "Notes:", ""] + notes
    return out


def build(data: dict) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    order = rank_heatmap(data, FIG / "rank_heatmap.png")
    L = [("<!-- GENERATED by scripts/build_public_leaderboard.py from docs/results/public_leaderboard.yaml."
         " Do not edit by hand. -->"), "",
         "# Public OCR and document-parsing benchmarks",
         "",
         (f"Publicly released models on six public benchmarks, each scored with the benchmark's official "
         f"scorer (run {data['as_of']}). Three benchmarks are English/multilingual page parsing "
         "(OmniDocBench v1.6, its v1.5 baseline, and the Real5 capture conditions); three are Korean "
         "(KoOCR-Bench, MDPBench Korean, KDoc-OCRBench-V2)."),
         "",
         "How to read the tables:",
         "",
         ("- Rows are ordered by **Overall**. In every column the best value is **bold**; columns marked ↓ "
         "are lower-is-better."),
         (f"- {MARK} marks a license other than Apache-2.0, MIT or BSD (copyleft, a commercial-use "
         "threshold, a regional exclusion, extra usage terms, or no stated license). Check the terms "
         "before commercial use. These models are orange in the charts."),
         ("- *reported* is the score the model's authors published. It can differ from ours because of "
         "page subsets, settings, or scorer versions."),
         ("- A model missing from a table, or a cell shown as –, means the official scorer halted on "
         "that model's predictions."),
         "",
         "## Across benchmarks",
         "",
         "![Rank heatmap](figures/public_leaderboard/rank_heatmap.png)",
         "",
         ("Models are ordered by their mean rank over the benchmarks they were scored on. Ranks are "
         "within this table, out of 19 models at most."),
         "",
         "| Model | License | Benchmarks scored | Mean rank |",
         "| --- | --- | ---: | ---: |"]
    for m, r, n in order:
        L.append(f"| {label(m, data)} | {data['models'][m]['license']} | {n} | {r:.1f} |")
    for b in data["benchmarks"]:
        dot_chart(b, data, FIG / f"{b['id']}.png")
        L += ["", f"## {b['title']}", "",
              f"{b['pages']} · {b['scorer']}. {b['description'].strip()}", "",
              f"![{b['title']}](figures/public_leaderboard/{b['id']}.png)", ""]
        L += table_md(b, data)
    L += ["", "---", "", ("Regenerate with `python scripts/build_public_leaderboard.py` "
          "(needs the `report` extra for matplotlib).")]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--check", action="store_true", help="only validate the YAML")
    args = ap.parse_args(argv)
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    validate(data)
    if args.check:
        print("ok")
        return 0
    PAGE.write_text(build(data), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)} and {len(data['benchmarks']) + 1} figures in "
          f"{FIG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
