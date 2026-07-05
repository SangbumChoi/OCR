#!/usr/bin/env python3
"""Reading-order analysis: content-vs-order split + the switch-threshold characterization.

Runs deterministic MOCK readers over the reading-order probe (and, when the merged corpus is on
disk, over the OmniDocBench heldout rows — their ``regions`` are stored in reading order) and
scores them with the split metrics (``content_bag`` vs ``order_tau``). Each mock reader is a KNOWN
reading strategy, so the report demonstrates that the measurement isolates what it claims:

* ``correct``       — reads in the gold order                    (content 1.0, order 1.0)
* ``row-major``     — reads ACROSS parallel columns/boxes        (content 1.0, order << 1)
* ``reversed``      — right order reversed                        (content 1.0, order 0.0)
* ``half-reader``   — reads only the first half, in order         (content 0.5, order 1.0)
* ``gap-sensitive`` — column-major until the box gap exceeds 150px, then row-major — the
                      "logic switch" hypothesis; its order_tau-vs-gap curve shows the flip.

Real models slot into the same harness later: generate predictions for
``data/probes/reading_order/probe.jsonl`` (plus the OmniDocBench set) and score with the same
metrics. Writes docs/results/reading_order.md + docs/report/figures/reading_order_switch.png.

    python scripts/make_reading_order_probe.py && python scripts/analyze_reading_order.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.metrics import content_bag, order_tau  # noqa: E402

MD = ROOT / "docs" / "results" / "reading_order.md"
FIG = ROOT / "docs" / "report" / "figures" / "reading_order_switch.png"


def _row_major(elements: list[str]) -> list[str]:
    """Interleave the two column halves — reading ACROSS instead of down-then-across."""
    half = (len(elements) + 1) // 2
    left, right = elements[:half], elements[half:]
    out = []
    for i in range(half):
        out.append(left[i])
        if i < len(right):
            out.append(right[i])
    return out


def readers() -> dict:
    return {
        "correct": lambda els, meta: els,
        "row-major": lambda els, meta: _row_major(els),
        "reversed": lambda els, meta: list(reversed(els)),
        "half-reader": lambda els, meta: els[: max(1, len(els) // 2)],
        "gap-sensitive": lambda els, meta: (_row_major(els) if meta.get("gap_px", 0) > 150
                                            else els),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", default=str(ROOT / "data" / "probes" / "reading_order" / "probe.jsonl"))
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"),
                   help="merged UDD (for the OmniDocBench heldout section; skipped if absent)")
    args = p.parse_args()

    probe_path = Path(args.probe)
    if not probe_path.exists():
        sys.exit("[reading-order] run scripts/make_reading_order_probe.py first")
    rows = [json.loads(ln) for ln in probe_path.read_text().splitlines() if ln.strip()]
    order_rows = [r for r in rows if r["metric"] == "order_tau"
                  and r["answer_type"] != "reading-order:sweep"]
    sweep_rows = sorted((r for r in rows if r["answer_type"] == "reading-order:sweep"),
                        key=lambda r: r["meta"]["gap_px"])

    # ---- per-layout content/order split for each known reader
    split: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in order_rows:
        elements = r["answers"][0].split("\n")
        for name, fn in readers().items():
            pred = " ".join(fn(elements, r["meta"]))
            split[name][r["meta"]["layout"]].append(
                (content_bag(pred, r["answers"]), order_tau(pred, r["answers"])))
    layouts = sorted({r["meta"]["layout"] for r in order_rows})

    # ---- the switch-threshold curve on the gap sweep
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in sweep_rows:
        elements = r["answers"][0].split("\n")
        for name, fn in readers().items():
            pred = " ".join(fn(elements, r["meta"]))
            curves[name].append((r["meta"]["gap_px"], order_tau(pred, r["answers"])))

    # ---- OmniDocBench heldout (public data with real reading-order GT), when available
    omni_note = "_merged corpus not on disk — OmniDocBench section pending rebuild_"
    if Path(args.src).exists():
        try:
            from datasets import load_from_disk
            ds = load_from_disk(args.src)
            n_omni = sum(1 for i in range(len(ds))
                         if ds["source"][i] == "omnidocbench" and ds["fold"][i] == "heldout"
                         and ds["n_regions"][i] > 1)
            omni_note = (f"**{n_omni} OmniDocBench heldout pages** carry real reading-order GT "
                         "(`regions` are stored in reading order) — score real models on them with "
                         "`metric=content_bag/order_tau` using the region texts as the gold "
                         "element list.")
        except Exception as exc:
            omni_note = f"_corpus load failed: {type(exc).__name__}_"

    # ---- figure: the switch curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, pts in curves.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=name)
    ax.set_xlabel("gap between parallel boxes (px)")
    ax.set_ylabel("order_tau (1 = column-major gold order)")
    ax.set_title("Switch-threshold characterization — order score vs box gap\n"
                 "(mock readers with KNOWN strategies; the gap-sensitive reader flips at 150px)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- report
    lines = ["# Reading order — content/order split + switch characterization", "",
             "Score ORDER separately from CONTENT (`metrics/order.py`): `content_bag` = fraction "
             "of elements read at all; `order_tau` = normalized Kendall rank correlation of the "
             "found elements vs the gold reading order. The GAP isolates order failures from "
             "recognition failures. Mock readers with known strategies validate the measurement:",
             "", "| reader | " + " | ".join(f"{lo} (content / order)" for lo in layouts) + " |",
             "|---|" + "---|" * len(layouts)]
    for name in readers():
        cells = []
        for lo in layouts:
            pairs = split[name][lo]
            c = sum(x for x, _ in pairs) / len(pairs)
            o = sum(y for _, y in pairs) / len(pairs)
            cells.append(f"{c:.2f} / {o:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["",
              "Reading guide: `row-major` and `reversed` keep content 1.00 while order collapses — "
              "exactly the failure a plain transcript metric cannot see; `half-reader` shows the "
              "inverse (content 0.50, order 1.00). Each probe image also carries a **k-th element** "
              "QA and a **segmentation-count** QA (exact) — correlate segmentation accuracy with "
              "order accuracy across real models to test the hypothesis that paragraph/layout "
              "segmentation is the capability underlying reading order.", "",
              "![switch curve](../report/figures/reading_order_switch.png)", "",
              "The gap-sweep curve above is the *logic-switch* characterization: a real model's "
              "`order_tau` plotted against the box gap reveals where (and whether cleanly) its "
              "reading strategy flips between column-major and row-major — the `gap-sensitive` "
              "mock reader shows the signature shape.", "",
              "## Public-data validation", "", omni_note]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {MD}\n[ok] {FIG}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
