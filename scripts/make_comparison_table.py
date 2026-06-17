#!/usr/bin/env python3
"""Aggregate all run summaries into the comparison table (Markdown + CSV + JSON).

Walks ``results/`` for ``summary.json`` files (one per model x benchmark), and emits a
pivoted table: rows = models, columns = benchmarks, plus calibration (ECE) and - if a
robustness run is present - the worst-case retention. Also writes a per-perturbation
robustness breakdown for the knowledge-gap section.

Example
-------
    python scripts/make_comparison_table.py --results-dir results --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_summaries(results_dir: Path) -> list[dict]:
    out = []
    for f in results_dir.rglob("summary.json"):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] skipping {f}: {exc}")
    return out


def _robustness_retention(results_dir: Path) -> dict[str, dict[str, float]]:
    """For each model with a robustness run, retention = score(pert)/score(clean) per family."""
    retention: dict[str, dict[str, float]] = {}
    for f in results_dir.rglob("per_sample.json"):
        rows = json.loads(f.read_text(encoding="utf-8"))
        if not rows or "perturbation" not in rows[0]:
            continue
        model = f.parent.parent.name if f.parent.name == f.parent.name else f.parent.name
        # group scores by perturbation
        by_pert: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_pert[r["perturbation"]].append(r["score"])
        clean = sum(by_pert.get("clean", [0])) / max(1, len(by_pert.get("clean", [1])))
        ret = {}
        for pert, scores in by_pert.items():
            if pert == "clean":
                continue
            avg = sum(scores) / len(scores)
            ret[pert] = round(avg / clean, 3) if clean else 0.0
        retention[model] = ret
    return retention


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    summaries = _load_summaries(results_dir)
    if not summaries:
        raise SystemExit(f"No summary.json found under {results_dir}. Run scripts/evaluate.py first.")

    models = sorted({s["model"] for s in summaries})
    benches = sorted({s["benchmark"] for s in summaries})
    params = {s["model"]: s.get("param_count_m") for s in summaries}

    cell: dict[tuple[str, str], dict] = {}
    for s in summaries:
        cell[(s["model"], s["benchmark"])] = s

    # ---- Markdown ----
    lines = ["# Comparison Table\n", "## Headline scores (per-benchmark primary metric)\n"]
    header = ["Model", "Params (M)"] + benches + ["Mean ECE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in models:
        row = [m, f"{params.get(m) or 0:.0f}"]
        eces = []
        for b in benches:
            c = cell.get((m, b))
            if c:
                row.append(f"{c['score']:.3f}")
                if c.get("ece") is not None:
                    eces.append(c["ece"])
            else:
                row.append("–")
        row.append(f"{sum(eces)/len(eces):.3f}" if eces else "–")
        lines.append("| " + " | ".join(row) + " |")

    retention = _robustness_retention(results_dir)
    if retention:
        lines += ["\n## Robustness retention (perturbed score / clean score)\n"]
        perts = sorted({p for r in retention.values() for p in r})
        h2 = ["Model"] + perts + ["Worst"]
        lines.append("| " + " | ".join(h2) + " |")
        lines.append("|" + "|".join(["---"] * len(h2)) + "|")
        for m, r in sorted(retention.items()):
            vals = [f"{r.get(p, float('nan')):.2f}" if p in r else "–" for p in perts]
            worst = min(r.values()) if r else float("nan")
            lines.append("| " + " | ".join([m] + vals + [f"{worst:.2f}"]) + " |")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- CSV ----
    with open(out_dir / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_m", "benchmark", "primary_metric", "score", "accuracy", "ece", "answer_rate"])
        for s in summaries:
            w.writerow([
                s["model"], s.get("param_count_m"), s["benchmark"], s.get("primary_metric"),
                s["score"], s.get("accuracy"), s.get("ece"), s.get("answer_rate"),
            ])

    (out_dir / "comparison_table.json").write_text(
        json.dumps({"summaries": summaries, "robustness": retention}, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\n[done] wrote comparison_table.{{md,csv,json}} to {out_dir}")


if __name__ == "__main__":
    main()
