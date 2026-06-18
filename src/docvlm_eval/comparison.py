"""Aggregate per-run summaries into the comparison table (Markdown + CSV + JSON)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def load_summaries(results_dir: Path) -> list[dict]:
    out = []
    for f in results_dir.rglob("summary.json"):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            print(f"[warn] skipping {f}: {exc}")
    return out


def robustness_retention(results_dir: Path) -> dict[str, dict[str, float]]:
    """For each model with a robustness run, retention = score(pert)/score(clean) per family."""
    retention: dict[str, dict[str, float]] = {}
    for f in results_dir.rglob("per_sample.json"):
        rows = json.loads(f.read_text(encoding="utf-8"))
        if not rows or "perturbation" not in rows[0]:
            continue
        model = f.parent.parent.name
        by_pert: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_pert[r["perturbation"]].append(r["score"])
        clean_scores = by_pert.get("clean", [])
        clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
        ret = {}
        for pert, scores in by_pert.items():
            if pert == "clean":
                continue
            avg = sum(scores) / len(scores)
            ret[pert] = round(avg / clean, 3) if clean else 0.0
        retention[model] = ret
    return retention


def build_tables(results_dir: str | Path, out_dir: str | Path) -> str:
    """Write comparison_table.{md,csv,json}; return the Markdown text."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    summaries = load_summaries(results_dir)
    if not summaries:
        raise SystemExit(f"No summary.json found under {results_dir}. Run evaluation first.")

    models = sorted({s["model"] for s in summaries})
    benches = sorted({s["benchmark"] for s in summaries})
    params = {s["model"]: s.get("param_count_m") for s in summaries}
    cell = {(s["model"], s["benchmark"]): s for s in summaries}

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

    retention = robustness_retention(results_dir)
    if retention:
        lines += ["\n## Robustness retention (perturbed / clean)\n"]
        perts = sorted({p for r in retention.values() for p in r})
        h2 = ["Model"] + perts + ["Worst"]
        lines.append("| " + " | ".join(h2) + " |")
        lines.append("|" + "|".join(["---"] * len(h2)) + "|")
        for m, r in sorted(retention.items()):
            vals = [f"{r[p]:.2f}" if p in r else "–" for p in perts]
            worst = min(r.values()) if r else float("nan")
            lines.append("| " + " | ".join([m] + vals + [f"{worst:.2f}"]) + " |")

    from .report_md import prettify_tables
    out_dir.mkdir(parents=True, exist_ok=True)
    md = prettify_tables("\n".join(lines)) + "\n"
    (out_dir / "comparison_table.md").write_text(md, encoding="utf-8")

    with open(out_dir / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_m", "benchmark", "primary_metric", "score", "accuracy", "ece", "answer_rate"])
        for s in summaries:
            w.writerow([s["model"], s.get("param_count_m"), s["benchmark"], s.get("primary_metric"),
                        s["score"], s.get("accuracy"), s.get("ece"), s.get("answer_rate")])

    (out_dir / "comparison_table.json").write_text(
        json.dumps({"summaries": summaries, "robustness": retention}, indent=2), encoding="utf-8"
    )
    return md
