#!/usr/bin/env python3
"""Render the task-value ablation: is each UDD task worth adding (at a fixed data budget)?

Reads ``docs/results/task_value_results.json`` (written by ``scripts/run_task_value.py``) and, for
each model, computes the per-task Δ against the un-tuned baseline on each probe, then writes:

  docs/results/task_value.md            table: baseline + per-task score & Δ (capability / realistic)
  docs/report/figures/task_value.png    bar chart of the capability Δ per task (green=+, red=−)

A positive Δ = training on that task alone (equal N) improved the validation suite -> the task earns
its slot; ≈0 / negative -> it does not help (or transfers poorly). ``task:all`` is the mixed-mix
reference. Run after run_task_value.py; with no results file it prints how to generate one.

    python scripts/analyze_task_value.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "task_value_results.json"
FIG = ROOT / "docs" / "report" / "figures" / "task_value.png"
MD = ROOT / "docs" / "results" / "task_value.md"
PROBES = ["capability", "realistic", "spatial"]


def _score(payload: dict, probe: str):
    return ((payload or {}).get("probes", {}).get(probe, {}) or {}).get("score")


def _fmt(v):
    return "—" if v is None else f"{v:.1f}"


def _delta(v, base):
    return None if (v is None or base is None) else v - base


def _demo_note(doc: dict) -> str:
    return (" **[DEMO — illustrative numbers; rerun `run_task_value.py` on a GPU to replace.]**"
            if doc.get("demo") else "")


def build_md(doc: dict) -> str:
    out = ["# Task-value ablation — is each task worth adding?" + _demo_note(doc), "",
           "For each UDD task we LoRA-fine-tune the base **on that task alone, with an equal number of "
           "samples**, and score the fixed synthetic probe suite. `Δ` is versus the un-tuned baseline; "
           "a positive Δ means the task earns its slot at that data budget, ≈0/negative means it does "
           "not help (or transfers poorly to the validation distribution). `task:all` = the mixed mix.",
           ""]
    models = doc.get("models", {})
    if not models:
        out.append("_No results yet — run `scripts/run_task_value.py` (GPU) to populate this._")
        return "\n".join(out) + "\n"
    for model, arms in models.items():
        base = arms.get("baseline", {})
        control = next((a.get("control", {}) for k, a in arms.items() if k.startswith("task:")), {})
        n = control.get("count", "?")
        out += [f"## {model}", "",
                f"Equal budget: **N={n} samples/task**, steps={control.get('steps', '?')}, "
                f"LoRA placement={control.get('placement', '?')}.", "",
                "| training set | capability | Δ | realistic | Δ | verdict |",
                "|---|---|---|---|---|---|"]
        b_cap, b_real = _score(base, "capability"), _score(base, "realistic")
        out.append(f"| _baseline (no FT)_ | {_fmt(b_cap)} | — | {_fmt(b_real)} | — | reference |")
        rows = []
        for arm, payload in arms.items():
            if not arm.startswith("task:"):
                continue
            cap, real = _score(payload, "capability"), _score(payload, "realistic")
            dcap, dreal = _delta(cap, b_cap), _delta(real, b_real)
            rows.append((arm.split(":", 1)[1], cap, dcap, real, dreal))
        rows.sort(key=lambda r: (r[0] == "all", -(r[2] if r[2] is not None else -99)))
        for task, cap, dcap, real, dreal in rows:
            verdict = ("worth adding" if (dcap or 0) > 0.5 else
                       "marginal" if (dcap or 0) > -0.5 else "not worth it")
            dc = "—" if dcap is None else f"{dcap:+.1f}"
            dr = "—" if dreal is None else f"{dreal:+.1f}"
            label = "**all (mixed)**" if task == "all" else task
            out += [f"| {label} | {_fmt(cap)} | {dc} | {_fmt(real)} | {dr} | {verdict} |"]
        out.append("")
    out += ["![task value](../report/figures/task_value.png)", ""]
    return "\n".join(out)


def plot(doc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = doc.get("models", {})
    series = {}   # model -> [(task, dcap)]
    for model, arms in models.items():
        base = _score(arms.get("baseline", {}), "capability")
        pts = []
        for arm, payload in arms.items():
            if not arm.startswith("task:"):
                continue
            d = _delta(_score(payload, "capability"), base)
            if d is not None:
                pts.append((arm.split(":", 1)[1], d))
        if pts:
            pts.sort(key=lambda r: (r[0] == "all", -r[1]))
            series[model] = pts
    if not series:
        print("[task-value] no scored tasks yet — nothing to plot."); return

    fig, axes = plt.subplots(1, len(series), figsize=(6.2 * len(series), 5), squeeze=False)
    for ax, (model, pts) in zip(axes[0], series.items()):
        labels = [t for t, _ in pts]
        vals = [d for _, d in pts]
        colors = ["#2a8a2a" if v >= 0 else "#b03030" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("capability Δ vs baseline")
        ax.set_title(f"{model}: per-task value (equal N)", fontsize=12)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:+.1f}", (i, v), textcoords="offset points",
                        xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=8)
    fig.suptitle("Is each task worth adding? capability Δ after training on ONE task (fixed budget)"
                 + ("  [DEMO]" if doc.get("demo") else ""),
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {FIG}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RESULTS))
    args = p.parse_args()
    path = Path(args.results)
    if not path.exists():
        print(f"[task-value] no results at {path}.\n"
              f"  1) python scripts/build_task_trainsets.py --per-task 30\n"
              f"  2) python scripts/run_task_value.py --count 30   (GPU)\n"
              f"  3) python scripts/analyze_task_value.py")
        MD.write_text(build_md({}), encoding="utf-8")
        print(f"[ok] wrote placeholder {MD}")
        return
    doc = json.loads(path.read_text())
    MD.write_text(build_md(doc), encoding="utf-8")
    print(f"[ok] {MD}")
    plot(doc)


if __name__ == "__main__":
    main()
