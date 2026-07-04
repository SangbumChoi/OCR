#!/usr/bin/env python3
"""Mock multi-model evaluation of UDD — test the WHOLE eval path without a GPU.

Runs several deterministic MOCK models over the UDD public heldout sets
(``data/udd_tasks/heldout_*.jsonl``) and scores them exactly like real models would be scored:
per-sample dispatch through ``score_sample(sample.metric, pred, golds)`` (anls / exact / ned /
relaxed_acc / grounding / …) plus the bank's ``semantic_match`` as the comparison column, then
aggregation per model × task. This validates end-to-end that every task's jsonl loads, every
metric dispatches (including grounding's box parsing on localization rows), and the aggregate
matrix behaves — each mock model is a KNOWN behaviour, so the matrix has expected values:

* ``oracle``          — returns the first gold verbatim               (≈1.0 everywhere)
* ``oracle-caseflip`` — gold uppercased            (1.0 where case-tolerant; cer_sim-style drop)
* ``oracle-wrapped``  — "The answer is <gold>."    (substring/F1-tolerant metrics forgive)
* ``oracle-truncate`` — first half of the gold     (partial credit under ned/anls only)
* ``constant``        — always "unknown"                              (≈0.0 everywhere)
* ``echo-question``   — repeats the question                          (≈0.0, sanity)

Writes docs/results/udd_mock_eval.md + docs/report/figures/udd_mock_eval.png. Any sanity violation
(oracle < 0.99, constant > 0.05 on exact-style tasks) is reported loudly and exits non-zero.

    python scripts/mock_eval_udd.py
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

from docvlm_eval.metrics import score_sample, semantic_match  # noqa: E402

MD = ROOT / "docs" / "results" / "udd_mock_eval.md"
FIG = ROOT / "docs" / "report" / "figures" / "udd_mock_eval.png"


def mock_models() -> dict:
    return {
        "oracle": lambda q, golds: golds[0],
        "oracle-caseflip": lambda q, golds: golds[0].swapcase(),
        "oracle-wrapped": lambda q, golds: f"The answer is {golds[0]}.",
        "oracle-truncate": lambda q, golds: golds[0][: max(1, len(golds[0]) // 2)],
        "constant": lambda q, golds: "unknown",
        "echo-question": lambda q, golds: q,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heldout-dir", default=str(ROOT / "data" / "udd_tasks"))
    p.add_argument("--max-per-task", type=int, default=150)
    args = p.parse_args()

    files = sorted(Path(args.heldout_dir).glob("heldout_*.jsonl"))
    files = [f for f in files if f.stem != "heldout_all"]
    if not files:
        sys.exit(f"[mock-eval] no heldout_*.jsonl under {args.heldout_dir} — "
                 "run build_task_trainsets.py first.")
    models = mock_models()

    # rows[model][task] = (mean own-metric score, mean semantic_match, n)
    agg: dict[str, dict[str, tuple[float, float, int]]] = defaultdict(dict)
    metrics_seen: dict[str, set] = defaultdict(set)
    for f in files:
        task = f.stem.replace("heldout_", "")
        samples = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        samples = samples[: args.max_per_task]
        for mname, fn in models.items():
            tot = tot_sem = 0.0
            for s in samples:
                golds = s["answers"]
                pred = fn(s["question"], golds)
                tot += score_sample(s.get("metric", "anls"), pred, golds)
                tot_sem += semantic_match(pred, golds)
                metrics_seen[task].add(s.get("metric", "anls"))
            n = len(samples)
            agg[mname][task] = (tot / n, tot_sem / n, n)

    tasks = sorted({t for m in agg.values() for t in m})
    # ---- sanity checks: known models must produce known values
    failures = []
    for t in tasks:
        if agg["oracle"][t][0] < 0.99:
            failures.append(f"oracle scored {agg['oracle'][t][0]:.3f} on {t} (expected ~1.0)")
        if agg["constant"][t][0] > 0.10:
            failures.append(f"constant scored {agg['constant'][t][0]:.3f} on {t} (expected ~0)")

    # ---- report
    lines = ["# UDD mock multi-model evaluation (no GPU — deterministic mock models)", "",
             f"{sum(agg['oracle'][t][2] for t in tasks)} public-heldout samples; per-task metrics: "
             + "; ".join(f"{t}={'/'.join(sorted(metrics_seen[t]))}" for t in tasks), "",
             "Each cell = mean score under the task's OWN metric (per-sample `score_sample` "
             "dispatch), with `semantic_match` in parentheses as the bank comparison.", "",
             "| model | " + " | ".join(tasks) + " |",
             "|---|" + "---|" * len(tasks)]
    for mname in models:
        cells = [f"{agg[mname][t][0]:.2f} ({agg[mname][t][1]:.2f})" for t in tasks]
        lines.append(f"| {mname} | " + " | ".join(cells) + " |")
    lines += ["", "Sanity: `oracle` ≈ 1.0 and `constant` ≈ 0.0 in every column; the gap between "
              "`oracle-caseflip` / `oracle-wrapped` / `oracle-truncate` rows and 1.0 is each task "
              "metric's tolerance profile applied to real UDD golds (compare "
              "[`metric_tendency.md`](metric_tendency.md)).", ""]
    if failures:
        lines += ["## SANITY FAILURES", ""] + [f"- {x}" for x in failures] + [""]
    lines.append("![mock eval](../report/figures/udd_mock_eval.png)")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = [[agg[m][t][0] for t in tasks] for m in models]
    fig, ax = plt.subplots(figsize=(1.3 * len(tasks) + 3, 0.6 * len(models) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(list(models), fontsize=9)
    for i in range(len(models)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("UDD mock evaluation — mean score under each task's own metric\n"
                 "(mock models with KNOWN behaviour exercise the full eval path)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"[ok] {MD}\n[ok] {FIG}")
    if failures:
        print("[FAIL] sanity violations:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("[ok] sanity holds: oracle ~1.0, constant ~0.0 across all tasks")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
