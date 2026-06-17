#!/usr/bin/env python3
"""Run a set of models across the cross-benchmark preview set and STORE + aggregate results.

Stores results/<model>/all_preview/{predictions.jsonl,summary.json,per_sample.json} and writes
a model x benchmark score matrix to results/matrix.md + results/matrix.json.

    # CPU smoke (dummy only):
    python scripts/run_matrix.py --models dummy-echo --device cpu
    # try the smallest real models on CPU:
    python scripts/run_matrix.py --models smolvlm-256m florence2-base --device cpu --dtype float32
    # everything (needs a GPU):
    python scripts/run_matrix.py --all --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks import load_jsonl  # noqa: E402
from docvlm_eval.models import list_models  # noqa: E402
from docvlm_eval.pipeline import run_evaluation  # noqa: E402

PREVIEW = "data/benchmarks/all_preview.jsonl"


def main() -> None:
    import docvlm_eval.models.dummy  # noqa: F401

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["dummy-echo"])
    p.add_argument("--all", action="store_true", help="every registered model")
    p.add_argument("--benchmark", default=PREVIEW)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="ignore cached predictions")
    a = p.parse_args()

    models = list_models() if a.all else a.models
    samples = load_jsonl(a.benchmark)
    bench_name = Path(a.benchmark).stem
    results_dir = Path(a.results_dir)

    status: dict[str, str] = {}
    per_model_scores: dict[str, dict[str, float]] = {}
    for m in models:
        out = results_dir / m / bench_name
        try:
            run_evaluation(
                model_key=m, samples=samples, out_dir=str(out), device=a.device,
                dtype=a.dtype, max_new_tokens=a.max_new_tokens, limit=a.limit,
                benchmark_name=bench_name, resume=not a.no_resume,
            )
            status[m] = "ok"
        except Exception as exc:  # capture inference failures per model, keep going
            status[m] = f"FAIL: {type(exc).__name__}: {str(exc)[:160]}"
            print(f"[fail] {m}: {status[m]}")

    # ---- build the matrix CUMULATIVELY: include every model that has results for this
    #      benchmark under results_dir, not just the ones run in this invocation. ----
    for ps_file in results_dir.glob(f"*/{bench_name}/per_sample.json"):
        m = ps_file.parent.parent.name
        rows = json.loads(ps_file.read_text())
        per_model_scores[m] = {r["sample_id"]: r["score"] for r in rows}
        status.setdefault(m, "ok (cached)")

    benches = [s.sample_id for s in samples]
    models = sorted(set(models) | set(per_model_scores))
    lines = ["# Cross-benchmark result matrix (preview set)\n",
             f"Models run: {len(per_model_scores)}/{len(models)} · benchmarks: {len(benches)}\n",
             "Per-cell = task score (ANLS / relaxed-acc / OCRBench / exact). One preview sample"
             " per benchmark, so treat as a *plumbing + sanity* matrix, not leaderboard accuracy.\n"]
    header = ["model"] + benches
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in models:
        if m not in per_model_scores:
            lines.append(f"| {m} | " + " | ".join(["✗"] * len(benches)) + " |")
            continue
        row = [m] + [f"{per_model_scores[m].get(b, float('nan')):.2f}" for b in benches]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Run status\n")
    for m in models:
        lines.append(f"- **{m}**: {status.get(m, '?')}")

    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / f"matrix_{bench_name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results_dir / f"matrix_{bench_name}.json").write_text(
        json.dumps({"status": status, "scores": per_model_scores, "benchmarks": benches}, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[done] matrix -> {md_path}")


if __name__ == "__main__":
    main()
