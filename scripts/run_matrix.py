#!/usr/bin/env python3
"""Run a set of models across the cross-benchmark preview set and STORE + aggregate results.

Stores docs/results/<model>/all_preview/{predictions.jsonl,summary.json,per_sample.json} and writes
a model x benchmark score matrix to docs/results/matrix.md + docs/results/matrix.json.

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
from docvlm_eval.metrics import aggregate  # noqa: E402
from docvlm_eval.models import list_models  # noqa: E402
from docvlm_eval.pipeline import run_evaluation  # noqa: E402
from docvlm_eval.schema import Prediction  # noqa: E402

PREVIEW = "data/benchmarks/all_preview.jsonl"


# --- carried over verbatim from a run's summary.json so a re-score keeps the run metadata ---
_META_KEYS = (
    "model", "hf_id", "param_count_m", "benchmark", "device", "dtype", "attn",
    "load_seconds", "avg_latency_s", "p90_latency_s", "total_infer_s",
    "peak_gpu_mb", "peak_cpu_mb",
)


def rescore_cached(samples, results_dir: Path, bench_name: str) -> dict[str, str]:
    """Re-aggregate every model's CACHED predictions against the CURRENT benchmark jsonl.

    No model is loaded — this only re-runs the scorer, so it refreshes ``summary.json`` and
    ``per_sample.json`` (e.g. after the answer_type taxonomy or a metric changed) for every
    model dir that already has a committed ``predictions.jsonl``. Models without cached
    predictions are skipped (they need a real GPU run).
    """
    status: dict[str, str] = {}
    for pred_file in sorted(results_dir.glob(f"*/{bench_name}/predictions.jsonl")):
        m = pred_file.parent.parent.name
        preds: dict[str, Prediction] = {}
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                preds[d["sample_id"]] = Prediction(**d)
        if not preds:
            status[m] = "skip (empty predictions)"
            continue
        result = aggregate(samples, preds)
        # preserve run metadata (params/latency/device) from the prior summary.json if present
        sm_path = pred_file.parent / "summary.json"
        if sm_path.exists():
            prior = json.loads(sm_path.read_text(encoding="utf-8"))
            result["summary"].update({k: prior[k] for k in _META_KEYS if k in prior})
        else:
            result["summary"]["model"] = m
            result["summary"]["benchmark"] = bench_name
        # backfill static metadata (params/hf_id) from the registry class attributes when the
        # prior summary lacked them (e.g. a partial run that never wrote a full summary).
        if not result["summary"].get("param_count_m"):
            from docvlm_eval.models.registry import _REGISTRY
            import docvlm_eval.models  # noqa: F401 (populate registry)
            from docvlm_eval.models import list_models  # noqa: F401
            list_models()
            cls = _REGISTRY.get(m)
            if cls is not None:
                result["summary"].setdefault("param_count_m", getattr(cls, "param_count_m", None))
                result["summary"].setdefault("hf_id", getattr(cls, "hf_id", None))
        sm_path.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
        (pred_file.parent / "per_sample.json").write_text(
            json.dumps(result["per_sample"], indent=2, ensure_ascii=False), encoding="utf-8")
        covered = sum(1 for s in samples if s.sample_id in preds)
        status[m] = f"rescored ({covered}/{len(samples)} samples)"
    return status


def main() -> None:
    import docvlm_eval.models.dummy  # noqa: F401

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["dummy-echo"])
    p.add_argument("--all", action="store_true", help="every registered model")
    p.add_argument("--benchmark", default=PREVIEW)
    p.add_argument("--results-dir", default="docs/results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="auto", help="auto | eager | sdpa | flash_attention_2")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="ignore cached predictions")
    p.add_argument("--rescore", action="store_true",
                   help="re-aggregate cached predictions against the current benchmark jsonl "
                        "WITHOUT loading any model (refreshes summary/per_sample after a "
                        "taxonomy/metric change), then rebuild the matrix")
    a = p.parse_args()

    models = list_models() if a.all else a.models
    samples = load_jsonl(a.benchmark)
    bench_name = Path(a.benchmark).stem
    results_dir = Path(a.results_dir)

    status: dict[str, str] = {}
    per_model_scores: dict[str, dict[str, float]] = {}
    if a.rescore:
        status.update(rescore_cached(samples, results_dir, bench_name))
        for m, st in sorted(status.items()):
            print(f"[rescore] {m}: {st}")
    run_list = [] if a.rescore else models
    n_models = len(run_list)
    for i, m in enumerate(run_list, 1):
        print(f"[run_matrix {i}/{n_models}] loading + evaluating {m} on {bench_name} "
              f"({n_models - i} model(s) left)", flush=True)
        out = results_dir / m / bench_name
        try:
            run_evaluation(
                model_key=m, samples=samples, out_dir=str(out), device=a.device,
                dtype=a.dtype, max_new_tokens=a.max_new_tokens, limit=a.limit,
                benchmark_name=bench_name, resume=not a.no_resume, attn=a.attn,
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

    # efficiency (time + memory) from each model's summary.json
    eff: dict[str, dict] = {}
    for sm_file in results_dir.glob(f"*/{bench_name}/summary.json"):
        m = sm_file.parent.parent.name
        try:
            eff[m] = json.loads(sm_file.read_text())
        except Exception:
            pass

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
    # ---- efficiency table: CPU/GPU inference time + memory, measured by the model wrapper ----
    lines.append("\n## Efficiency (load / latency / memory)\n")
    eh = ["model", "device", "params(M)", "load(s)", "avg lat(s)", "p90(s)", "peak CPU(MB)", "peak GPU(MB)"]
    lines.append("| " + " | ".join(eh) + " |")
    lines.append("|" + "|".join(["---"] * len(eh)) + "|")
    for m in models:
        s = eff.get(m)
        if not s:
            continue
        lines.append("| " + " | ".join(str(x) for x in [
            m, s.get("device", "-"), f"{s.get('param_count_m') or 0:.0f}",
            s.get("load_seconds", "-"), s.get("avg_latency_s", "-"), s.get("p90_latency_s", "-"),
            s.get("peak_cpu_mb", "-"), s.get("peak_gpu_mb", "-"),
        ]) + " |")

    lines.append("\n## Run status\n")
    for m in models:
        lines.append(f"- **{m}**: {status.get(m, '?')}")

    from docvlm_eval.report_md import prettify_tables
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / f"matrix_{bench_name}.md"
    md_path.write_text(prettify_tables("\n".join(lines)) + "\n", encoding="utf-8")
    (results_dir / f"matrix_{bench_name}.json").write_text(
        json.dumps({"status": status, "scores": per_model_scores, "benchmarks": benches}, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[done] matrix -> {md_path}")


if __name__ == "__main__":
    main()
