"""Console entry points for the unified package.

Exposed via ``[project.scripts]`` in pyproject.toml so, after ``pip install -e .``:

    docvlm-eval        --model ... --benchmark ... --out ...   # run a model on a benchmark
    docvlm-build-bench --benchmark all --limit 300             # build benchmark JSONL from HF
    docvlm-fetch       [--only docvqa ...]                     # one preview sample per benchmark
    docvlm-robustness  --base ... --out-dir ...                # build the paired robustness probe
    docvlm-table       --results-dir results                   # aggregate the comparison table

Each function is also callable in-process (used by the thin ``scripts/*.py`` shims and tests).
"""

from __future__ import annotations

import argparse
import json

from .benchmarks import load_jsonl, save_jsonl


# --------------------------------------------------------------------- evaluate
def evaluate(argv: list[str] | None = None) -> None:
    from .models import list_models
    from .pipeline import run_evaluation
    import docvlm_eval.models.dummy  # noqa: F401  (register CPU smoke model)

    p = argparse.ArgumentParser(prog="docvlm-eval", description="Run a model on a benchmark.")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--model")
    p.add_argument("--benchmark", help="normalised benchmark JSONL")
    p.add_argument("--benchmark-name", default="benchmark")
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args(argv)

    if a.list_models:
        print("\n".join(list_models()))
        return
    for req in ("model", "benchmark", "out"):
        if getattr(a, req) is None:
            raise SystemExit(f"--{req} is required (or use --list-models)")
    summary = run_evaluation(
        model_key=a.model, samples=load_jsonl(a.benchmark), out_dir=a.out,
        device=a.device, dtype=a.dtype, max_new_tokens=a.max_new_tokens,
        limit=a.limit, benchmark_name=a.benchmark_name, resume=not a.no_resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- build_benchmarks
def build_benchmarks(argv: list[str] | None = None) -> None:
    from .benchmarks.hf_builders import BUILDERS

    p = argparse.ArgumentParser(prog="docvlm-build-bench", description="Build benchmark JSONL from HF.")
    p.add_argument("--benchmark", choices=list(BUILDERS) + ["all"], required=True)
    p.add_argument("--out-dir", default="data/benchmarks")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    names = list(BUILDERS) if a.benchmark == "all" else [a.benchmark]
    for name in names:
        print(f"[build] {name} ...")
        print(f"[done]  {BUILDERS[name](a.out_dir, limit=a.limit)}")


# ----------------------------------------------------------------- fetch_samples
def fetch_samples(argv: list[str] | None = None) -> None:
    from .benchmarks.catalog import fetch_many, fetch_one, load_catalog

    p = argparse.ArgumentParser(prog="docvlm-fetch", description="Fetch preview sample(s) per benchmark.")
    p.add_argument("--only", nargs="+")
    p.add_argument("--out-dir", default="data/benchmarks")
    p.add_argument("--force", action="store_true")
    p.add_argument("--refresh-meta", action="store_true")
    p.add_argument("--catalog", default=None)
    p.add_argument("--n", type=int, default=1,
                   help="samples per benchmark; >1 writes <key>/samples/NN.jpg + samples.jsonl")
    p.add_argument("--max-px", type=int, default=1000, help="downscale longest side (when --n > 1)")
    a = p.parse_args(argv)
    entries = [e for e in load_catalog(a.catalog) if not a.only or e["key"] in a.only]
    stats: dict[str, int] = {}
    for e in entries:
        if a.n > 1:
            r = fetch_many(e, a.out_dir, n=a.n, force=a.force, max_px=a.max_px)
        else:
            r = fetch_one(e, a.out_dir, force=a.force, refresh_meta=a.refresh_meta)
        stats[r] = stats.get(r, 0) + 1
    print(f"\n[done] {stats} over {len(entries)} catalog entries")


# --------------------------------------------------------------- build_robustness
def build_robustness(argv: list[str] | None = None) -> None:
    from .benchmarks.robustness import VISUAL, build_robustness_set
    from pathlib import Path

    p = argparse.ArgumentParser(prog="docvlm-robustness", description="Build the paired robustness probe.")
    p.add_argument("--base", required=True)
    p.add_argument("--out-dir", default="data/robustness/docvqa")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--perturbations", nargs="+", default=VISUAL + ["term_paraphrase"])
    a = p.parse_args(argv)
    base = load_jsonl(a.base)[: a.limit]
    expanded = build_robustness_set(base, a.out_dir, perturbations=a.perturbations)
    out_jsonl = Path(a.out_dir) / "robustness.jsonl"
    save_jsonl(expanded, out_jsonl)
    print(f"[done] {len(base)} base -> {len(expanded)} probe samples -> {out_jsonl}")


# ------------------------------------------------------------------ comparison
def comparison_table(argv: list[str] | None = None) -> None:
    from .comparison import build_tables

    p = argparse.ArgumentParser(prog="docvlm-table", description="Aggregate the comparison table.")
    p.add_argument("--results-dir", default="docs/results")
    p.add_argument("--out-dir", default="docs/results")
    a = p.parse_args(argv)
    print(build_tables(a.results_dir, a.out_dir))
    print(f"[done] wrote comparison_table.{{md,csv,json}} to {a.out_dir}")
