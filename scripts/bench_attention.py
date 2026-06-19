#!/usr/bin/env python3
"""Flash-attention before/after speed benchmark.

Runs a model on a small benchmark once per attention backend (eager / sdpa / flash_attention_2)
and reports load time, avg/p90 latency and peak GPU memory, so we can decide whether to make
flash-attention the default. Backends that aren't available (e.g. flash_attention_2 without the
flash-attn package, or on CPU) are skipped + reported.

    python scripts/bench_attention.py --models internvl2_5-1b smolvlm-500m --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import load_jsonl  # noqa: E402
from docvlm_eval.pipeline import run_evaluation  # noqa: E402
from docvlm_eval.report_md import prettify_tables  # noqa: E402

DEFAULT_BENCH = "data/benchmarks/capability_probe/capability.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--benchmark", default=DEFAULT_BENCH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    # flash_attention_2 needs the flash-attn package AND Ampere+ (unavailable on T4/Turing), so
    # the default compares the universally-available backends; add it explicitly to try flash.
    p.add_argument("--attns", nargs="+", default=["eager", "sdpa"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out", default="docs/results/attention_benchmark.md")
    a = p.parse_args()

    samples = load_jsonl(a.benchmark)
    rows = []
    for m in a.models:
        for attn in a.attns:
            try:
                s = run_evaluation(
                    model_key=m, samples=samples,
                    out_dir=f"docs/results/_attnbench/{m}/{attn}", device=a.device, dtype=a.dtype,
                    max_new_tokens=a.max_new_tokens, benchmark_name="attnbench",
                    resume=False, attn=attn,
                )
                rows.append((m, attn, s.get("load_seconds"), s.get("avg_latency_s"),
                             s.get("p90_latency_s"), s.get("peak_gpu_mb"), "ok"))
            except Exception as exc:
                rows.append((m, attn, "-", "-", "-", "-", f"FAIL: {type(exc).__name__}"))
                print(f"[skip] {m}/{attn}: {exc}")

    hdr = ["model", "attn", "load(s)", "avg lat(s)", "p90(s)", "peak GPU(MB)", "status"]
    lines = ["# Flash-attention speed benchmark\n",
             f"Benchmark: `{a.benchmark}` · device: {a.device} · dtype: {a.dtype}\n",
             "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    # speedup vs eager per model
    eager = {m: r[3] for (m, attn, *r0), r in [((r[0], r[1]), r) for r in rows] if r[1] == "eager" and isinstance(r[3], (int, float))}
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    # recommendation
    lines.append("\n## Reading\n")
    lines.append("Compare `avg lat(s)` across backends per model. If `flash_attention_2` is "
                 "consistently faster (and memory-lighter) than `sdpa`/`eager` with no quality "
                 "change, set `--attn flash_attention_2` (or change the adapter default).")
    md = prettify_tables("\n".join(lines)) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
