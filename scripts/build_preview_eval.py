#!/usr/bin/env python3
"""Build an OFFLINE evaluation benchmark from the committed 10-sample previews.

Converts ``data/benchmarks/<key>/samples.jsonl`` (raw HF ground truth) into eval Samples
(question + answers + metric) and writes ``data/benchmarks/preview_eval.jsonl`` so every model can
be run on a 10-sample slice of every benchmark with no network:

    python scripts/build_preview_eval.py
    python scripts/run_matrix.py --models smolvlm-256m smolvlm-500m \
      --benchmark data/benchmarks/preview_eval.jsonl --benchmark-name preview_eval
    # or a quick CPU smoke:
    docvlm-eval --model dummy-echo --benchmark data/benchmarks/preview_eval.jsonl \
      --benchmark-name preview_eval --out /tmp/pe --device cpu
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.preview_eval import build_preview_eval  # noqa: E402

BENCH = ROOT / "data" / "benchmarks"


def main():
    samples, stats = build_preview_eval(BENCH)
    if not samples:
        raise SystemExit("no preview samples found — run scripts/fetch_benchmark_samples.py --n 10 first")
    out = BENCH / "preview_eval.jsonl"
    save_jsonl(samples, out)
    print(f"[done] {len(samples)} samples from {stats['benchmarks']} benchmarks -> {out}")
    print("  by benchmark:", dict(Counter(s.meta['benchmark'] for s in samples)))
    print("  by metric   :", dict(Counter(s.metric for s in samples)))
    if stats["skipped"]:
        print("  skipped (no spec / no flat GT):", stats["skipped"])


if __name__ == "__main__":
    main()
