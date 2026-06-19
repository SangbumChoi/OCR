#!/usr/bin/env python3
"""Turn the realistic synthetic cases into an eval-pipeline benchmark JSONL.

Walks ``data/probes/realistic_cases/<key>/gt.json`` and emits Samples (qa / spotting /
table / probes) via ``docvlm_eval.synth.load_realistic_samples`` so every model can be run on
these cases with the normal harness:

    python scripts/build_realistic_benchmark.py            # clean -> realistic_cases.jsonl
    python scripts/build_realistic_benchmark.py --variant degraded
    docvlm-eval --model dummy-echo \
      --benchmark data/probes/realistic_cases/realistic_cases.jsonl \
      --benchmark-name realistic_cases --out /tmp/rc --device cpu
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.synth import load_realistic_samples  # noqa: E402

CASES = ROOT / "data" / "probes" / "realistic_cases"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "degraded"], default="clean",
                    help="which rendered image to point Samples at")
    ap.add_argument("--no-probes", action="store_true", help="exclude abstain/consistency probes")
    ap.add_argument("--out", default=None, help="output JSONL (default realistic_cases[ _degraded].jsonl)")
    args = ap.parse_args()

    samples = load_realistic_samples(CASES, variant=args.variant, include_probes=not args.no_probes)
    if not samples:
        raise SystemExit(f"no cases found under {CASES} — run scripts/make_realistic_cases.py first")

    out = Path(args.out) if args.out else CASES / (
        "realistic_cases.jsonl" if args.variant == "clean" else "realistic_cases_degraded.jsonl")
    save_jsonl(samples, out)

    by_metric = Counter(s.metric for s in samples)
    by_type = Counter(s.answer_type for s in samples)
    print(f"[done] {len(samples)} samples ({args.variant}) -> {out}")
    print("  by metric     :", dict(by_metric))
    print("  by answer_type:", dict(by_type))


if __name__ == "__main__":
    main()
