#!/usr/bin/env python3
"""Single evaluation entrypoint (Task PoC requirement).

Loads ANY registered candidate model, runs it on a normalised benchmark JSONL, and writes
per-model scores. Adding a new model needs zero changes here - just register an adapter.

Examples
--------
List models:
    python scripts/evaluate.py --list-models

Evaluate one model on one benchmark:
    python scripts/evaluate.py --model internvl3-1b \
        --benchmark data/benchmarks/docvqa.jsonl --benchmark-name docvqa \
        --out results/internvl3-1b/docvqa --limit 200

Smoke test on CPU with the bundled custom benchmark + the dummy model:
    python scripts/evaluate.py --model dummy-echo \
        --benchmark data/custom/custom.jsonl --benchmark-name custom \
        --out /tmp/custom --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks import load_jsonl  # noqa: E402
from docvlm_eval.models import list_models  # noqa: E402
from docvlm_eval.pipeline import run_evaluation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-models", action="store_true", help="print registered model keys and exit")
    p.add_argument("--model", type=str, help="registered model key (see --list-models)")
    p.add_argument("--benchmark", type=str, help="path to normalised benchmark JSONL")
    p.add_argument("--benchmark-name", type=str, default="benchmark")
    p.add_argument("--out", type=str, help="output directory for predictions + summary")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-resume", action="store_true", help="ignore existing predictions.jsonl")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # register the tiny CPU model used only for smoke tests
    import docvlm_eval.models.dummy  # noqa: F401

    if args.list_models:
        print("\n".join(list_models()))
        return

    for req in ("model", "benchmark", "out"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req} is required (or use --list-models)")

    samples = load_jsonl(args.benchmark)
    summary = run_evaluation(
        model_key=args.model,
        samples=samples,
        out_dir=args.out,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        benchmark_name=args.benchmark_name,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
