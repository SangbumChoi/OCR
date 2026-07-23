#!/usr/bin/env python3
"""Export, generate, or apply quality-gated cross-tokenizer teacher targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.student.teacher_targets import (
    apply_teacher_predictions,
    export_teacher_requests,
    generate_teacher_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--src", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--max-requests", type=int)
    export.add_argument("--selection-seed", type=int, default=0)

    generate = commands.add_parser("generate")
    generate.add_argument("--requests", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--model-revision")
    generate.add_argument("--device", default="cuda")
    generate.add_argument("--dtype", default="bfloat16")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--no-resume", action="store_true")

    apply = commands.add_parser("apply")
    apply.add_argument("--src", type=Path, required=True)
    apply.add_argument("--requests", type=Path, required=True)
    apply.add_argument("--predictions", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--min-score", type=float, default=0.8)
    apply.add_argument("--min-acceptance-rate", type=float, default=0.0)
    apply.add_argument("--target-format", choices=["answer", "response"], default="answer")
    apply.add_argument("--accepted-target-count", type=int)
    apply.add_argument("--selection-seed", type=int, default=0)
    apply.add_argument("--expected-model")
    apply.add_argument("--expected-revision")

    args = parser.parse_args()
    if args.command == "export":
        from datasets import load_from_disk

        result = export_teacher_requests(
            load_from_disk(str(args.src)),
            args.output,
            max_requests=args.max_requests,
            selection_seed=args.selection_seed,
        )
    elif args.command == "generate":
        result = generate_teacher_predictions(
            args.requests,
            args.output,
            model_key=args.model,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            resume=not args.no_resume,
        )
    else:
        from datasets import load_from_disk

        result = apply_teacher_predictions(
            load_from_disk(str(args.src)),
            args.requests,
            args.predictions,
            args.output,
            min_score=args.min_score,
            min_acceptance_rate=args.min_acceptance_rate,
            target_format=args.target_format,
            accepted_target_count=args.accepted_target_count,
            selection_seed=args.selection_seed,
            expected_model=args.expected_model,
            expected_revision=args.expected_revision,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
