#!/usr/bin/env python3
"""Construct, count, selectively initialize, and optionally save the native sub-1B student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.config import StudentConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--tiny", action="store_true", help="Build a small contract-test model.")
    parser.add_argument(
        "--tiny-vocab-size",
        type=int,
        default=256,
        help="Vocabulary size for --tiny; use at least 260 with a trained byte tokenizer.",
    )
    parser.add_argument("--device", default="meta", choices=["meta", "cpu", "cuda"])
    parser.add_argument("--allow-full-memory", action="store_true")
    parser.add_argument("--init-arm", default="I0_random")
    parser.add_argument("--vision-source", type=Path)
    parser.add_argument("--vision-family", default="student", choices=["student", "siglip"])
    parser.add_argument("--language-source", type=Path)
    parser.add_argument("--language-family", default="student", choices=["student", "llama"])
    parser.add_argument(
        "--token-map",
        type=Path,
        help="JSON mapping of target token IDs to source token IDs for embedding transfer.",
    )
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    import torch

    from docvlm_eval.student.model import DocumentVLMStudent, count_unique_parameters
    from docvlm_eval.student.checkpoint import load_checkpoint_state
    from docvlm_eval.student.transfer import selective_transfer

    blueprint = load_blueprint(args.config)
    estimates, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    config = (
        StudentConfig.tiny(vocab_size=args.tiny_vocab_size)
        if args.tiny
        else StudentConfig.from_blueprint(blueprint)
    )
    if not args.tiny and args.device != "meta" and not args.allow_full_memory:
        raise SystemExit("full construction allocates several GB; pass --allow-full-memory")
    with torch.device(args.device):
        model = DocumentVLMStudent(config)
    counts = count_unique_parameters(model)
    for name, count in counts.items():
        print(f"{name:>10}: {count:>12,} parameters")
    if not args.tiny:
        print(f" estimator: {estimates['total']:>12,} parameters")
        if counts["total"] >= int(blueprint["budget"]["max_parameters"]):
            raise SystemExit("constructed model exceeds the deployment parameter budget")

    arms = {arm["id"]: arm for arm in blueprint["initialization_arms"]}
    if args.init_arm not in arms:
        raise SystemExit(f"unknown initialization arm {args.init_arm!r}")
    arm = arms[args.init_arm]
    reports = []
    if args.device == "meta" and (args.vision_source or args.language_source):
        raise SystemExit("weight transfer requires a materialized cpu/cuda model")
    if args.vision_source:
        reports.append(
            selective_transfer(
                model,
                load_checkpoint_state(args.vision_source),
                {"vision": arm["vision_transfer"]},
                family=args.vision_family,
            ).to_dict()
        )
    if args.language_source:
        token_map = None
        if args.token_map:
            raw_token_map = json.loads(args.token_map.read_text(encoding="utf-8"))
            token_map = {int(target): int(source) for target, source in raw_token_map.items()}
        reports.append(
            selective_transfer(
                model,
                load_checkpoint_state(args.language_source),
                {"language": arm["language_transfer"]},
                family=args.language_family,
                token_map=token_map,
            ).to_dict()
        )
    if args.init_arm != "I0_random":
        required = []
        if arm["vision_transfer"] and not args.vision_source:
            required.append("--vision-source")
        if arm["language_transfer"] and not args.language_source:
            required.append("--language-source")
        if required:
            raise SystemExit(f"{args.init_arm} requires {' and '.join(required)}")

    if args.save:
        if args.device == "meta":
            raise SystemExit("cannot save a meta-device model")
        metadata = {
            "blueprint": str(args.config),
            "initialization_arm": args.init_arm,
            "transfer_reports": reports,
            "parameter_counts": counts,
        }
        model.save_pretrained(args.save, metadata=metadata)
        print(f"Saved {args.save}")
    elif reports:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
