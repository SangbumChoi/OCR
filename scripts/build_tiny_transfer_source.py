#!/usr/bin/env python3
"""Build a deterministic cross-architecture checkpoint for CPU transfer smoke tests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from docvlm_eval.student.config import StudentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--vision-layers", type=int, required=True)
    parser.add_argument("--language-layers", type=int, required=True)
    parser.add_argument("--language-mlp-width", type=int, required=True)
    args = parser.parse_args()
    if (
        args.seed < 0
        or args.vocab_size < 260
        or args.vision_layers <= 0
        or args.language_layers <= 0
        or args.language_mlp_width <= 0
    ):
        raise SystemExit("fixture dimensions and seed are invalid")

    import torch

    from docvlm_eval.student.model import (
        DocumentVLMStudent,
        count_unique_parameters,
    )

    torch.manual_seed(args.seed)
    base = StudentConfig.tiny(vocab_size=args.vocab_size)
    config = replace(
        base,
        vision=replace(base.vision, layers=args.vision_layers),
        language=replace(
            base.language,
            layers=args.language_layers,
            mlp_width=args.language_mlp_width,
        ),
    )
    errors = config.validate()
    if errors:
        raise SystemExit("\n".join(errors))
    model = DocumentVLMStudent(config)
    metadata = {
        "artifact_scope": "cross_architecture_transfer_contract_only",
        "quality_claim_authorized": False,
        "initialization_arm": "fixture_random_source",
        "initialization_seed": args.seed,
        "transfer_reports": [],
        "parameter_counts": count_unique_parameters(model),
        "source_architecture": config.to_dict(),
    }
    model.save_pretrained(args.output, metadata=metadata)
    print(
        f"Saved deterministic transfer fixture to {args.output} "
        f"({metadata['parameter_counts']['total']} parameters)"
    )


if __name__ == "__main__":
    main()
