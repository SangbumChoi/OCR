#!/usr/bin/env python3
"""Train the student's byte-level multilingual tokenizer from UDD supervision text."""

from __future__ import annotations

import argparse
from pathlib import Path

from docvlm_eval.architecture import load_blueprint
from docvlm_eval.student.tokenizer import DocumentTokenizer, iter_udd_text


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--src",
        type=Path,
        default=ROOT / "data" / "udd" / "hf" / "_all",
        help="Local datasets.load_from_disk UDD path.",
    )
    source.add_argument("--repo", help="Hugging Face UDD repository ID.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "student_tokenizer",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset, load_from_disk

    if args.repo:
        dataset = load_dataset(args.repo, split=args.split)
    else:
        dataset = load_from_disk(str(args.src))
    blueprint = load_blueprint(args.config)
    model_vocab_size = int(blueprint["student"]["language"]["vocab_size"])
    vocab_size = args.vocab_size or model_vocab_size
    if vocab_size > model_vocab_size:
        raise SystemExit(
            f"tokenizer vocab_size={vocab_size} exceeds model vocab_size={model_vocab_size}"
        )
    tokenizer = DocumentTokenizer.train(
        iter_udd_text(dataset),
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        show_progress=not args.no_progress,
    )
    tokenizer.save_pretrained(args.output)
    print(
        f"Saved byte-level tokenizer with {tokenizer.vocab_size:,} tokens "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
