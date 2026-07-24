#!/usr/bin/env python3
"""Verify and attest one exact parent-to-curriculum-round handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.continuation import (
    materialize_continuation_tokenizer,
    resolve_continuation_contract,
    write_continuation_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--replay-fraction", type=float, required=True)
    parser.add_argument("--replay-seed", type=int, required=True)
    parser.add_argument("--optimizer-policy", required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-output", type=Path, required=True)
    args = parser.parse_args()

    blueprint = load_blueprint(args.blueprint)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    contract = resolve_continuation_contract(
        {
            "enabled": True,
            "parent_root": str(args.parent_root),
            "round_index": args.round_index,
            "optimizer_policy": args.optimizer_policy,
            "replay_fraction": args.replay_fraction,
            "replay_seed": args.replay_seed,
        },
        repo_root=Path.cwd(),
        blueprint=blueprint,
    )
    write_continuation_manifest(contract, args.output)
    materialize_continuation_tokenizer(contract, args.tokenizer_output)


if __name__ == "__main__":
    main()
