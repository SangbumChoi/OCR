#!/usr/bin/env python3
"""Create or verify a deterministic native-student experiment attestation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docvlm_eval.student.evidence import (
    build_experiment_attestation,
    verify_experiment_attestation,
)
from docvlm_eval.student.experiment import build_experiment_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "configs" / "sub1b_experiment.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--hash-mode",
        choices=("full", "metadata"),
        default="full",
        help="Full hashes every evidence file; metadata omits hashes above 1 MiB.",
    )
    args = parser.parse_args()
    if args.output is not None and args.verify is not None:
        parser.error("--output and --verify are mutually exclusive")

    plan = build_experiment_plan(
        args.experiment,
        repo_root=ROOT,
        python=sys.executable,
    )
    output = args.output or Path(plan.root) / "evidence_attestation.json"
    if args.verify is not None:
        result = verify_experiment_attestation(
            plan,
            args.verify,
            repo_root=ROOT,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not result["valid"]:
            raise SystemExit(1)
        return

    result = build_experiment_attestation(
        plan,
        repo_root=ROOT,
        output=output,
        hash_mode=args.hash_mode,
    )
    print(
        json.dumps(
            {
                "path": str(output.resolve()),
                "attestation_sha256": result["attestation_sha256"],
                "contract_status": result["contract_status"],
                "capability_status": result["capability_status"],
                "claim_scope": result["claim_scope"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
