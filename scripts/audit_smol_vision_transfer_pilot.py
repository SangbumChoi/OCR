#!/usr/bin/env python3
"""Build the fail-closed readiness audit for the Smol vision pilot."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.sweep import compile_sweep_plan
from docvlm_eval.student.transfer_readiness import (
    audit_smol_vision_transfer_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
    )
    parser.add_argument(
        "--vision-preflight",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_smol_vision_real_source_preflight.json"
        ),
    )
    parser.add_argument(
        "--language-preflight",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_lfm_real_source_preflight.json"
        ),
    )
    parser.add_argument(
        "--source-selection",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_source_matrix.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_readiness.json"
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="docvlm-smol-readiness-"
    ) as temporary:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        result = audit_smol_vision_transfer_pilot(
            plan,
            repo_root=ROOT,
            sweep_path=args.sweep,
            vision_preflight_path=args.vision_preflight,
            language_preflight_path=args.language_preflight,
            source_selection_path=args.source_selection,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checks": result["counts"],
                "fingerprint": result["fingerprint"],
                "output": str(args.output.resolve()),
                "overall_status": result["overall_status"],
                "pilot_submission_authorized": result[
                    "pilot_submission_authorized"
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    if result["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
