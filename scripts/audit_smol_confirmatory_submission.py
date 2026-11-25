#!/usr/bin/env python3
"""Audit whether sealed pilot evidence authorizes the Smol confirmatory sweep."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from docvlm_eval.student.confirmatory_submission import (
    audit_smol_confirmatory_submission,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
    )
    parser.add_argument(
        "--confirmatory-sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_smol_vision_transfer_sweep.yaml",
    )
    parser.add_argument(
        "--pilot-readiness",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_readiness.json"
        ),
    )
    parser.add_argument(
        "--pilot-execution",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_execution_state.json"
        ),
    )
    parser.add_argument(
        "--pilot-comparison",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "sweeps"
            / "docvlm-smol-vision-transfer-pilot"
            / "comparison.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_confirmatory_submission.json"
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="docvlm-smol-submission-"
    ) as temporary:
        compile_root = Path(temporary)
        pilot_plan = compile_sweep_plan(
            args.pilot_sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=compile_root / "pilot",
        )
        confirmatory_plan = compile_sweep_plan(
            args.confirmatory_sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=compile_root / "confirmatory",
        )
        result = audit_smol_confirmatory_submission(
            pilot_plan,
            confirmatory_plan,
            pilot_readiness=_read_json(args.pilot_readiness),
            pilot_execution=_read_json(args.pilot_execution),
            pilot_comparison=(
                _read_json(args.pilot_comparison)
                if args.pilot_comparison.is_file()
                else None
            ),
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
                "authorized": result[
                    "confirmatory_submission_authorized"
                ],
                "counts": result["counts"],
                "fingerprint": result["fingerprint"],
                "output": str(args.output.resolve()),
                "status": result["overall_status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
