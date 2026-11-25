#!/usr/bin/env python3
"""Build a compact fail-closed readiness audit for the LFM transfer pilot."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.sweep import compile_sweep_plan
from docvlm_eval.student.transfer_readiness import audit_lfm_transfer_pilot


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_lfm_language_transfer_pilot.yaml",
    )
    parser.add_argument(
        "--preflight",
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
            / "lfm_selective_transfer_pilot_readiness.json"
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="docvlm-lfm-readiness-") as temp:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temp,
        )
        result = audit_lfm_transfer_pilot(
            plan,
            repo_root=ROOT,
            sweep_path=args.sweep,
            preflight_path=args.preflight,
            source_selection_path=args.source_selection,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_status": result["overall_status"],
                "pilot_submission_authorized": result[
                    "pilot_submission_authorized"
                ],
                "checks": result["counts"],
                "output": str(args.output.resolve()),
                "fingerprint": result["fingerprint"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    if result["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
