#!/usr/bin/env python3
"""Build a compact content-addressed Smol pilot evidence handoff."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.pilot_handoff import build_smol_pilot_handoff
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "sweeps"
            / "docvlm-smol-vision-transfer-pilot"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "handoffs",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="docvlm-smol-handoff-build-"
    ) as temporary:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        result = build_smol_pilot_handoff(
            plan,
            sweep_root=args.sweep_root,
            output_root=args.output_root,
        )
    print(
        json.dumps(
            {
                "fingerprint": result["fingerprint"],
                "reused": result["reused"],
                "root": result["root"],
                "run_count": len(result["expected_runs"]),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
