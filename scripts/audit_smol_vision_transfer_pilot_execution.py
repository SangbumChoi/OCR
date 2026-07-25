#!/usr/bin/env python3
"""Build the compact observed execution state for the Smol vision pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from docvlm_eval.student.pilot_execution import (
    audit_smol_pilot_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_readiness.json"
        ),
    )
    parser.add_argument(
        "--wandb-snapshot",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "docvlm_ablation_wandb_run_inventory.json"
        ),
    )
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "sweeps"
            / "docvlm-smol-vision-transfer-pilot"
            / "sweep_run_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_execution_state.json"
        ),
    )
    args = parser.parse_args()

    result = audit_smol_pilot_execution(
        _read_json(args.readiness),
        _read_json(args.wandb_snapshot),
        local_summary=(
            _read_json(args.local_summary)
            if args.local_summary.is_file()
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
                "output": str(args.output.resolve()),
                "state": result["state"],
                "training_execution_attested": result[
                    "training_execution_attested"
                ],
                "wandb": result["wandb"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
