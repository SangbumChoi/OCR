#!/usr/bin/env python3
"""Audit the complete sub-1B document-VLM objective against current evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.goal_readiness import (
    audit_end_to_end_goal_readiness,
)
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
        "--public-dataset-readiness",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "public_udd_training_readiness.json"
        ),
    )
    parser.add_argument(
        "--execution-state",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_transfer_pilot_execution_state.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "end_to_end_goal_readiness.json"
        ),
    )
    parser.add_argument("--quality-evidence", type=Path)
    parser.add_argument("--promotion-evidence", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="docvlm-goal-readiness-"
    ) as temporary:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        result = audit_end_to_end_goal_readiness(
            plan,
            repo_root=ROOT,
            method_catalog_path=(
                ROOT / "configs" / "frontier_method_catalog.jsonl"
            ),
            method_evidence_path=(
                ROOT / "configs" / "frontier_method_evidence.yaml"
            ),
            synth_config_path=ROOT / "configs" / "synth_data.yaml",
            public_dataset_readiness_path=args.public_dataset_readiness,
            vision_preflight_path=(
                ROOT
                / "docs"
                / "results"
                / "selective_transfer_smol_vision_real_source_preflight.json"
            ),
            language_preflight_path=(
                ROOT
                / "docs"
                / "results"
                / "selective_transfer_lfm_real_source_preflight.json"
            ),
            pilot_readiness_path=(
                ROOT
                / "docs"
                / "results"
                / "smol_vision_transfer_pilot_readiness.json"
            ),
            execution_state_path=args.execution_state,
            quality_evidence_path=args.quality_evidence,
            promotion_evidence_path=args.promotion_evidence,
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
                "counts": result["counts"],
                "fingerprint": result["fingerprint"],
                "goal_complete": result["goal_complete"],
                "next_required_evidence": result[
                    "next_required_evidence"
                ],
                "output": str(args.output.resolve()),
                "overall_status": result["overall_status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
