#!/usr/bin/env python3
"""Build compact Smol quality and promotion evidence after the sealed sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.confirmatory_evidence import (
    build_confirmatory_evidence,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
SWEEP = "docvlm-smol-vision-transfer-sweep"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=ROOT / "configs" / "sub1b_smol_vision_transfer_sweep.yaml",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "sweeps"
            / SWEEP
            / "comparison.json"
        ),
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_heldout_quality_evidence.json"
        ),
    )
    parser.add_argument(
        "--promotion-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "smol_vision_multi_seed_promotion_evidence.json"
        ),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="docvlm-smol-confirmatory-"
    ) as compile_root:
        plan = compile_sweep_plan(
            args.sweep_config,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=compile_root,
        )
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    quality, promotion = build_confirmatory_evidence(
        comparison,
        expected_sweep=SWEEP,
        expected_sweep_fingerprint=plan.fingerprint,
        baseline="lfm_language_only",
        candidate="lfm_smol_dual",
    )
    _atomic_write(args.quality_output, quality)
    _atomic_write(args.promotion_output, promotion)
    print(
        json.dumps(
            {
                "promotion": {
                    "authorized": promotion[
                        "promotion_claim_authorized"
                    ],
                    "fingerprint": promotion["fingerprint"],
                    "output": str(args.promotion_output.resolve()),
                },
                "quality": {
                    "authorized": quality[
                        "quality_claim_authorized"
                    ],
                    "fingerprint": quality["fingerprint"],
                    "output": str(args.quality_output.resolve()),
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
