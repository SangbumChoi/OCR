#!/usr/bin/env python3
"""Restore a verified Smol pilot evidence handoff from W&B."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.pilot_handoff import restore_smol_pilot_handoff
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "smol-vision-transfer-pilot-handoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="sbdc")
    parser.add_argument("--project", default="docvlm-ablation")
    parser.add_argument("--alias", default="latest")
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
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    artifact_name = (
        f"{args.entity}/{args.project}/{ARTIFACT}:{args.alias}"
    )
    artifact = api.artifact(
        artifact_name,
        type="execution-evidence-handoff",
    )
    with tempfile.TemporaryDirectory(
        prefix="docvlm-smol-handoff-restore-"
    ) as temporary:
        download_root = Path(
            artifact.download(root=Path(temporary) / "artifact")
        )
        compile_root = Path(temporary) / "compiled"
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=compile_root,
        )
        result = restore_smol_pilot_handoff(
            download_root,
            plan=plan,
            sweep_root=args.sweep_root,
        )
    print(
        json.dumps(
            {
                "artifact": artifact_name,
                "fingerprint": result["fingerprint"],
                "restored": result["restored"],
                "reused": result["reused"],
                "sweep_root": result["sweep_root"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
