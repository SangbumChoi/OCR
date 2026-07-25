#!/usr/bin/env python3
"""Publish a verified Smol pilot evidence handoff as a W&B Artifact."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docvlm_eval.student.pilot_handoff import build_smol_pilot_handoff
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "smol-vision-transfer-pilot-handoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="sbdc")
    parser.add_argument("--project", default="docvlm-ablation")
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
        prefix="docvlm-smol-handoff-publish-"
    ) as temporary:
        plan = compile_sweep_plan(
            args.sweep,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        handoff = build_smol_pilot_handoff(
            plan,
            sweep_root=args.sweep_root,
            output_root=args.output_root,
        )

    import wandb

    digest = handoff["fingerprint"].split(":", 1)[1]
    with wandb.init(
        entity=args.entity,
        project=args.project,
        job_type="execution-evidence-handoff",
        name=f"smol-pilot-handoff-{digest[:12]}",
        config={
            "claim_scope": handoff["claim_scope"],
            "handoff_fingerprint": handoff["fingerprint"],
            "sweep_fingerprint": handoff["sweep_fingerprint"],
        },
    ) as run:
        artifact = wandb.Artifact(
            ARTIFACT,
            type="execution-evidence-handoff",
            metadata={
                "claim_scope": handoff["claim_scope"],
                "handoff_fingerprint": handoff["fingerprint"],
                "sweep_fingerprint": handoff["sweep_fingerprint"],
            },
        )
        artifact.add_dir(handoff["root"])
        logged = run.log_artifact(
            artifact,
            aliases=["latest", f"sha-{digest[:12]}"],
        )
        logged.wait()
        artifact_version = logged.version
    print(
        json.dumps(
            {
                "artifact": (
                    f"{args.entity}/{args.project}/{ARTIFACT}:"
                    f"{artifact_version}"
                ),
                "fingerprint": handoff["fingerprint"],
                "run_count": len(handoff["expected_runs"]),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
