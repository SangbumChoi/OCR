#!/usr/bin/env python3
"""Run exact, attested failure-driven student curriculum rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import yaml

from docvlm_eval.student.continuation import (
    prepare_initial_round_spec,
    prepare_next_round_spec,
    write_round_spec,
)
from docvlm_eval.student.evidence import build_experiment_attestation
from docvlm_eval.student.experiment import ExperimentRunner, build_experiment_plan
from docvlm_eval.student.synthesis_policy import (
    file_fingerprint,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_immutable_spec(spec: dict, path: Path) -> None:
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != spec:
            raise RuntimeError(
                f"existing curriculum round spec differs: {path}"
            )
        return
    write_round_spec(spec, path)


def _round_record(round_index: int, plan, attestation: dict) -> dict:
    root = Path(plan.root)
    synthesis_plan = (
        root / "artifacts" / "synthetic" / "next_train_plan.json"
    )
    record = {
        "round_index": round_index,
        "experiment": plan.name,
        "root": str(root),
        "experiment_fingerprint": plan.fingerprint,
        "attestation": str(root / "evidence_attestation.json"),
        "attestation_sha256": attestation["attestation_sha256"],
        "contract_status": attestation["contract_status"],
        "capability_status": attestation["capability_status"],
        "synthesis_plan": str(synthesis_plan),
        "synthesis_plan_file_fingerprint": file_fingerprint(synthesis_plan),
        "stage_count": len(plan.stages),
    }
    replay_manifest_path = (
        root
        / "artifacts"
        / "continuation"
        / "train_with_replay.manifest.json"
    )
    if replay_manifest_path.is_file():
        replay = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        record["replay"] = {
            "manifest": str(replay_manifest_path),
            "manifest_fingerprint": replay["manifest_fingerprint"],
            "active_sample_count": replay["output_sample_count"],
            "selected_replay_count": replay["selected_replay_count"],
            "selected_replay_origin_counts": replay[
                "selected_replay_origin_counts"
            ],
            "memory_sample_count": replay["memory_sample_count"],
            "memory_origin_counts": replay["memory_origin_counts"],
            "memory_fingerprint": replay["memory_output"]["fingerprint"],
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "configs" / "sub1b_experiment.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    parser.add_argument("--replay-seed-base", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if not 0.0 <= args.replay_fraction < 1.0:
        parser.error("--replay-fraction must be within [0, 1)")
    if args.replay_seed_base < 0:
        parser.error("--replay-seed-base must be non-negative")

    curriculum_root = args.output_root.resolve()
    round_zero_root = curriculum_root / "round-000"
    round_zero_spec = prepare_initial_round_spec(
        experiment=args.experiment,
        output_root=round_zero_root,
    )
    if args.dry_run:
        with tempfile.TemporaryDirectory(
            prefix="docvlm-curriculum-dry-run-"
        ) as temporary:
            config = Path(temporary) / "round-000.yaml"
            write_round_spec(round_zero_spec, config)
            plan = build_experiment_plan(
                config,
                repo_root=ROOT,
                python=sys.executable,
            )
            result = ExperimentRunner(plan, repo_root=ROOT).run(
                dry_run=True
            )
        result["projected_rounds"] = args.rounds
        result["future_rounds_require_parent_attestation"] = max(
            0,
            args.rounds - 1,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    specs_root = curriculum_root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    round_records = []
    parent_root: Path | None = None
    for round_index in range(args.rounds):
        round_root = curriculum_root / f"round-{round_index:03d}"
        config = specs_root / f"round-{round_index:03d}.yaml"
        if round_index == 0:
            spec = round_zero_spec
        else:
            assert parent_root is not None
            spec = prepare_next_round_spec(
                parent_root=parent_root,
                output_root=round_root,
                round_index=round_index,
                replay_fraction=args.replay_fraction,
                replay_seed=args.replay_seed_base + round_index,
            )
        _write_immutable_spec(spec, config)
        plan = build_experiment_plan(
            config,
            repo_root=ROOT,
            python=sys.executable,
        )
        result = ExperimentRunner(plan, repo_root=ROOT).run(
            resume=not args.no_resume,
        )
        if result["pipeline_complete"] is not True:
            raise RuntimeError(
                f"curriculum round {round_index} did not complete"
            )
        attestation_path = round_root / "evidence_attestation.json"
        attestation = build_experiment_attestation(
            plan,
            repo_root=ROOT,
            output=attestation_path,
            hash_mode="full",
        )
        if attestation["contract_status"] != "pass":
            raise RuntimeError(
                f"curriculum round {round_index} failed execution attestation"
            )
        round_records.append(
            _round_record(round_index, plan, attestation)
        )
        parent_root = round_root

    summary = {
        "schema_version": 2,
        "policy": (
            "attested_failure_driven_curriculum_with_cumulative_replay"
        ),
        "requested_rounds": args.rounds,
        "completed_rounds": len(round_records),
        "replay_fraction": args.replay_fraction,
        "replay_seed_base": args.replay_seed_base,
        "rounds": round_records,
        "final_round_root": round_records[-1]["root"],
        "final_experiment_fingerprint": round_records[-1][
            "experiment_fingerprint"
        ],
    }
    summary["curriculum_fingerprint"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                summary,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    write_json_atomic(curriculum_root / "curriculum_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
