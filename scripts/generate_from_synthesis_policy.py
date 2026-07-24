#!/usr/bin/env python3
"""Execute a training-authorized synthetic generation plan exactly."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from docvlm_eval.student.synthesis_policy import (
    file_fingerprint,
    iter_plan_documents,
    validate_generation_plan,
    validate_generation_plan_source,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


def _composition_tier(gt: dict[str, Any]) -> str:
    render = gt.get("render") or {}
    if int(render.get("document_count") or 1) > 1:
        return "multi_document"
    if int(render.get("rendered_page_count") or 1) > 1:
        return "multi_page"
    return "single_document"


def _verify_document(
    gt_path: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    difficulty = gt.get("difficulty") or {}
    actual = {
        "generator_case": gt.get("generator_case"),
        "language": str((gt.get("languages") or ["und"])[0]),
        "difficulty_level": int(
            difficulty.get("level")
            if isinstance(difficulty, dict)
            else difficulty
        ),
        "layout_family": (gt.get("render") or {}).get("layout_family"),
        "composition_tier": _composition_tier(gt),
    }
    expected = {
        key: job[key]
        for key in (
            "generator_case",
            "language",
            "difficulty_level",
            "layout_family",
            "composition_tier",
        )
    }
    if actual != expected:
        raise RuntimeError(
            "generated document does not match its policy arm: "
            f"expected={expected!r} actual={actual!r} path={gt_path}"
        )
    return {
        **actual,
        "path": str(gt_path),
        "content_fingerprint": (
            (gt.get("semantic_graph") or {}).get("content_fingerprint")
        ),
        "layout_fingerprint": (
            (gt.get("render") or {}).get("layout_fingerprint")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for the exact document generator",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_generation_plan(plan, require_training_authorized=True)
    validate_generation_plan_source(plan)
    if args.out.exists() and (
        not args.out.is_dir() or any(args.out.iterdir())
    ):
        raise RuntimeError(
            f"policy output must be empty to prevent stale documents: {args.out}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    generation = plan.get("generation") or {}
    for ordinal, (job, replica) in enumerate(iter_plan_documents(plan)):
        document_out = (
            args.out / str(job["output_subdir"]) / f"replica-{replica:05d}"
        )
        seed = (int(job["seed"]) + replica) % (2**31 - 1)
        command = [
            args.python,
            str(ROOT / "scripts" / "make_realistic_cases.py"),
            "--config",
            str(args.config),
            "--only",
            str(job["generator_case"]),
            "--count",
            "1",
            "--seed",
            str(seed),
            "--language",
            str(job["language"]),
            "--difficulty-level",
            str(int(job["difficulty_level"])),
            "--split-name",
            "train",
            "--out",
            str(document_out),
        ]
        if job.get("layout_family"):
            command.extend(["--hard-layout", str(job["layout_family"])])
        if bool(generation.get("no_degrade", False)):
            command.append("--no-degrade")
        subprocess.run(command, cwd=ROOT, check=True)
        gt_paths = sorted(document_out.rglob("gt.json"))
        if len(gt_paths) != 1:
            raise RuntimeError(
                f"policy document {ordinal} emitted {len(gt_paths)} gt.json files"
            )
        record = _verify_document(gt_paths[0], job)
        record.update(
            {
                "ordinal": ordinal,
                "arm_id": job["arm_id"],
                "seed": seed,
            }
        )
        records.append(record)

    if len(records) != int(plan["budget"]):
        raise RuntimeError(
            f"generated {len(records)} documents for budget {plan['budget']}"
        )
    execution = {
        "schema_version": 1,
        "status": "pass",
        "split": "train",
        "plan": str(args.plan.resolve()),
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan_file_fingerprint": file_fingerprint(args.plan),
        "config": str(args.config.resolve()),
        "config_fingerprint": file_fingerprint(args.config),
        "documents": len(records),
        "records": records,
    }
    write_json_atomic(args.out / "index.json", execution)
    write_json_atomic(
        args.out / "gen_config.json",
        {
            "schema_version": 1,
            "mode": "failure_driven_policy",
            "plan_fingerprint": plan["plan_fingerprint"],
            "source_split": plan["source"]["split"],
            "training_authorized": plan["training_authorized"],
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "documents": len(records),
                "plan_fingerprint": plan["plan_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
